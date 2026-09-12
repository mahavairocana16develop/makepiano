# makepiano

YouTube の URL からピアノ譜（PDF / MIDI / MusicXML / SVG）を自動生成するツール。
[klang.io](https://studio.klang.io/) のようなことをローカルで行います。

## 仕組み

1. **yt-dlp** で YouTube から音声を取得
2. （任意）**Demucs** で音源分離し、ボーカル・ドラム・ベースを除去 or ピアノ成分だけ抽出
3. **ByteDance の piano_transcription_inference**（ピアノ専用の高精度モデル）で音声 → MIDI
4. **librosa** でビート追跡してテンポ推定・小節割り（テンポ揺れにも追従）
5. **music21** で左手/右手に分割・クオンタイズ・調判定して MusicXML 化
6. **verovio** で楽譜を SVG にレンダリング、**cairosvg** で PDF 化

MuseScore などの外部アプリは不要です。

## セットアップ

```bash
brew install ffmpeg cairo   # 未インストールの場合
uv sync
```

初回実行時に採譜モデル（約172 MB）を `~/piano_transcription_inference_data/` にダウンロードします。

## 使い方

### CLI

```bash
uv run makepiano "https://www.youtube.com/watch?v=..."
```

出力は `output/<動画タイトル>/` に保存されます:

| ファイル | 内容 |
|---|---|
| `score.pdf` | ピアノ譜 |
| `score.musicxml` | MuseScore 等で編集可能な楽譜データ |
| `transcription.mid` | 採譜結果の MIDI（演奏タイミングそのまま） |
| `svg/page-NNN.svg` | ページごとの楽譜画像 |
| `source.wav` | ダウンロードした音声（`rescore` 用） |

主なオプション:

| オプション | 説明 |
|---|---|
| `--stem other` | 音源分離してから採譜（下記） |
| `--bpm 120` | ビート追跡を使わず固定テンポにする |
| `--beats-per-bar 3` | 拍子（3 = 3/4） |
| `--grid 2` | 最小音価（2=8分, 4=16分, 3=3連） |
| `--split 60` | 左右の手を分ける MIDI ノート番号（60 = 中央ド） |
| `--beat-offset 1` | 小節線を N 拍ずらす（小節の頭がずれているとき） |
| `--min-velocity 25` | この音量未満の音を捨てる |
| `--max-pitch 96` | この高さ（MIDI ノート番号）より上の音を捨てる。歌・シンバルの混入対策 |
| `--min-note-ms 60` | これより短い音を捨てる（採譜ノイズ対策） |
| `--no-legato` | 音を次の和音まで伸ばさず、実測の長さで書く |
| `--no-chords` | コードネームを付けない |

### ボーカル入り・バンド音源から伴奏を取り出す

```bash
uv run makepiano "https://www.youtube.com/watch?v=..." --stem other
```

| `--stem` | 内容 | 向いている音源 |
|---|---|---|
| `none`（既定） | 分離しない | ピアノソロ |
| `other` | ボーカル・ドラム・ベースを除去した残り（htdemucs） | ピアノ伴奏の弾き語り・バンド曲。まずはこれ |
| `piano` | ピアノ成分だけ抽出（htdemucs_6s） | ギターやシンセも混ざるとき。分離品質はやや不安定 |
| `no_vocals` | ボーカルだけ除去 | ピアノ＋歌 |

初回はモデル（80〜90 MB）をダウンロードします。分離は CPU で曲の長さの 1〜2 倍程度かかります。
分離後の音声は `source.wav`（元ミックスは `source_mix.wav`）として保存されます。

採譜をやり直さずに、設定を変えて楽譜だけ作り直す:

```bash
uv run makepiano rescore "output/<動画タイトル>" --beats-per-bar 3 --grid 2
```

### ローカルの動画・音声ファイルから

画面収録（.mov / .mp4）や音声ファイルもそのまま渡せます。

```bash
uv run makepiano ~/Desktop/画面収録.mov --stem other
```

### Web UI

```bash
uv run makepiano-web
```

http://127.0.0.1:8000 を開き、URL を貼って「楽譜を作る」。
PDF / MIDI / MusicXML のダウンロードと、次のプレビュー・再生機能が使えます。

- **シートビュー**：楽譜。再生中は鳴っている音符が赤くハイライトされ、自動スクロールします
- **ピアノビュー**：88 鍵の鍵盤と落下ノート表示（右手＝青、左手＝緑）。楽譜と同じ量子化済みノートを表示します
- **再生モード**：「オリジナル サウンド」（元音声）、「分離後の伴奏」（音源分離した場合）、「シンセサイザー サウンド」（採譜結果をブラウザ内のピアノ音源で演奏）
- シンセ再生は既定で採譜した演奏そのもの（実タイミング・強弱・ペダル保持）を鳴らします。「楽譜どおりに再生」で量子化済みの楽譜を固定テンポで機械的に鳴らす Klangio 風にも切り替え可能
- 再生速度 0.5x〜1.25x、シーク、Space キーで再生/停止

再生用データは `playback.json`（ノート一覧と、楽譜要素↔秒のタイムマップ）と `audio_original.m4a` / `audio_stem.m4a` に保存されます。

## 制限・注意

- ピアノ専用モデルのため、歌やバンド音源はそのままだと精度が落ちます。`--stem other` で分離してから採譜してください。
- 拍子は自動判定しません（デフォルト 4/4）。ワルツなどは `--beats-per-bar 3` を指定してください。
- ルバートの強い演奏ではリズムが不正確になります。`--bpm` で固定テンポにすると改善することがあります。
- 各手は「同時に鳴り始めた音 = 和音」として単声で記譜し、既定では次の和音まで音を伸ばします（ポップス譜風）。
- コードネームは構成音からの推定です。テンションや転回形は簡略化されます。
- 処理時間は Apple Silicon の CPU で 4 分の曲あたり 1 分程度です。
