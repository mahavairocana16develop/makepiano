"""MusicXML -> SVG pages (verovio) -> PDF (cairosvg)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import verovio


def render_svgs(xml_path: Path, out_dir: Path, log=print, timemap_out: Path | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tk = verovio.toolkit(False)
    # The default resource path set at import time is not visible from worker threads (web UI).
    tk.setResourcePath(str(Path(verovio.__file__).parent / "data"))
    tk.setOptions({
        "pageHeight": 2970,
        "pageWidth": 2100,
        "pageMarginTop": 100,
        "pageMarginBottom": 100,
        "pageMarginLeft": 100,
        "pageMarginRight": 100,
        "scale": 36,
        "spacingSystem": 8,
        "spacingStaff": 10,
        "adjustPageHeight": False,
        "breaks": "auto",
        "header": "auto",
        "footer": "auto",
        "svgViewBox": True,
    })
    if not tk.loadFile(str(xml_path)):
        raise RuntimeError("verovio failed to load MusicXML")
    pages = []
    for i in range(1, tk.getPageCount() + 1):
        p = out_dir / f"page-{i:03d}.svg"
        p.write_text(tk.renderToSVG(i), encoding="utf-8")
        pages.append(p)
    if timemap_out is not None:
        import json
        tm = tk.renderToTimemap()
        timemap_out.write_text(tm if isinstance(tm, str) else json.dumps(tm), encoding="utf-8")
    log(f"[render] {len(pages)} page(s) of SVG")
    return pages


def _cairo_env() -> dict[str, str]:
    env = dict(os.environ)
    for cand in ("/opt/homebrew/lib", "/usr/local/lib"):
        if Path(cand).exists():
            env["DYLD_FALLBACK_LIBRARY_PATH"] = cand + ":" + env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
            break
    return env


# cairosvg has no font fallback, so the text font must itself cover CJK for Japanese titles.
# Apple's Hiragino .ttc collections make cairo fail with "out of memory", so they are not listed.
PDF_FONT_CANDIDATES = ["Arial Unicode MS", "Hiragino Sans GB", "Noto Serif CJK JP", "Noto Sans CJK JP",
                       "Yu Mincho", "MS Mincho", "Times New Roman"]


def _pdf_font() -> str:
    try:
        families = subprocess.run(["fc-list", ":", "family"], capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        return "Times, serif"
    for cand in PDF_FONT_CANDIDATES:
        if cand in families:
            return cand
    return "Times, serif"


def svgs_to_pdf(svgs: list[Path], pdf_out: Path, log=print) -> Path | None:
    """Merge SVG pages into one PDF. Runs in a subprocess so Homebrew's cairo is found on macOS."""
    font = _pdf_font()
    code = (
        "import sys, io, re, cairosvg\n"
        "from pypdf import PdfReader, PdfWriter\n"
        "font, svgs, out = sys.argv[1], sys.argv[2:-1], sys.argv[-1]\n"
        "w = PdfWriter()\n"
        "for s in svgs:\n"
        "    data = open(s, 'rb').read().replace(b'font-family=\"Times, serif\"', ('font-family=\"%s\"' % font).encode()).decode('utf-8')\n"
        "    acc = {'\\uea64': '\\u266d', '\\uea65': '\\u266e', '\\uea66': '\\u266f'}  # verovio figbass glyphs -> flat/natural/sharp\n"
        "    data = re.sub(r'<tspan font-family=\"Leipzig\" font-size=\"(\\d+)px\">(.)</tspan>',\n"
        "                  lambda m: '<tspan font-size=\"%dpx\">%s</tspan>' % (int(m.group(1)) * 0.6, acc.get(m.group(2), m.group(2))), data)\n"
        "    data = data.encode('utf-8')\n"
        "    w.append(PdfReader(io.BytesIO(cairosvg.svg2pdf(bytestring=data))))\n"
        "w.write(out)\n"
    )
    r = subprocess.run([sys.executable, "-c", code, font, *map(str, svgs), str(pdf_out)],
                       env=_cairo_env(), capture_output=True, text=True)
    if r.returncode != 0:
        log(f"[render] PDF generation failed (SVG is still available):\n{r.stderr.strip()[-800:]}")
        return None
    log(f"[render] wrote {pdf_out.name}")
    return pdf_out
