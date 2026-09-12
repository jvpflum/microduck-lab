#!/usr/bin/env python3
"""Render the DuckWing Markdown technical report as a shareable Word file."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown


STYLE = """
@page { size: Letter; margin: 0.8in; }
body { font-family: 'Liberation Sans', Arial, sans-serif; color: #17202a;
       font-size: 10.5pt; line-height: 1.38; max-width: 7in; margin: auto; }
h1 { color: #172554; font-size: 24pt; margin-bottom: 8pt; }
h2 { color: #1e3a8a; font-size: 16pt; border-bottom: 1px solid #cbd5e1;
     padding-bottom: 3pt; margin-top: 22pt; }
h3 { color: #334155; font-size: 12pt; margin-top: 16pt; }
p { margin: 6pt 0 9pt; }
table { border-collapse: collapse; width: 100%; margin: 12pt 0; font-size: 9pt; }
th { background: #e8eefc; color: #172554; }
th, td { border: 1px solid #aab4c3; padding: 5pt; text-align: left; }
code { font-family: 'Liberation Mono', monospace; background: #f1f5f9; }
pre { background: #f1f5f9; border: 1px solid #cbd5e1; padding: 8pt;
      white-space: pre-wrap; font-size: 8.5pt; }
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    body = markdown.markdown(
        source.read_text(), extensions=("tables", "fenced_code", "sane_lists")
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        + STYLE + "</style></head><body>" + body + "</body></html>"
    )
    with tempfile.TemporaryDirectory(prefix="duckwing-paper-") as directory:
        temporary = Path(directory)
        html_path = temporary / f"{output.stem}.html"
        html_path.write_text(html)
        profile = temporary / "libreoffice-profile"
        result = subprocess.run(
            [
                "libreoffice",
                f"-env:UserInstallation={profile.as_uri()}",
                "--headless",
                "--convert-to", "docx:Office Open XML Text",
                "--outdir", str(temporary),
                str(html_path),
            ],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if result.returncode:
            raise SystemExit(f"LibreOffice conversion failed: {result.stdout.strip()}")
        rendered = temporary / f"{output.stem}.docx"
        if not rendered.is_file():
            raise SystemExit("LibreOffice did not produce the expected DOCX")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rendered, output)
    print(output)


if __name__ == "__main__":
    main()
