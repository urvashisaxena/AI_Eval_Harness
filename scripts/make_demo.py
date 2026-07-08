#!/usr/bin/env python3
"""Render the evalharness demo video (examples/demo.mp4 + examples/demo.gif).

Runs the real CLI against the bundled example dataset in a temp registry,
captures the actual output, and renders it as an animated terminal session
(typing effect + line-by-line output) using Pillow, encoded to MP4 via
imageio-ffmpeg and to GIF via Pillow.

Usage:  python scripts/make_demo.py
Deps:   pip install pillow imageio imageio-ffmpeg
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# ---- appearance -------------------------------------------------------------
COLS, ROWS = 96, 28
FONT_SIZE = 15
PAD, BAR_H = 14, 30
FPS = 10

BG = (17, 19, 27)
BAR = (32, 35, 46)
FG = (208, 212, 224)
DIM = (130, 136, 152)
GREEN = (117, 200, 128)
RED = (235, 108, 108)
YELLOW = (226, 192, 100)
CYAN = (110, 190, 220)
PROMPT = (117, 200, 128)


def line_color(text: str):
    if "[FAIL]" in text or "FAIL" in text or "❌" in text or "✗" in text:
        return RED
    if "[WARN]" in text or "⚠" in text:
        return YELLOW
    if ("[OK" in text or "PASSED" in text or "✅" in text or "PASS" in text
            or "Baseline set" in text):
        return GREEN
    if text.startswith("Run ") or text.startswith("Wrote") or text.startswith("Comparison"):
        return CYAN
    if re.match(r"^\s{2}\w+\s+\d", text):  # metric rows
        return FG
    return DIM


class Terminal:
    """Accumulates styled lines and renders frames."""

    def __init__(self):
        self.font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        self.bold = ImageFont.truetype(FONT_BOLD, FONT_SIZE)
        bbox = self.font.getbbox("M")
        self.cw, self.lh = bbox[2] - bbox[0], FONT_SIZE + 6
        self.width = COLS * self.cw + 2 * PAD
        self.height = ROWS * self.lh + 2 * PAD + BAR_H
        self.lines: list[tuple[str, tuple]] = []  # (text, color)
        self.frames: list[tuple[Image.Image, int]] = []  # (frame, duration_ms)

    # -- content ----------------------------------------------------------
    def add_line(self, text: str, color=None):
        for chunk in _wrap(text, COLS) or [""]:
            self.lines.append((chunk, color or line_color(text)))

    def type_command(self, cmd: str):
        """Typing animation for `$ cmd` (several chars per frame)."""
        shown = ""
        step = 3
        for i in range(0, len(cmd) + 1, step):
            shown = cmd[: i + 1]
            self._push_prompt_frame(shown, cursor=True, ms=45)
        self._push_prompt_frame(cmd, cursor=False, ms=350)
        # commit the finished command line(s) into the scrollback
        for j, chunk in enumerate(_wrap("$ " + cmd, COLS)):
            self.lines.append((chunk, FG if j else PROMPT))

    def emit_output(self, text: str, lines_per_frame: int = 2, ms: int = 120):
        pending = []
        for raw in text.rstrip("\n").splitlines():
            pending.extend(_wrap(raw, COLS) or [""])
        buf = []
        for ln in pending:
            buf.append(ln)
            if len(buf) >= lines_per_frame:
                for b in buf:
                    self.lines.append((b, line_color(b)))
                buf = []
                self.snapshot(ms)
        for b in buf:
            self.lines.append((b, line_color(b)))
        self.snapshot(ms)

    def pause(self, ms: int):
        self.snapshot(ms)

    # -- rendering ----------------------------------------------------------
    def _base(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGB", (self.width, self.height), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, self.width, BAR_H], fill=BAR)
        for i, c in enumerate([(236, 106, 95), (245, 191, 79), (98, 197, 84)]):
            d.ellipse([12 + i * 22, 9, 24 + i * 22, 21], fill=c)
        title = "evalharness — AI eval / governance harness demo"
        d.text((self.width // 2 - len(title) * self.cw // 2, 7), title,
               font=self.font, fill=DIM)
        return img, d

    def _draw_lines(self, d, visible):
        y = BAR_H + PAD
        for text, color in visible:
            d.text((PAD, y), text, font=self.font, fill=color)
            y += self.lh

    def snapshot(self, ms: int = 100):
        img, d = self._base()
        self._draw_lines(d, self.lines[-ROWS:])
        self.frames.append((img, ms))

    def _push_prompt_frame(self, partial_cmd: str, cursor: bool, ms: int):
        img, d = self._base()
        prompt_lines = _wrap("$ " + partial_cmd + ("█" if cursor else ""), COLS)
        visible = (self.lines + [(pl, FG) for pl in prompt_lines])[-ROWS:]
        # recolor the "$" of the last prompt
        self._draw_lines(d, visible)
        # overdraw prompt glyph in green on the first prompt line if visible
        idx = len(visible) - len(prompt_lines)
        if 0 <= idx < ROWS:
            d.text((PAD, BAR_H + PAD + idx * self.lh), "$", font=self.bold, fill=PROMPT)
        self.frames.append((img, ms))


# DejaVu Sans Mono lacks emoji; substitute glyphs it does have.
_TRANSLIT = {"❌": "✗", "✅": "✓", "⚠️": "!", "⚠": "!", "📈": "↑", "█": "█"}


def _clean(text: str) -> str:
    for k, v in _TRANSLIT.items():
        text = text.replace(k, v)
    return text


def _wrap(text: str, cols: int) -> list[str]:
    text = _clean(text)
    if len(text) <= cols:
        return [text]
    out = []
    while text:
        out.append(text[:cols])
        text = text[cols:]
        if text:
            text = "    " + text  # continuation indent
    return out


def run_cli(args: list[str], cwd: Path) -> tuple[str, int]:
    proc = subprocess.run(
        [sys.executable, "-m", "evalharness.cli", *args],
        cwd=cwd, capture_output=True, text=True,
    )
    return (proc.stdout + proc.stderr), proc.returncode


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="evalharness-demo-"))
    home = tmp / ".evalharness"
    term = Terminal()

    term.add_line("# evalharness — model scorecards with LLM-as-judge, RAGAS metrics,", CYAN)
    term.add_line("# hallucination rate, and regression gating.", CYAN)
    term.add_line("")
    term.snapshot(1400)

    steps: list[tuple[str, list[str], str | None]] = [
        (
            "evalharness run --dataset examples/dataset.jsonl "
            "--predictions examples/predictions_v1.jsonl --label v1 "
            "--config configs/ci-heuristic.yaml --set-baseline",
            ["run", "--dataset", "examples/dataset.jsonl",
             "--predictions", "examples/predictions_v1.jsonl", "--label", "v1",
             "--config", "configs/ci-heuristic.yaml",
             "--home", str(home), "--set-baseline"],
            "score v1 and promote it to baseline",
        ),
        (
            "evalharness run --dataset examples/dataset.jsonl "
            "--predictions examples/predictions_v2.jsonl --label v2 "
            "--config configs/ci-heuristic.yaml --gate",
            ["run", "--dataset", "examples/dataset.jsonl",
             "--predictions", "examples/predictions_v2.jsonl", "--label", "v2",
             "--config", "configs/ci-heuristic.yaml",
             "--home", str(home), "--gate"],
            "v2 hallucinates → the governance gate fails (exit 1)",
        ),
    ]

    last_rc = 0
    for cmd, args, comment in steps:
        if comment:
            term.add_line("")
            term.add_line(f"# {comment}", DIM)
            term.snapshot(900)
        term.type_command(cmd)
        out, last_rc = run_cli(args, ROOT)
        term.emit_output(out.replace(str(home), ".evalharness"))
        term.pause(1600)

    term.add_line("")
    term.add_line(f"# exit code: {last_rc} — CI blocks the regression", RED)
    term.snapshot(1200)

    # scorecard step (find the v2 run id from the registry)
    v2_id = sorted(p.stem for p in (home / "runs").glob("*.json"))[-1]
    cmd = f"evalharness scorecard {v2_id} -o scorecard.md --html scorecard.html"
    term.add_line("")
    term.add_line("# render the scorecard (markdown + HTML)", DIM)
    term.snapshot(900)
    term.type_command(cmd)
    out, _ = run_cli(
        ["scorecard", v2_id, "--config", "configs/ci-heuristic.yaml",
         "--home", str(home), "-o", str(tmp / "scorecard.md"),
         "--html", str(tmp / "scorecard.html")],
        ROOT,
    )
    term.emit_output(out.replace(str(tmp) + "/", ""))
    term.pause(800)

    term.type_command(f"head -22 scorecard.md")
    md_head = "\n".join((tmp / "scorecard.md").read_text().splitlines()[:22])
    term.emit_output(md_head, lines_per_frame=3, ms=100)
    term.pause(3000)

    # ---- encode -----------------------------------------------------------
    out_dir = ROOT / "examples"
    frames = term.frames

    import imageio.v2 as imageio
    import numpy as np

    mp4_path = out_dir / "demo.mp4"
    with imageio.get_writer(mp4_path, fps=FPS, codec="libx264",
                            quality=8, macro_block_size=1) as w:
        for img, ms in frames:
            arr = np.asarray(img)
            for _ in range(max(1, round(ms / (1000 / FPS)))):
                w.append_data(arr)
    print(f"wrote {mp4_path} ({mp4_path.stat().st_size/1e6:.2f} MB)")

    gif_path = out_dir / "demo.gif"
    pal_frames = [img.quantize(colors=64, dither=Image.Dither.NONE) for img, _ in frames]
    pal_frames[0].save(
        gif_path, save_all=True, append_images=pal_frames[1:],
        duration=[ms for _, ms in frames], loop=0, optimize=True,
    )
    print(f"wrote {gif_path} ({gif_path.stat().st_size/1e6:.2f} MB)")

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
