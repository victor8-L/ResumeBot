"""Deterministic rendering: view model -> HTML -> PDF, with a one-page fit loop.

The model picks content; this file owns the page. Fitting runs on two axes, both
measured against real PDF output rather than estimated:

  horizontal — pick the largest type size at which no bullet wraps to a second line
  vertical   — pick the loosest line spacing that still lands on exactly one page

Only if the tightest spacing still overflows does it start dropping content, cutting
the lowest-priority bullets the model marked as expendable.
"""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "templates"

# Times New Roman is the default. Your source resumes were set in Calibri, so the
# `calibri` stack (needs `brew install --cask font-carlito` off this machine) is kept
# as an option along with a couple of other faces.
FONT_STACKS = {
    "times": '"Times New Roman", Times, "Liberation Serif", Georgia, serif',
    "calibri": 'Calibri, Carlito, "Helvetica Neue", Arial, sans-serif',
    "narrow": '"Arial Narrow", Calibri, Carlito, "Helvetica Neue", sans-serif',
    "helvetica": '"Helvetica Neue", Helvetica, Arial, sans-serif',
}
DEFAULT_FONT = "times"

# Horizontal axis: margin + type size. Sorted below by how many characters fit on a
# line, most generous first, so the search can stop at the first non-wrapping step.
WIDTH_STEPS = [
    {"margin": "0.55in", "font_size": "11.5pt"},
    {"margin": "0.5in", "font_size": "11pt"},
    {"margin": "0.45in", "font_size": "11pt"},
    {"margin": "0.5in", "font_size": "10.5pt"},
    {"margin": "0.45in", "font_size": "10.5pt"},
    {"margin": "0.4in", "font_size": "10.5pt"},
    {"margin": "0.5in", "font_size": "10pt"},
    {"margin": "0.45in", "font_size": "10pt"},
    {"margin": "0.4in", "font_size": "10pt"},
    {"margin": "0.4in", "font_size": "9.5pt"},
    {"margin": "0.35in", "font_size": "9.5pt"},
    {"margin": "0.35in", "font_size": "9pt"},
]


def _capacity(step: dict) -> float:
    """Characters per line, up to a constant — usable width over type size."""
    margin = float(step["margin"].rstrip("in"))
    size = float(step["font_size"].rstrip("pt"))
    return (8.5 - 2 * margin) / size


# Ascending capacity: biggest type (fewest characters per line, most likely to wrap)
# first, so the search returns the most generous step that still fits on one line.
WIDTH_STEPS.sort(key=_capacity)

# Vertical axis: leading and the gaps between entries and sections. Loosest first.
SPACING_STEPS = [
    {"line_height": "1.50", "section_gap": "13px", "entry_gap": "11px", "name_size": "19pt"},
    {"line_height": "1.42", "section_gap": "11px", "entry_gap": "9px", "name_size": "19pt"},
    {"line_height": "1.35", "section_gap": "10px", "entry_gap": "8px", "name_size": "18pt"},
    {"line_height": "1.29", "section_gap": "9px", "entry_gap": "7px", "name_size": "18pt"},
    {"line_height": "1.24", "section_gap": "8px", "entry_gap": "6px", "name_size": "17pt"},
    {"line_height": "1.20", "section_gap": "6px", "entry_gap": "5px", "name_size": "17pt"},
    {"line_height": "1.16", "section_gap": "5px", "entry_gap": "4px", "name_size": "16pt"},
    {"line_height": "1.12", "section_gap": "4px", "entry_gap": "3px", "name_size": "15pt"},
]

CHROME_CANDIDATES = [
    os.environ.get("CHROME_PATH", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "google-chrome",
    "chromium",
]


class RenderError(Exception):
    pass


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if not candidate:
            continue
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise RenderError(
        "No Chrome/Chromium found for PDF rendering. Install Google Chrome, or set "
        "CHROME_PATH to a Chromium-based browser binary."
    )


def render_html(view: dict, width: dict, spacing: dict, font: str = DEFAULT_FONT) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("resume.html.jinja")
    payload = dict(view)
    payload["page"] = {**width, **spacing, "font_stack": FONT_STACKS[font]}
    # The template takes plain strings for bullets; the view carries dicts.
    payload["sections"] = [
        {
            "title": s["title"],
            "entries": [
                {**e, "bullets": [b["text"] for b in e["bullets"]]} for e in s["entries"]
            ],
        }
        for s in view["sections"]
    ]
    return template.render(**payload)


def html_to_pdf(html: str, out_pdf: Path, *, timeout: float = 60.0) -> int:
    """Print HTML to PDF with headless Chrome. Returns the page count.

    Recent Chrome builds write the PDF and then keep running instead of exiting, so
    we watch for the finished file and shut the browser down ourselves rather than
    waiting on the process.
    """
    chrome = find_chrome()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if out_pdf.exists():
        out_pdf.unlink()

    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "resume.html"
        html_path.write_text(html, encoding="utf-8")
        # A throwaway profile dir keeps this from fighting a running Chrome instance.
        cmd = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-sync",
            "--disable-component-update",
            f"--user-data-dir={tmp}/profile",
            "--no-pdf-header-footer",
            "--virtual-time-budget=2000",
            f"--print-to-pdf={out_pdf}",
            html_path.as_uri(),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + timeout
            last_size = -1
            while time.monotonic() < deadline:
                if out_pdf.exists():
                    size = out_pdf.stat().st_size
                    if size > 0 and size == last_size:
                        break  # file written and stable
                    last_size = size
                elif proc.poll() is not None:
                    break  # exited without writing anything
                time.sleep(0.1)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    if not out_pdf.exists():
        stderr = (proc.stderr.read() or b"").decode(errors="replace") if proc.stderr else ""
        raise RenderError(f"Chrome failed to produce a PDF.\n{stderr[-2000:]}")

    return len(PdfReader(str(out_pdf)).pages)


def _normalize(text: str) -> str:
    """Make PDF-extracted text comparable to corpus text.

    NFKC expands the ligatures the font substitutes in — Carlito renders "financial"
    with an fi ligature, which extracts as a single character and breaks a naive
    substring match. Also folds smart quotes and collapses whitespace.
    """
    text = unicodedata.normalize("NFKC", text)
    for fancy, plain in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
        text = text.replace(fancy, plain)
    return " ".join(text.split())


def count_wrapped(out_pdf: Path, view: dict) -> int:
    """How many bullets spilled onto a second line, read back off the rendered PDF.

    A bullet that fits on one line appears as one line of extracted text, so its head
    and tail land together. If it wrapped, they don't.
    """
    reader = PdfReader(str(out_pdf))
    lines = [
        _normalize(line)
        for page in reader.pages
        for line in (page.extract_text() or "").splitlines()
        if line.strip()
    ]
    wrapped = 0
    for section in view["sections"]:
        for entry in section["entries"]:
            for bullet in entry["bullets"]:
                text = _normalize(bullet["text"])
                head, tail = text[:40], text[-25:]
                if not any(head in line and tail in line for line in lines):
                    wrapped += 1
    return wrapped


FOOTER_LABELS = {"skills": "Skills:", "activities": "Activities:", "interests": "Interests:"}
FOOTER_SEPARATORS = {"skills": "; ", "activities": ", ", "interests": ", "}


def wrapped_footer_lines(out_pdf: Path, view: dict) -> list[str]:
    """Which of the Skills / Activities / Interests lines spilled onto a second line."""
    lines = [
        _normalize(line)
        for page in PdfReader(str(out_pdf)).pages
        for line in (page.extract_text() or "").splitlines()
        if line.strip()
    ]
    wrapped = []
    for key, label in FOOTER_LABELS.items():
        items = view["footer"].get(key) or []
        if not items:
            continue
        tail = _normalize(items[-1])
        if not any(label in line and tail in line for line in lines):
            wrapped.append(key)
    return wrapped


def _first_true(count: int, predicate) -> int | None:
    """Binary search for the lowest index where predicate flips False -> True."""
    lo, hi, found = 0, count - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            found, hi = mid, mid - 1
        else:
            lo = mid + 1
    return found


def _trim_once(view: dict) -> str | None:
    """Remove exactly one piece of content. Returns a description, or None if nothing left.

    Order: cuttable bullets (highest priority number first, latest on the page first),
    then interests, then activities, then whole entries from the bottom.
    """
    candidates = []
    for si, section in enumerate(view["sections"]):
        for ei, entry in enumerate(section["entries"]):
            if len(entry["bullets"]) <= 1:
                continue  # never leave an entry bulletless
            for bi, bullet in enumerate(entry["bullets"]):
                candidates.append((bullet["priority"], si, ei, bi, entry, bullet))

    cuttable = [c for c in candidates if c[0] > 1]
    pool = cuttable or candidates
    if pool:
        _, _, _, bi, entry, bullet = max(pool, key=lambda c: (c[0], c[1], c[2], c[3]))
        entry["bullets"].pop(bi)
        return f"bullet {bullet['id']} (priority {bullet['priority']})"

    for key in ("interests", "activities"):
        if view["footer"].get(key):
            view["footer"][key] = []
            return f"{key} line"

    for section in reversed(view["sections"]):
        if section["entries"]:
            entry = section["entries"].pop()
            if not section["entries"]:
                view["sections"].remove(section)
            return f"entry {entry['id']}"

    return None


def fit_to_one_page(
    view: dict, out_pdf: Path, *, font: str = DEFAULT_FONT, max_trims: int = 30
) -> dict:
    """Fit horizontally (no wrapped bullets), then vertically (fill the page)."""
    view = copy.deepcopy(view)
    renders = 0

    def draw(width: dict, spacing: dict) -> int:
        nonlocal renders
        renders += 1
        return html_to_pdf(render_html(view, width, spacing, font), out_pdf)

    # --- Horizontal: the most generous type at which nothing wraps. ---
    tightest_spacing = SPACING_STEPS[-1]

    def no_wrap(i: int) -> bool:
        draw(WIDTH_STEPS[i], tightest_spacing)
        return count_wrapped(out_pdf, view) == 0

    width_index = _first_true(len(WIDTH_STEPS), no_wrap)
    if width_index is None:
        # Some bullet is too long for any step; fall back to the smallest type.
        width_index = len(WIDTH_STEPS) - 1
    width = WIDTH_STEPS[width_index]

    # --- Footer lines are single-line too. Drop trailing items until they fit. ---
    trimmed: list[str] = []
    for _ in range(6):
        draw(width, tightest_spacing)
        overflowing = wrapped_footer_lines(out_pdf, view)
        if not overflowing:
            break
        for key in overflowing:
            items = view["footer"].get(key) or []
            if len(items) > 1:
                trimmed.append(f"{items[-1]!r} from the {key} line")
                view["footer"][key] = items[:-1]

    # --- Vertical: the loosest spacing that still lands on one page. ---
    def fits(i: int) -> bool:
        return draw(width, SPACING_STEPS[i]) == 1

    spacing_index = _first_true(len(SPACING_STEPS), fits)

    # --- Only now, if even the tightest spacing overflows, start cutting. ---
    if spacing_index is None:
        spacing_index = len(SPACING_STEPS) - 1
        for _ in range(max_trims):
            pages = draw(width, SPACING_STEPS[spacing_index])
            if pages == 1:
                # Room may have opened up; re-loosen the spacing.
                spacing_index = _first_true(len(SPACING_STEPS), fits) or spacing_index
                break
            removed = _trim_once(view)
            if removed is None:
                break
            trimmed.append(removed)

    pages = draw(width, SPACING_STEPS[spacing_index])
    return {
        "pages": pages,
        "font": font,
        "font_size": width["font_size"],
        "margin": width["margin"],
        "line_height": SPACING_STEPS[spacing_index]["line_height"],
        "wrapped": count_wrapped(out_pdf, view),
        "wrapped_footer": wrapped_footer_lines(out_pdf, view),
        "renders": renders,
        "trimmed": trimmed,
        "view": view,
    }
