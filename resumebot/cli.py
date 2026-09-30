"""resumeBot CLI: job description in, one-page tailored PDF out."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

from . import corpus, render

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"


def _load_env(path: Path | None = None) -> None:
    """Read ROOT/.env into os.environ. A real shell variable always wins."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug or "resume")[:60]


def _read_jd(args) -> tuple[str, str]:
    """Return (job description, how it arrived)."""
    if args.jd:
        return Path(args.jd).read_text(errors="replace"), f"file {args.jd}"
    if args.text:
        return args.text, "--text"
    if not sys.stdin.isatty():
        return sys.stdin.read(), "pipe"
    print("Paste the job description, then press Ctrl-D:\n", file=sys.stderr)
    return sys.stdin.read(), "terminal paste"


def _check_jd(jd: str, source: str) -> None:
    """Report what actually arrived.

    Pasting straight into the terminal goes through the tty line discipline, which
    truncates any single line past ~1KB. A posting pasted as a few very long unwrapped
    paragraphs can therefore lose most of its text silently, and the first visible
    symptom is a thin, confused selection. Printing the size makes that obvious, and
    `pbpaste |` avoids it entirely.
    """
    lines = jd.splitlines()
    longest = max((len(line) for line in lines), default=0)
    print(
        f"  read {len(jd)} chars / {len(lines)} lines from {source}",
        file=sys.stderr,
    )
    if source == "terminal paste" and longest >= 1023:
        print(
            f"  ! a line came in at exactly {longest} chars — the terminal truncates "
            f"lines at ~1KB on paste.\n"
            f"    Re-run as:  pbpaste | python3.12 -m resumebot.cli tailor",
            file=sys.stderr,
        )
    elif len(jd.strip()) < 300:
        print(
            "  ! that is very short for a job description — check the whole posting "
            "arrived before trusting the result",
            file=sys.stderr,
        )


def _report(result: dict, plan: dict, pdf_path: Path, phrasing: list[str] | None = None) -> None:
    view = result["view"]
    bullets = sum(len(e["bullets"]) for s in view["sections"] for e in s["entries"])
    entries = sum(len(s["entries"]) for s in view["sections"])

    print(f"\n  {pdf_path}")
    print(f"  {result['pages']} page(s) | {entries} entries | {bullets} bullets")
    if plan.get("domains"):
        print(f"  read as: {', '.join(plan['domains'])}")
    print(f"  {result['font']} {result['font_size']} | margin {result['margin']} "
          f"| leading {result['line_height']} | {result['renders']} renders")
    if result.get("wrapped_footer"):
        print(f"  ! {', '.join(result['wrapped_footer'])} line still wraps — "
              f"trim it in profile.json")
    if result["wrapped"]:
        print(f"  ! {result['wrapped']} bullet(s) wrap to a second line — "
              f"shorten them in the corpus, or try --font narrow")
    if result["trimmed"]:
        print("  trimmed to fit: " + "; ".join(result["trimmed"]))
    if result["pages"] != 1:
        print("  ! could not reach one page — check the plan or add density steps")

    for section in view["sections"]:
        print(f"\n  {section['title']}")
        for entry in section["entries"]:
            print(f"    {entry['org']} — {entry['role']}")
            for bullet in entry["bullets"]:
                print(f"      [p{bullet['priority']}] {bullet['text'][:88]}")

    if phrasing:
        print("\n  Older phrasing used: " + "; ".join(phrasing))

    if plan.get("notes"):
        print(f"\n  Why this framing:\n    {plan['notes']}")
    usage = plan.get("_usage")
    if usage:
        print(f"\n  {usage['model']}: {usage['input_tokens']} in / "
              f"{usage['output_tokens']} out "
              f"({usage.get('cache_read_input_tokens', 0)} cached)")


def _require_key() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    print(
        "No ANTHROPIC_API_KEY found.\n\n"
        "  1. Create a key at https://console.anthropic.com/settings/keys\n"
        "     (this is separate from your claude.ai login and is billed per use)\n"
        f"  2. Put it in {ROOT / '.env'}:\n"
        "         ANTHROPIC_API_KEY=sk-ant-...\n"
        "     or export it in your shell.\n\n"
        "  `render` and `show` work without a key.",
        file=sys.stderr,
    )
    return False


def cmd_tailor(args) -> int:
    if not _require_key():
        return 1
    from . import select  # imported here so `render` works without the anthropic package

    jd, source = _read_jd(args)
    if not jd.strip():
        print("No job description provided.", file=sys.stderr)
        return 1
    _check_jd(jd, source)

    profile, entries = corpus.load()
    stem = args.name or f"{date.today():%Y-%m-%d}-run"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    debug_path = OUT_DIR / f"{stem}.raw.json" if args.debug else None

    # Save the posting before spending the API call. A pasted job description is gone the
    # moment the run fails, and re-pasting it is the last thing you want to be doing.
    jd_path = OUT_DIR / f"{stem}.jd.txt"
    jd_path.write_text(jd)

    print("Selecting content...", file=sys.stderr)
    try:
        plan = select.build_plan(
            jd, profile, entries, model=args.model, effort=args.effort,
            debug_path=debug_path,
        )
    except RuntimeError as exc:
        print(f"\n{exc}\n\n  The job description was saved to {jd_path}\n"
              f"  Retry without re-pasting:  python3.12 -m resumebot.cli tailor "
              f"{jd_path}", file=sys.stderr)
        return 1
    if plan["_usage"].get("attempts", 1) > 1:
        print(f"  (retried {plan['_usage']['attempts']}x — earlier response was unusable)",
              file=sys.stderr)

    view = corpus.resolve(plan, profile, entries)

    stem = args.name or f"{date.today():%Y-%m-%d}-{_slug(plan.get('target', 'resume'))}"
    jd_path = jd_path.rename(OUT_DIR / f"{stem}.jd.txt")
    pdf_path = Path(args.out) if args.out else OUT_DIR / f"{stem}.pdf"
    plan_path = pdf_path.with_suffix(".plan.json")

    print("Rendering...", file=sys.stderr)
    result = render.fit_to_one_page(view, pdf_path, font=args.font)

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2))
    _report(result, plan, pdf_path,
            select.phrasing_notes(plan, entries, corpus.load_versions()))
    print(f"\n  plan saved: {plan_path}  (edit it, then `resumebot render`)")

    if args.open:
        subprocess.run(["open", str(pdf_path)], check=False)
    return 0


def cmd_render(args) -> int:
    """Re-render a saved plan with no API call — for hand-edits and template tweaks."""
    plan = json.loads(Path(args.plan).read_text())
    profile, entries = corpus.load()
    view = corpus.resolve(plan, profile, entries)

    pdf_path = Path(args.out) if args.out else Path(args.plan).with_suffix("").with_suffix(".pdf")
    result = render.fit_to_one_page(view, pdf_path, font=args.font)
    _report(result, plan, pdf_path)
    if args.open:
        subprocess.run(["open", str(pdf_path)], check=False)
    return 0


def cmd_ingest(args) -> int:
    if not _require_key():
        return 1
    from . import ingest

    docs = ingest.collect()
    if not docs:
        print("No readable resumes in data/.", file=sys.stderr)
        return 1
    print(f"Read {len(docs)} file(s): {', '.join(docs)}", file=sys.stderr)

    entries = ingest.build_entries(docs, model=args.model)
    out_path = Path(args.out) if args.out else corpus.CORPUS_DIR / "entries.json"
    if out_path.exists() and not args.force:
        out_path = out_path.with_suffix(".new.json")
        print(f"(existing corpus kept; writing to {out_path.name} — diff before replacing)")
    out_path.write_text(json.dumps(entries, indent=2))

    bullets = sum(len(e["bullets"]) for e in entries)
    print(f"\n  {out_path}\n  {len(entries)} entries | {bullets} bullets")
    print("  Note: profile.json (contact, education, skills/interests pools) is hand-maintained.")
    return 0


def cmd_show(args) -> int:
    profile, entries = corpus.load()
    print(corpus.menu(profile, entries))
    return 0


def cmd_audit(args) -> int:
    """Count the evidence behind each domain, so a thin bucket is a number not a hunch."""
    profile, entries = corpus.load()
    print()
    print(corpus.audit(profile, entries))
    return 0


def _explain_api_error(exc: Exception) -> str | None:
    """Turn the API failures you actually hit into one actionable line."""
    message = str(getattr(exc, "message", exc))
    status = getattr(exc, "status_code", None)
    console = "https://console.anthropic.com"

    if status == 401:
        return (
            f"Authentication failed — the API key was rejected.\n"
            f"  Check ANTHROPIC_API_KEY in {ROOT / '.env'}: a real key starts 'sk-ant-api03-'\n"
            f"  and runs past 100 characters. Manage keys at {console}/settings/keys"
        )
    if status == 400 and "credit balance" in message.lower():
        return (
            f"Your API account is out of credits.\n"
            f"  Add credits at {console}/settings/billing — API usage is billed\n"
            f"  separately from any claude.ai subscription."
        )
    if status == 429:
        return f"Rate limited. Wait a moment and retry; limits are at {console}/settings/limits"
    if status == 404 and "model" in message.lower():
        return f"Unknown model. Check --model against {console}/settings/models"
    return None


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = argparse.ArgumentParser(prog="resumebot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tailor", help="job description -> tailored one-page PDF")
    p.add_argument("jd", nargs="?", help="path to a .txt job description (or pipe/paste it)")
    p.add_argument("--text", help="job description as an inline string")
    p.add_argument("--out", help="output PDF path")
    p.add_argument("--name", help="output filename stem (default: date + target slug)")
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--open", action="store_true", help="open the PDF when done")
    p.add_argument("--debug", action="store_true", help="save the raw model response")
    p.add_argument("--font", default=render.DEFAULT_FONT, choices=sorted(render.FONT_STACKS),
                   help="typeface family (default: calibri, falls back if not installed)")
    p.set_defaults(func=cmd_tailor)

    p = sub.add_parser("render", help="re-render a saved plan (no API call)")
    p.add_argument("plan", help="path to a .plan.json")
    p.add_argument("--out", help="output PDF path")
    p.add_argument("--open", action="store_true")
    p.add_argument("--font", default=render.DEFAULT_FONT, choices=sorted(render.FONT_STACKS))
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("ingest", help="rebuild corpus/entries.json from data/")
    p.add_argument("--out", help="output JSON path")
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--force", action="store_true", help="overwrite corpus/entries.json in place")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("show", help="print the corpus as the model sees it")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("audit", help="domain coverage and freshness of the corpus")
    p.set_defaults(func=cmd_audit)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (corpus.CorpusError, render.RenderError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        explanation = _explain_api_error(exc)
        if explanation is None:
            raise
        print(f"\n{explanation}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
