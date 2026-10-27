"""resumeBot CLI: job description in, one-page tailored PDF out."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

from . import corpus, gaps, lint, render

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


def _quality_notes(result: dict, plan: dict, profile: dict, entries: list[dict]) -> None:
    """Requirement coverage, and any lint finding about something actually on the page."""
    shown = gaps.page_ids(result["view"], plan, profile, entries)
    coverage = gaps.report(gaps.coverage(plan, profile, entries, on_page=shown))
    if coverage:
        print(f"\n{coverage}")

    shown |= {e["id"] for s in result["view"]["sections"] for e in s["entries"]}
    on_page = [f for f in lint.lint(profile, entries) if f.where in shown]
    if on_page:
        print("\n  Lint findings on this page (fix in the corpus, then `render` again):")
        for f in on_page:
            print(f"    {'!' if f.level == 'error' else '?'} {f.where}: {f.message}")


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
    _quality_notes(result, plan, profile, entries)
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
    _quality_notes(result, plan, profile, entries)
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


def _new_findings(before: list, after: list) -> list:
    seen = {(f.level, f.where, f.message) for f in before}
    return [f for f in after if (f.level, f.where, f.message) not in seen]


def cmd_merge(args) -> int:
    """Fold one new resume into the corpus: propose (one API call), then --apply (none)."""
    from . import merge

    profile, entries = corpus.load()
    versions = corpus.load_versions()

    if args.apply:
        if corpus.CORPUS_DIR.name != "corpus":
            print("There is no corpus/ yet — the shipped example is read-only.\n"
                  "  cp -r corpus.example corpus", file=sys.stderr)
            return 1
        proposal = json.loads(Path(args.file).read_text())
        new_profile, new_entries, new_versions, log = merge.apply(
            proposal, profile, entries, versions
        )
        backup = corpus.CORPUS_DIR / f"backup-{datetime.now():%Y-%m-%d-%H%M%S}"
        backup.mkdir()
        for name in ("profile.json", "entries.json", "versions.json"):
            if (corpus.CORPUS_DIR / name).exists():
                shutil.copy2(corpus.CORPUS_DIR / name, backup / name)
        corpus.save(new_profile, new_entries, new_versions)
        print("\n".join(f"  {line}" for line in log))
        print(f"\n  applied to {corpus.CORPUS_DIR}  (previous files in {backup.name}/)\n")
        print(lint.report(lint.lint(new_profile, new_entries, new_versions)))
        return 0

    if not _require_key():
        return 1
    path = Path(args.file)
    already = [slug for slug, v in versions.items() if v.get("file") == path.name]
    if already and not args.slug:
        print(f"{path.name} is already registered as {', '.join(already)}. "
              f"Pass --slug NAME to merge it again as a new version.", file=sys.stderr)
        return 1
    source = merge.source_for(path, versions, written=args.date, slug=args.slug)

    from . import ingest

    text = ingest.extract_text(path)
    print(f"  read {len(text)} chars from {path.name}; will register as "
          f"{source['slug']} (written {source['date']})", file=sys.stderr)
    print("Comparing against the corpus...", file=sys.stderr)
    try:
        proposal = merge.propose(
            text, source, profile, entries, versions, model=args.model, effort=args.effort
        )
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"merge-{source['slug']}.json"
    out_path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False) + "\n")

    new_profile, new_entries, new_versions, log = merge.apply(
        proposal, profile, entries, versions
    )
    print("\n  PROPOSED CHANGES (nothing written yet)")
    print("\n".join(f"  {line}" for line in log))
    fresh = _new_findings(
        lint.lint(profile, entries, versions), lint.lint(new_profile, new_entries, new_versions)
    )
    if fresh:
        print("\n  Lint findings this merge would introduce:")
        print("\n".join(f"    {'!' if f.level == 'error' else '?'} {f.where}: {f.message}"
                        for f in fresh))
    usage = proposal["_usage"]
    print(f"\n  {usage['model']}: {usage['input_tokens']} in / {usage['output_tokens']} out")
    print(f"\n  proposal saved: {out_path}\n"
          f"  Edit it if anything is wrong (change an action, delete a row), then apply:\n"
          f"    python -m resumebot.cli merge --apply {out_path}")
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
    findings = lint.lint(profile, entries)
    errors = sum(f.level == "error" for f in findings)
    print(f"\n  lint: {errors} error(s), {len(findings) - errors} to check"
          + ("  — run `lint` for details" if findings else ""))
    return 0


def cmd_lint(args) -> int:
    """Corpus text that would print wrong: tense, dated words, conflicting numbers."""
    profile, entries = corpus.load()
    findings = lint.lint(profile, entries)
    print()
    print(lint.report(findings))
    return 1 if any(f.level == "error" for f in findings) else 0


def cmd_gaps(args) -> int:
    """Tally requirements across saved plans: what to write or learn next."""
    profile, entries = corpus.load()
    plans = [
        (path.name, json.loads(path.read_text()))
        for path in sorted(Path(args.dir).glob("*.plan.json"))
    ]
    print()
    print(gaps.summary(plans, profile, entries))
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

    p = sub.add_parser("merge", help="fold one new resume into the corpus (propose, then --apply)")
    p.add_argument("file", help="a resume (.pdf/.docx/.txt), or with --apply a saved proposal")
    p.add_argument("--apply", action="store_true",
                   help="apply a saved proposal JSON to corpus/ (no API call)")
    p.add_argument("--date", help="when the resume was written, YYYY-MM-DD (default: file date)")
    p.add_argument("--slug", help="version id to register it under (default: from filename)")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("show", help="print the corpus as the model sees it")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("audit", help="domain coverage and freshness of the corpus")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("lint", help="find corpus text that would print wrong (no API call)")
    p.set_defaults(func=cmd_lint)

    p = sub.add_parser("gaps", help="what saved plans asked for that the corpus can't show")
    p.add_argument("--dir", default=str(OUT_DIR), help="folder of .plan.json files")
    p.set_defaults(func=cmd_gaps)

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
