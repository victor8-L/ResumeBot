"""Load the resume corpus and turn a model-produced plan into a render-ready view model."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# corpus/ holds your real material and is gitignored. A fresh clone has none, so fall
# back to the fictional example corpus that ships with the repo.
CORPUS_DIR = ROOT / "corpus"
if not CORPUS_DIR.exists():
    CORPUS_DIR = ROOT / "corpus.example"

SECTION_TITLES = [
    "Professional Experience",
    "Projects & Research",
    "Leadership & Experience",
]

# The three career tracks Victor applies into. Every entry, bullet, course, skill and
# activity is tagged with the subset it is evidence for; material that is domain-neutral
# (lifeguarding, sailing) carries an empty list and is selected on merit alone.
DOMAINS = ["finance", "national-security", "data-science"]

# Freshness tiers, newest first. A bullet's tier is the newest resume it appears in.
FRESHNESS_TIERS = [("current", "2026-06-01"), ("recent", "2026-01-01"), ("archive", "")]


class CorpusError(Exception):
    pass


def load(corpus_dir: Path | None = None) -> tuple[dict, list[dict]]:
    corpus_dir = corpus_dir or CORPUS_DIR
    profile = json.loads((corpus_dir / "profile.json").read_text())
    entries = json.loads((corpus_dir / "entries.json").read_text())
    return profile, entries


def load_versions(corpus_dir: Path | None = None) -> dict:
    """id -> {date, file, label, domains} for every past resume the corpus cites."""
    path = (corpus_dir or CORPUS_DIR) / "versions.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("versions", {})


def freshness(bullet: dict, versions: dict) -> tuple[str, str]:
    """(tier, YYYY-MM) for a bullet, from the newest resume that still carries it.

    This is the tie-breaker the selector uses between alternate phrasings: the wording
    on June's resume beats the wording on last September's. It is deliberately *only* a
    tie-breaker — plenty of archive bullets say something no current bullet says.
    """
    dates = [
        versions[v]["date"] for v in bullet.get("resume_versions", []) if v in versions
    ]
    if not dates:
        return "archive", "unknown"
    newest = max(dates)
    tier = next(name for name, floor in FRESHNESS_TIERS if newest >= floor)
    return tier, newest[:7]


def index(profile: dict, entries: list[dict]) -> dict:
    """Flat id -> object lookups, so a plan can be validated cheaply."""
    by_entry = {e["id"]: e for e in entries}
    by_bullet = {b["id"]: (e, b) for e in entries for b in e["bullets"]}
    pools = {}
    for ed in profile["education"]:
        for c in ed["credential_options"]:
            pools[c["id"]] = c
        for c in ed["coursework_pool"]:
            pools[c["id"]] = c
    for key in ("skills_pool", "activities_pool", "interests_pool"):
        for item in profile[key]:
            pools[item["id"]] = item
    return {"entries": by_entry, "bullets": by_bullet, "pools": pools}


def _domains(item: dict) -> str:
    return ", ".join(item.get("domains") or []) or "any"


def _annotate(item: dict) -> str:
    return f"domains: {_domains(item)} | tags: {', '.join(item.get('tags', []))}"


def menu(profile: dict, entries: list[dict], versions: dict | None = None) -> str:
    """The corpus rendered as compact text for the selector prompt.

    Every selectable atom appears here with the exact id the model must return, the
    domains it is evidence for, and how recently Victor last used it.
    """
    versions = load_versions() if versions is None else versions
    out: list[str] = []

    out.append(
        "Domains: finance | national-security | data-science. `domains: any` means the "
        "item is domain-neutral and judged on merit. Freshness (current / recent / "
        "archive) is when this exact wording last appeared on a real resume."
    )

    ed = profile["education"][0]
    out.append("\n## EDUCATION — pick exactly one credential line")
    for c in ed["credential_options"]:
        out.append(f"  [{c['id']}] {c['text']}   ({_annotate(c)})")
    out.append("\n## COURSEWORK POOL — pick 3-4, ordered by relevance")
    for c in ed["coursework_pool"]:
        out.append(f"  [{c['id']}] {c['text']}   ({_annotate(c)})")

    out.append("\n## EXPERIENCE ENTRIES — bullets are listed freshest first")
    for e in entries:
        out.append(
            f"\n[{e['id']}] {e['org']} — {e['role']} | {e['location']} | {e['dates']}"
            f"\n  default section: {e['default_section']} | {_annotate(e)}"
        )
        for b in e["bullets"]:
            tier, when = freshness(b, versions)
            stamp = tier if tier == "current" else f"{tier}, last used {when}"
            alt = f" | ALT-GROUP: {b['alt_group']}" if b.get("alt_group") else ""
            out.append(
                f"    [{b['id']}] {b['text']}"
                f"\n      ({stamp} | {_annotate(b)}{alt})"
            )

    for label, key, hint in (
        ("SKILLS", "skills_pool", "pick 5-9"),
        ("ACTIVITIES", "activities_pool", "pick 3-5"),
        ("INTERESTS", "interests_pool", "pick 4-5"),
    ):
        out.append(f"\n## {label} POOL — {hint}, ordered")
        for item in profile[key]:
            out.append(f"  [{item['id']}] {item['text']}   ({_annotate(item)})")

    return "\n".join(out)


def audit(profile: dict, entries: list[dict], versions: dict | None = None) -> str:
    """Report what the corpus can actually say about each domain.

    A resume can only be as balanced as the material behind it. This counts the evidence
    per domain so a thin bucket shows up as a number instead of as a weak resume.
    """
    versions = load_versions() if versions is None else versions
    lines: list[str] = []

    ed = profile["education"][0]
    pools = {
        "entries": entries,
        "bullets": [b for e in entries for b in e["bullets"]],
        "coursework": ed["coursework_pool"],
        "skills": profile["skills_pool"],
        "activities": profile["activities_pool"],
    }

    lines.append("  DOMAIN COVERAGE")
    header = "    " + "".ljust(12) + "".join(d.ljust(18) for d in DOMAINS) + "neutral"
    lines.append(header)
    for name, items in pools.items():
        counts = [
            sum(1 for i in items if d in (i.get("domains") or [])) for d in DOMAINS
        ]
        neutral = sum(1 for i in items if not i.get("domains"))
        row = "".join(str(c).ljust(18) for c in counts)
        lines.append(f"    {name.ljust(12)}{row}{neutral}")

    lines.append("\n  BULLETS BY DOMAIN AND ENTRY")
    for domain in DOMAINS:
        owners = Counter(
            e["id"]
            for e in entries
            for b in e["bullets"]
            if domain in (b.get("domains") or [])
        )
        total = sum(owners.values())
        detail = ", ".join(f"{k} ({v})" for k, v in owners.most_common()) or "nothing"
        lines.append(f"    {domain}: {total} bullets across {len(owners)} entries")
        lines.append(f"      {detail}")

    lines.append("\n  FRESHNESS")
    tiers = Counter(
        freshness(b, versions)[0] for e in entries for b in e["bullets"]
    )
    lines.append(
        "    " + " | ".join(f"{name}: {tiers.get(name, 0)}" for name, _ in FRESHNESS_TIERS)
    )
    stale_alts = []
    for entry in entries:
        groups: dict[str, list[tuple[str, str]]] = {}
        for bullet in entry["bullets"]:
            if bullet.get("alt_group"):
                groups.setdefault(bullet["alt_group"], []).append(
                    (bullet["id"], freshness(bullet, versions)[0])
                )
        for group, members in groups.items():
            best = min(members, key=lambda m: [t for t, _ in FRESHNESS_TIERS].index(m[1]))
            stale_alts.append(f"{group}: prefer {best[0]} ({best[1]})")
    if stale_alts:
        lines.append("    preferred phrasing per alt-group:")
        lines.extend(f"      {s}" for s in stale_alts)

    return "\n".join(lines)


def resolve(plan: dict, profile: dict, entries: list[dict]) -> dict:
    """Plan (ids) -> view model (text). Raises CorpusError on any unknown id."""
    idx = index(profile, entries)

    def pool(item_id: str) -> str:
        if item_id not in idx["pools"]:
            raise CorpusError(f"unknown id in plan: {item_id!r}")
        return idx["pools"][item_id]["text"]

    ed = profile["education"][0]
    education = [
        {
            "org": ed["org"],
            "location": ed["location"],
            "dates": ed["dates"],
            "credential": pool(plan["credential_id"]),
            "detail_line": ed.get("detail_line", ""),
            "coursework": [pool(c) for c in plan.get("coursework_ids", [])],
        }
    ]

    # Flat selection rows -> grouped sections. Order is first-appearance, so the model
    # controls layout order purely by the order it emits rows.
    grouped: dict[str, dict[str, dict]] = {}
    for sel in plan["selections"]:
        found = idx["bullets"].get(sel["bullet_id"])
        if found is None:
            raise CorpusError(f"unknown bullet id in plan: {sel['bullet_id']!r}")
        owner, bullet = found
        if owner["id"] != sel["entry_id"]:
            raise CorpusError(
                f"bullet {bullet['id']!r} belongs to entry {owner['id']!r}, "
                f"not {sel['entry_id']!r}"
            )
        if sel["section"] not in SECTION_TITLES:
            raise CorpusError(f"unknown section in plan: {sel['section']!r}")

        section = grouped.setdefault(sel["section"], {})
        entry = section.setdefault(
            owner["id"],
            {
                "id": owner["id"],
                "org": owner["org"],
                "role": owner["role"],
                "location": owner["location"],
                "dates": owner["dates"],
                "bullets": [],
            },
        )
        entry["bullets"].append(
            {"id": bullet["id"], "text": bullet["text"], "priority": sel["priority"]}
        )

    sections = [
        {"title": title, "entries": list(section_entries.values())}
        for title, section_entries in grouped.items()
        if section_entries
    ]

    if not sections:
        raise CorpusError(
            "This plan selects no experience — the resume would be an education stub. "
            "Check the `selections` array in the plan."
        )

    return {
        "name": profile["name"],
        "contact": profile["contact"],
        "education": education,
        "sections": sections,
        "footer": {
            "skills": [pool(i) for i in plan.get("skill_ids", [])],
            "activities": [pool(i) for i in plan.get("activity_ids", [])],
            "interests": [pool(i) for i in plan.get("interest_ids", [])],
        },
    }
