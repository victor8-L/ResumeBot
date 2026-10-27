"""What postings ask for, measured against what the corpus can show. No API call.

The selector lists each posting's requirements with the corpus ids that demonstrate them
(see `requirements` in select.SCHEMA). Coverage is then computed here, not taken from the
model: a requirement is on the page if any of its evidence survived the fitter, off the
page if the corpus has evidence that was not used, and missing if nothing in the corpus
shows it. Missing items across many postings are the list of bullets worth writing next.
"""

from __future__ import annotations

from . import corpus

STATUS_LABELS = {"on_page": "on page", "off_page": "off page", "missing": "MISSING"}


def plan_ids(plan: dict) -> set[str]:
    """Every corpus id a plan selected, before any trimming."""
    ids = {s["bullet_id"] for s in plan.get("selections") or []}
    ids.add(plan.get("credential_id"))
    for key in ("coursework_ids", "skill_ids", "activity_ids", "interest_ids"):
        ids.update(plan.get(key) or [])
    return ids


def page_ids(view: dict, plan: dict, profile: dict, entries: list[dict]) -> set[str]:
    """Ids that survived the fitter: trimmed bullets and dropped footer items don't count."""
    bullets = {b["id"] for s in view["sections"] for e in s["entries"] for b in e["bullets"]}
    ed = view["education"][0]
    shown = {ed["credential"], *ed["coursework"]}
    for items in view["footer"].values():
        shown.update(items)
    pools = corpus.index(profile, entries)["pools"]
    return bullets | {i for i in plan_ids(plan) if i in pools and pools[i]["text"] in shown}


def coverage(
    plan: dict, profile: dict, entries: list[dict], on_page: set[str] | None = None
) -> list[dict]:
    idx = corpus.index(profile, entries)
    known = set(idx["bullets"]) | set(idx["pools"])
    on_page = plan_ids(plan) if on_page is None else on_page
    rows = []
    for req in plan.get("requirements") or []:
        evidence = [i for i in req.get("evidence_ids") or [] if i in known]
        if any(i in on_page for i in evidence):
            status = "on_page"
        elif evidence:
            status = "off_page"
        else:
            status = "missing"
        rows.append({**req, "evidence_ids": evidence, "status": status})
    return rows


def report(rows: list[dict]) -> str:
    """One run: what the posting asked for, grouped by whether the page shows it."""
    if not rows:
        return ""
    lines = ["  What the posting asks for:"]
    for status, label in STATUS_LABELS.items():
        group = [r for r in rows if r["status"] == status]
        if not group:
            continue
        items = []
        for r in group:
            item = r["label"]
            if status != "on_page" and r["importance"] == "required":
                item += " (required)"
            if status == "off_page":
                item += f" [{', '.join(r['evidence_ids'])}]"
            items.append(item)
        lines.append(f"    {label.ljust(9)} {', '.join(items)}")
    return "\n".join(lines)


def summary(plans: list[tuple[str, dict]], profile: dict, entries: list[dict]) -> str:
    """Across saved plans: what keeps coming up that the corpus can't show."""
    usable = [(name, plan) for name, plan in plans if plan.get("requirements")]
    skipped = len(plans) - len(usable)
    header = f"  REQUIREMENTS ACROSS {len(usable)} SAVED PLAN(S)"
    if skipped:
        header += f" ({skipped} older plan(s) predate requirements and are skipped)"
    lines = [header]
    if not usable:
        lines.append("    Nothing to report yet — run `tailor` to build up plans.")
        return "\n".join(lines)

    tally: dict[str, dict] = {}
    for name, plan in usable:
        for row in coverage(plan, profile, entries):
            t = tally.setdefault(
                row["label"].lower(),
                {"label": row["label"], "asked": 0, "required": 0, "missing": [], "off_page": 0},
            )
            t["asked"] += 1
            t["required"] += row["importance"] == "required"
            if row["status"] == "missing":
                t["missing"].append(plan.get("target") or name)
            elif row["status"] == "off_page":
                t["off_page"] += 1

    gaps = sorted(
        (t for t in tally.values() if t["missing"]),
        key=lambda t: (-len(t["missing"]), -t["required"], t["label"].lower()),
    )
    lines.append("\n  Nothing in the corpus shows these — write the bullet or build the skill:")
    if not gaps:
        lines.append("    (none)")
    for t in gaps:
        where = "; ".join(t["missing"][:3]) + (" …" if len(t["missing"]) > 3 else "")
        lines.append(
            f"    {t['label'][:26].ljust(26)} missing in {len(t['missing'])}/{t['asked']}, "
            f"required in {t['required']}  ({where})"
        )

    cut = sorted(
        (t for t in tally.values() if t["off_page"]),
        key=lambda t: (-t["off_page"], t["label"].lower()),
    )
    if cut:
        lines.append("\n  In the corpus but not selected:")
        for t in cut:
            lines.append(f"    {t['label'][:26].ljust(26)} left off {t['off_page']} of {t['asked']}")

    asked = sorted(tally.values(), key=lambda t: (-t["asked"], t["label"].lower()))[:10]
    lines.append(
        "\n  Most asked for: " + ", ".join(f"{t['label']} ({t['asked']})" for t in asked)
    )
    return "\n".join(lines)
