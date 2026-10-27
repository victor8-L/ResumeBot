"""Deterministic checks for corpus text that would print wrong. No API call.

Every check targets a mistake that has actually reached a draft: present tense under a
role that has ended, wording that dates itself ("upcoming"), two bullets in one entry
that disagree about the same quantity, and a title that reads one way in the footer and
another in the entry. `error` findings print wrong as written; `check` findings are
usually wrong but sometimes deliberate.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date

from . import corpus


@dataclass
class Finding:
    level: str  # "error" or "check"
    where: str  # the bullet, entry, or pool id to fix
    message: str


MONTHS = {
    m: i
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july", "august",
         "september", "october", "november", "december"],
        start=1,
    )
}
# The month a season-dated role ("Summer 2026") is over by.
SEASONS = {"winter": 2, "spring": 5, "summer": 8, "fall": 12, "autumn": 12}

# Base-form openers that read as present tense ("Manage club budget..."). Openers ending
# in -ing are caught by suffix, so only the bare forms need listing.
PRESENT_VERBS = {
    "analyze", "assist", "build", "collaborate", "conduct", "coordinate", "create",
    "design", "develop", "drive", "help", "lead", "maintain", "manage", "mentor",
    "organize", "oversee", "own", "participate", "plan", "practice", "research", "run",
    "serve", "support", "teach", "track", "train", "work", "write",
}

DATED_WORDS = re.compile(
    r"\b(upcoming|incoming|currently|soon|will|pending|expected to|"
    r"(?:this|next) (?:year|semester|summer|fall|spring|winter))\b",
    re.IGNORECASE,
)

# A quantity: optional $, digits, optional scale/unit, optional "+". Ordinals ("2nd")
# and digits glued to words are excluded by the trailing lookahead.
QUANTITY = re.compile(
    r"(?<![\w.$])(\$?)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(K|M|B|TB|GB|x|%)?(\+?)(?![A-Za-z\d])"
)
SCALE = {"K": 1e3, "M": 1e6, "B": 1e9}
STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "from", "in", "into", "of", "on", "over",
    "per", "than", "the", "to", "under", "with", "more",
}
ABBREVIATIONS = {"vp": "vice president"}

# A "– Present" entry no resume has used for this long is probably over.
STALE_PRESENT_DAYS = 180


def role_ended(dates: str, today: date) -> tuple[bool, str]:
    """(ended, end text) for an entry's date range. Unparseable ranges count as open."""
    end = re.split(r"\s*[–—]\s*|\s+-\s+", dates)[-1].strip()
    lowered = end.lower()
    if lowered in {"present", "current", "now", "ongoing"}:
        return False, end
    years = re.findall(r"\b(?:19|20)\d{2}\b", lowered)
    if not years:
        return False, end
    words = re.findall(r"[a-z]+", lowered)
    months = [MONTHS[w] for w in words if w in MONTHS] + [SEASONS[w] for w in words if w in SEASONS]
    month = months[-1] if months else 12
    return (int(years[-1]), month) < (today.year, today.month), end


def present_tense_opener(text: str) -> str | None:
    match = re.match(r"[A-Za-z]+", text)
    if not match:
        return None
    word = match.group(0)
    lowered = word.lower()
    if (lowered.endswith("ing") and len(lowered) > 4) or lowered in PRESENT_VERBS:
        return word
    return None


def _singular(word: str) -> str:
    word = word.lower()
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def quantities(text: str) -> list[tuple[str, tuple[str, float], str]]:
    """(noun, (kind, value), as written) for each quantity in a bullet.

    The noun is the first content word after the number, so "40 clients" and
    "35+ client accounts" are both about "client". Bare years are skipped.
    """
    found = []
    for m in QUANTITY.finditer(text):
        dollar, digits, unit, plus = m.groups()
        number = float(digits.replace(",", ""))
        if not (dollar or unit or plus) and 1900 <= number <= 2099 and "," not in digits:
            continue
        words = re.findall(r"[A-Za-z][A-Za-z-]*", text[m.end():])
        noun = next((w for w in words if w.lower() not in STOPWORDS), None)
        if noun is None:
            continue
        if unit in SCALE:
            number *= SCALE[unit]
            unit = ""
        kind = "$" if dollar else (unit or "count")
        found.append((_singular(noun), (kind, number), m.group(0)))
    return found


def _comparable(a: tuple[str, float], b: tuple[str, float]) -> bool:
    """Same kind of measure: "112%" vs "2.3x" both claim a change; "10%" vs "600" do not."""
    relative = {"%", "x"}
    return a[0] == b[0] or (a[0] in relative and b[0] in relative)


def _words(text: str) -> set[str]:
    return {
        _singular(w)
        for w in re.findall(r"[A-Za-z][A-Za-z-]+", text)
        if len(w) > 2 and w.lower() not in STOPWORDS
    }


def overlap(a: str, b: str) -> float:
    """Share of the shorter text's content words that the other also uses."""
    wa, wb = _words(a), _words(b)
    if min(len(wa), len(wb)) < 5:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def _title_words(text: str) -> set[str]:
    text = text.lower()
    for short, long in ABBREVIATIONS.items():
        text = re.sub(rf"\b{short}\b", long, text)
    return set(re.findall(r"[a-z]+", text)) - STOPWORDS


def lint(
    profile: dict,
    entries: list[dict],
    versions: dict | None = None,
    today: date | None = None,
) -> list[Finding]:
    versions = corpus.load_versions() if versions is None else versions
    today = today or date.today()
    findings: list[Finding] = []

    # --- Integrity: things that break selection outright. ---
    ids = [e["id"] for e in entries] + [b["id"] for e in entries for b in e["bullets"]]
    ed = profile["education"][0]
    pool_items = (
        ed["credential_options"] + ed["coursework_pool"] + profile["skills_pool"]
        + profile["activities_pool"] + profile["interests_pool"]
    )
    ids += [item["id"] for item in pool_items]
    for item_id, count in Counter(ids).items():
        if count > 1:
            findings.append(Finding("error", item_id, f"id used {count} times"))

    group_owners: dict[str, set[str]] = {}
    group_sizes: Counter = Counter()
    for entry in entries:
        for bullet in entry["bullets"]:
            for version in bullet.get("resume_versions", []):
                if version not in versions:
                    findings.append(Finding(
                        "error", bullet["id"],
                        f"resume version {version!r} is not registered in versions.json",
                    ))
            for domain in bullet.get("domains") or []:
                if domain not in corpus.DOMAINS:
                    findings.append(Finding("error", bullet["id"], f"unknown domain {domain!r}"))
            if bullet.get("alt_group"):
                group_sizes[bullet["alt_group"]] += 1
                group_owners.setdefault(bullet["alt_group"], set()).add(entry["id"])
    for group, size in group_sizes.items():
        if size < 2:
            findings.append(Finding(
                "check", group, "alt_group has one member — it groups nothing"
            ))
        if len(group_owners[group]) > 1:
            findings.append(Finding(
                "error", group,
                f"alt_group spans entries {sorted(group_owners[group])} — alternates "
                f"must live in one entry",
            ))

    newest_overall = max((v["date"] for v in versions.values()), default="")

    for entry in entries:
        ended, end = role_ended(entry["dates"], today)

        # --- Tense: an ended role written in the present. ---
        if ended:
            for bullet in entry["bullets"]:
                opener = present_tense_opener(bullet["text"])
                if opener:
                    findings.append(Finding(
                        "error", bullet["id"],
                        f"present tense ({opener!r}) but {entry['id']} ended {end}",
                    ))

        # --- Wording that was true when written and goes stale on its own. ---
        for field, text in (("org", entry["org"]), ("role", entry["role"])):
            hit = DATED_WORDS.search(text)
            if hit:
                findings.append(Finding(
                    "check", entry["id"], f"{field} says {hit.group(0)!r} — dates itself"
                ))
        for bullet in entry["bullets"]:
            hit = DATED_WORDS.search(bullet["text"])
            if hit:
                findings.append(Finding(
                    "error" if ended else "check", bullet["id"],
                    f"says {hit.group(0)!r}"
                    + (f" but {entry['id']} ended {end}" if ended else " — dates itself"),
                ))

        # --- "– Present" that no recent resume has backed up. ---
        if not ended and end.lower() == "present" and newest_overall:
            last_used = max(
                (corpus.newest_date(b, versions) for b in entry["bullets"]), default=""
            )
            if last_used:
                gap = date.fromisoformat(newest_overall) - date.fromisoformat(last_used)
                if gap.days > STALE_PRESENT_DAYS:
                    findings.append(Finding(
                        "check", entry["id"],
                        f"dated '– Present' but no resume since {last_used[:7]} uses it "
                        f"— still current?",
                    ))

        # --- Two bullets in one entry that disagree about the same quantity. ---
        seen: dict[str, list[tuple[tuple[str, float], str, str]]] = {}
        reported: set[tuple[str, str]] = set()
        for bullet in entry["bullets"]:
            for noun, value, written in quantities(bullet["text"]):
                for other_value, other_written, other_id in seen.get(noun, []):
                    pair = (other_id, bullet["id"])
                    if (
                        other_value != value
                        and _comparable(other_value, value)
                        and other_id != bullet["id"]
                        and pair not in reported
                    ):
                        reported.add(pair)
                        findings.append(Finding(
                            "check", bullet["id"],
                            f"says {written!r} {noun} but {other_id} says "
                            f"{other_written!r} {noun}",
                        ))
                seen.setdefault(noun, []).append((value, written, bullet["id"]))

        # --- Near-identical bullets that the selector is free to print together. ---
        bullets = entry["bullets"]
        for i, a in enumerate(bullets):
            for b in bullets[i + 1:]:
                if a.get("alt_group") and a.get("alt_group") == b.get("alt_group"):
                    continue
                score = overlap(a["text"], b["text"])
                if score >= 0.6:
                    findings.append(Finding(
                        "check", b["id"],
                        f"{round(score * 100)}% word overlap with {a['id']} and no shared "
                        f"alt_group — both could print on one page",
                    ))

    # --- The same org with two different titles (entry vs. activities line). ---
    for item in profile["activities_pool"]:
        match = re.match(r"(.+?)\s*\(([^)]+)\)\s*$", item["text"])
        if not match:
            continue
        name, title = match.group(1).strip().lower(), match.group(2)
        for entry in entries:
            if entry["org"].strip().lower() != name:
                continue
            if not _title_words(title) <= _title_words(entry["role"]):
                findings.append(Finding(
                    "check", item["id"],
                    f"activities line says {title!r} but entry {entry['id']} says "
                    f"{entry['role']!r}",
                ))

    for item in pool_items:
        hit = DATED_WORDS.search(item["text"])
        if hit:
            findings.append(Finding("check", item["id"], f"says {hit.group(0)!r} — dates itself"))
    hit = DATED_WORDS.search(ed.get("detail_line", ""))
    if hit:
        findings.append(Finding("check", "detail_line", f"says {hit.group(0)!r} — dates itself"))

    order = {"error": 0, "check": 1}
    return sorted(findings, key=lambda f: order[f.level])


def report(findings: list[Finding]) -> str:
    errors = sum(f.level == "error" for f in findings)
    checks = len(findings) - errors
    if not findings:
        return "  LINT — clean"
    lines = [f"  LINT — {errors} error(s), {checks} to check"]
    for f in findings:
        mark = "!" if f.level == "error" else "?"
        lines.append(f"    {mark} {f.where.ljust(16)} {f.message}")
    return "\n".join(lines)
