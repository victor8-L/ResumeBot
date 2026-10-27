"""Fold one new resume into the corpus without rebuilding it.

`ingest` regenerates entries.json from every file in data/, which throws away the
hand-tuned tags, domains, and alt-groups. `merge` asks the model how each line of ONE new
resume relates to what the corpus already holds, saves that answer as a proposal you can
read and edit, and applies it deterministically:

    same       the existing bullet gains this resume in resume_versions
    reword     the existing bullet's text is replaced — only if this resume is newer
    alternate  a new bullet joins the existing one's alt_group
    new        a new bullet, under an existing entry or a new one

The model classifies; code decides. Where the two disagree (the model calls a changed
sentence "same", or a verbatim one "new") the text wins, and an older resume can never
overwrite wording that a newer one already uses.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import date
from pathlib import Path

from . import corpus

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM = """You maintain a candidate's resume corpus: a library of their real \
accomplishments, every bullet written by the candidate. A later step selects bullets from \
it and prints them verbatim on real job applications. You are given the corpus and ONE new \
resume the candidate wrote, and you report how the new resume relates to the corpus. You \
classify; you never write or improve wording.

For every bullet on the new resume, choose one `action`:
- same: the corpus already holds this sentence, allowing for punctuation, capitalization, \
quote style and PDF extraction noise. `existing_id` = that bullet.
- reword: an edited version of an existing bullet that should retire the old wording: a \
number changed, the tense changed because the role ended, a claim was corrected or \
softened, or it is plainly the next draft of the same line. `existing_id` = the bullet it \
replaces.
- alternate: a deliberately different framing of an accomplishment the corpus already \
states, where both wordings stay true and each suits different employers (a finance \
framing and a data-science framing of one project). `existing_id` = the closest existing \
wording.
- new: an accomplishment the corpus does not state at all. `existing_id` = null.

Between reword and alternate, ask whether the candidate would still send the old wording \
today. If it is now false, stale, or superseded, it is a reword. A bullet that combines two \
existing bullets is an alternate of the one it most resembles.

`text` is the bullet exactly as written on the new resume. Repair only extraction damage \
(a line broken mid-bullet, ligatures, doubled spaces, curly quotes, which become straight) \
and plain typos such as a stray capital mid-sentence or a product name split in two that \
the corpus spells as one word. Do not shorten, merge, or improve anything.

`entry_id` is the corpus entry the bullet sits under. When the resume shows a role the \
corpus lacks, add it to `new_entries` with a short kebab-case id and use that id. When an \
existing role's org name, title, location, or dates changed (an end date replacing \
"Present", a company renamed), put the new values in `entry_updates`, null for fields \
that did not change, and keep using the existing entry_id. A renamed employer is an \
update, not a new entry.

`domains` is the subset of finance / national-security / data-science the bullet is real \
evidence for; [] for domain-neutral material. Be strict and match how the corpus labels \
similar bullets. `tags` are 3-6 lowercase kebab-case retrieval labels, reusing the \
corpus's own vocabulary where a tag fits. For same and reword, repeat the existing \
bullet's domains and tags.

Education and footer:
- `detail_line`: the GPA / test score / honors line, only if it differs from the corpus; \
otherwise null.
- `pool_changes`: credential lines, courses, skills, activities and interests on the new \
resume that the corpus lacks (`add`) or words differently (`rename`, with `existing_id`). \
List nothing that already matches. The same item with a qualifier added or dropped \
("Excel" against "Excel (Financial Modeling)") is never an add: it is a rename if the new \
wording should replace the old, and otherwise nothing. Never infer that something was \
removed because it is absent: one resume is a selection, not the full record.

`conflicts`: every fact on the new resume that contradicts the corpus (a number, a date, \
a title, a name), one plain sentence each, even when an action above already handles it. \
The candidate reads these before anything is applied.

`resume_domains`: the tracks this resume as a whole is aimed at.

`reason` on each bullet: a few words on why, for the candidate's terminal."""

_DOMAINS = {"type": "array", "items": {"type": "string", "enum": corpus.DOMAINS}}
_TAGS = {"type": "array", "items": {"type": "string"}}
_MAYBE = {"anyOf": [{"type": "string"}, {"type": "null"}]}


def _obj(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


POOLS = ["credential_options", "coursework_pool", "skills_pool", "activities_pool", "interests_pool"]
POOL_PREFIX = {
    "credential_options": "cred",
    "coursework_pool": "course",
    "skills_pool": "skill",
    "activities_pool": "act",
    "interests_pool": "int",
}

SCHEMA = _obj({
    "resume_domains": _DOMAINS,
    "bullets": {"type": "array", "items": _obj({
        "text": {"type": "string"},
        "action": {"type": "string", "enum": ["same", "reword", "alternate", "new"]},
        "existing_id": _MAYBE,
        "entry_id": {"type": "string"},
        "domains": _DOMAINS,
        "tags": _TAGS,
        "reason": {"type": "string"},
    })},
    "new_entries": {"type": "array", "items": _obj({
        "id": {"type": "string"},
        "org": {"type": "string"},
        "role": {"type": "string"},
        "location": {"type": "string"},
        "dates": {"type": "string"},
        "default_section": {"type": "string", "enum": ["professional", "projects", "leadership"]},
        "domains": _DOMAINS,
        "tags": _TAGS,
    })},
    "entry_updates": {"type": "array", "items": _obj({
        "entry_id": {"type": "string"},
        "org": _MAYBE,
        "role": _MAYBE,
        "location": _MAYBE,
        "dates": _MAYBE,
    })},
    "detail_line": _MAYBE,
    "pool_changes": {"type": "array", "items": _obj({
        "pool": {"type": "string", "enum": POOLS},
        "action": {"type": "string", "enum": ["add", "rename"]},
        "existing_id": _MAYBE,
        "text": {"type": "string"},
        "domains": _DOMAINS,
        "tags": _TAGS,
    })},
    "conflicts": {"type": "array", "items": {"type": "string"}},
})


def clean(text: str) -> str:
    """Corpus spelling: straight quotes, single spaces."""
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    """Comparison key: what counts as 'the same sentence'."""
    text = clean(text).lower().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", re.sub(r"[^\w$%+]+", " ", text)).strip()


def _core(text: str) -> str:
    """A pool item without its qualifier: "Excel (Financial Modeling)" -> "excel"."""
    return normalize(re.sub(r"\([^)]*\)", " ", text))


def source_for(
    path: Path, versions: dict, *, written: str | None = None, slug: str | None = None
) -> dict:
    """How the new resume will be registered in versions.json.

    The label is the parenthetical in the filename when there is one
    ("Resume(Finance).pdf" -> "Finance"); the date is the file's
    modification time unless given.
    """
    paren = re.search(r"\(([^)]+)\)", path.stem)
    label = paren.group(1) if paren else path.stem
    if slug is None:
        base = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "resume"
        n = 1
        while f"{base}-v{n}" in versions:
            n += 1
        slug = f"{base}-v{n}"
    written = written or date.fromtimestamp(path.stat().st_mtime).isoformat()
    return {"file": path.name, "date": written, "slug": slug, "label": label}


def _pools(profile: dict) -> dict[str, list[dict]]:
    ed = profile["education"][0]
    return {
        "credential_options": ed["credential_options"],
        "coursework_pool": ed["coursework_pool"],
        "skills_pool": profile["skills_pool"],
        "activities_pool": profile["activities_pool"],
        "interests_pool": profile["interests_pool"],
    }


def check(proposal: dict, profile: dict, entries: list[dict]) -> list[str]:
    """Problems that make a proposal unappliable, phrased for the model to fix."""
    problems: list[str] = []
    idx = corpus.index(profile, entries)
    if not proposal.get("bullets"):
        problems.append("`bullets` is empty: every bullet on the resume needs a row")

    new_ids = [e["id"] for e in proposal.get("new_entries", [])]
    for new_id in new_ids:
        if new_id in idx["entries"]:
            problems.append(
                f"new entry {new_id!r} already exists; changes to it belong in entry_updates"
            )
    known_entries = set(idx["entries"]) | set(new_ids)

    for row in proposal.get("bullets", []):
        label = f"bullet {row['text'][:60]!r}"
        if row["entry_id"] not in known_entries:
            problems.append(f"{label}: entry_id {row['entry_id']!r} is not in the corpus or new_entries")
        if row["action"] != "new" and row.get("existing_id") not in idx["bullets"]:
            problems.append(
                f"{label}: action {row['action']!r} needs an existing_id from the corpus, "
                f"and {row.get('existing_id')!r} is not one"
            )

    for update in proposal.get("entry_updates", []):
        if update["entry_id"] not in idx["entries"]:
            problems.append(f"entry_updates names unknown entry {update['entry_id']!r}")

    pools = _pools(profile)
    for change in proposal.get("pool_changes", []):
        if change["action"] == "rename" and change.get("existing_id") not in {
            item["id"] for item in pools[change["pool"]]
        }:
            problems.append(
                f"pool rename of {change['text']!r}: {change.get('existing_id')!r} is not "
                f"an id in {change['pool']}"
            )
    return problems


def propose(
    resume_text: str,
    source: dict,
    profile: dict,
    entries: list[dict],
    versions: dict,
    *,
    model: str = DEFAULT_MODEL,
    effort: str = "high",
    client=None,
    max_attempts: int = 3,
) -> dict:
    """One API call (plus retries for unusable output) -> a proposal dict, not yet applied."""
    import anthropic  # here, not at module top, so `merge --apply` needs no API package

    client = client or anthropic.Anthropic()
    detail = profile["education"][0].get("detail_line", "")
    user_content = (
        "# CORPUS\n\n"
        f"{corpus.menu(profile, entries, versions)}\n\n"
        f"Education detail line: {detail}\n\n"
        f"# NEW RESUME: {source['file']} (written {source['date']})\n\n"
        f"{resume_text.strip()}\n\n"
        "Classify every bullet on the new resume against the corpus, and report entry, "
        "education, and footer changes."
    )
    messages: list[dict] = [{"role": "user", "content": user_content}]

    problems: list[str] = []
    for attempt in range(1, max_attempts + 1):
        # Streamed because thinking plus a long proposal can outrun a plain request's
        # timeout. The fallback beta re-runs a declined request on another model:
        # sanctions and cyber wording occasionally trips a classifier.
        with client.beta.messages.stream(
            model=model,
            max_tokens=32000,
            system=SYSTEM,
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined this request.")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("The merge proposal was truncated — raise max_tokens and retry.")

        text = next(b.text for b in response.content if b.type == "text")
        proposal = json.loads(text)
        problems = check(proposal, profile, entries)
        if not problems:
            return {
                "_source": source,
                **proposal,
                "_usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "model": response.model,
                    "attempts": attempt,
                },
            }

        messages = messages[:1] + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": "That proposal cannot be applied:\n- "
                + "\n- ".join(problems)
                + "\n\nRedo it completely, using only ids that appear in the corpus above.",
            },
        ]

    raise RuntimeError(
        f"The model returned an unusable proposal after {max_attempts} attempts:\n  - "
        + "\n  - ".join(problems)
    )


def _set_alt_group(bullet: dict, group: str) -> None:
    """Set alt_group just before resume_versions, where the rest of the file keeps it."""
    if bullet.get("alt_group") == group:
        return
    versions = bullet.pop("resume_versions", [])
    bullet["alt_group"] = group
    bullet["resume_versions"] = versions


def _next_bullet_id(entry: dict, taken: set[str]) -> str:
    ids = [b["id"] for b in entry["bullets"]]
    prefix = ids[0].rsplit("-", 1)[0] if ids and "-" in ids[0] else entry["id"]
    numbers = [
        int(m.group(1)) for i in ids if (m := re.fullmatch(re.escape(prefix) + r"-(\d+)", i))
    ]
    n = max(numbers, default=0) + 1
    while f"{prefix}-{n:02d}" in taken:
        n += 1
    return f"{prefix}-{n:02d}"


def apply(
    proposal: dict, profile: dict, entries: list[dict], versions: dict
) -> tuple[dict, list[dict], dict, list[str]]:
    """Proposal -> (profile, entries, versions, change log). Inputs are not modified."""
    source = proposal["_source"]
    slug, written = source["slug"], source["date"]
    if slug in versions:
        raise corpus.CorpusError(
            f"{slug!r} is already in versions.json, so this proposal was applied already. "
            f"To merge the file again, re-run `merge` with a different --slug."
        )
    problems = check(proposal, profile, entries)
    if problems:
        raise corpus.CorpusError("Proposal does not match the corpus:\n  - " + "\n  - ".join(problems))

    profile, entries, versions = (copy.deepcopy(x) for x in (profile, entries, versions))

    log = [f"! conflict: {c}" for c in proposal.get("conflicts", [])]
    newest_overall = max((v["date"] for v in versions.values()), default="")
    versions[slug] = {
        "date": written,
        "file": source["file"],
        "label": source["label"],
        "domains": proposal["resume_domains"],
    }
    # Newest first, so versions.json reads as a timeline. Stable for same-day ties.
    versions = dict(sorted(versions.items(), key=lambda kv: kv[1]["date"], reverse=True))
    log.append(f"+ version {slug}: {source['file']}, written {written}")

    def entry_newest(entry: dict) -> str:
        return max((corpus.newest_date(b, versions) for b in entry["bullets"]), default="")

    by_entry = {e["id"]: e for e in entries}

    # A role the corpus has never seen is almost always the newest one, so it goes on top.
    for position, new in enumerate(proposal.get("new_entries", [])):
        entry = {**new, "bullets": []}
        entries.insert(position, entry)
        by_entry[entry["id"]] = entry
        log.append(f"+ entry {entry['id']}: {entry['org']} — {entry['role']} ({entry['dates']})")

    for update in proposal.get("entry_updates", []):
        entry = by_entry[update["entry_id"]]
        fields = {
            k: clean(update[k])
            for k in ("org", "role", "location", "dates")
            if update.get(k) and clean(update[k]) != entry[k]
        }
        if not fields:
            continue
        if written < entry_newest(entry):
            log.append(
                f"- kept {entry['id']} {'/'.join(fields)}: a newer resume "
                f"({entry_newest(entry)}) already uses the current values"
            )
            continue
        for key, value in fields.items():
            log.append(f"~ {entry['id']} {key}: {entry[key]!r} -> {value!r}")
            entry[key] = value

    by_bullet = {b["id"]: (e, b) for e in entries for b in e["bullets"]}
    confirmed: list[str] = []
    touched: set[str] = set()

    for row in proposal["bullets"]:
        text = clean(row["text"])
        action = row["action"]
        found = by_bullet.get(row.get("existing_id") or "")

        # The text outranks the label in both directions.
        if action != "same":
            twin = next(
                (pair for pair in by_bullet.values() if normalize(pair[1]["text"]) == normalize(text)),
                None,
            )
            if twin:
                action, found = "same", twin
        if action == "same" and normalize(found[1]["text"]) != normalize(text):
            action = "reword"

        if action == "same":
            entry, bullet = found
            if slug not in bullet["resume_versions"]:
                bullet["resume_versions"].append(slug)
            confirmed.append(bullet["id"])

        elif action == "reword":
            entry, bullet = found
            newer = corpus.newest_date(bullet, versions)
            if written < newer:
                log.append(
                    f"- kept {bullet['id']}: its wording is on a newer resume ({newer})\n"
                    f"      skipped: {text}"
                )
                continue
            log.append(f"~ {bullet['id']} reworded\n      was: {bullet['text']}\n      now: {text}")
            bullet["text"] = text
            bullet["resume_versions"] = [slug]

        else:
            if action == "alternate":
                entry, base = found
                group = base.get("alt_group") or f"{base['id']}-alts"
                _set_alt_group(base, group)
            else:
                entry, base, group = by_entry[row["entry_id"]], None, None
            bullet = {
                "id": _next_bullet_id(entry, set(by_bullet)),
                "text": text,
                "domains": row["domains"],
                "tags": row["tags"],
            }
            if group:
                bullet["alt_group"] = group
            bullet["resume_versions"] = [slug]
            entry["bullets"].append(bullet)
            by_bullet[bullet["id"]] = (entry, bullet)
            where = f"alternate of {base['id']} ({group})" if base else f"in {entry['id']}"
            log.append(f"+ {bullet['id']} {where}: {text}")
            missing = [d for d in row["domains"] if d not in entry["domains"]]
            if missing:
                entry["domains"] = entry["domains"] + missing
                log.append(f"~ {entry['id']} domains += {missing}")

        touched.add(entry["id"])

    # The menu promises freshest-first within an entry; keep that true.
    for entry_id in touched:
        by_entry[entry_id]["bullets"].sort(
            key=lambda b: corpus.newest_date(b, versions), reverse=True
        )

    ed = profile["education"][0]
    line = proposal.get("detail_line")
    if line and clean(line) != ed.get("detail_line"):
        if written >= newest_overall:
            log.append(f"~ detail_line: {ed.get('detail_line')!r} -> {clean(line)!r}")
            ed["detail_line"] = clean(line)
        else:
            log.append(f"- kept detail_line: a newer resume exists; this one says {clean(line)!r}")

    pools = _pools(profile)
    taken = set(corpus.index(profile, entries)["pools"])
    for change in proposal.get("pool_changes", []):
        items = pools[change["pool"]]
        text = clean(change["text"])
        if change["action"] == "rename":
            item = next(i for i in items if i["id"] == change["existing_id"])
            if item["text"] == text:
                continue
            if written < newest_overall:
                log.append(f"- kept {item['id']} {item['text']!r}: a newer resume exists")
                continue
            log.append(f"~ {item['id']}: {item['text']!r} -> {text!r}")
            item["text"] = text
        else:
            if any(normalize(i["text"]) == normalize(text) for i in items):
                continue
            # "Excel" next to "Excel (Financial Modeling)" would let both print on one
            # footer line. Same core item -> leave it for a deliberate rename.
            twin = next((i for i in items if _core(i["text"]) == _core(text)), None)
            if twin:
                log.append(
                    f"- skipped adding {text!r}: same item as {twin['id']} ({twin['text']!r}); "
                    f"to switch wording, make it a rename in the proposal"
                )
                continue
            base = f"{POOL_PREFIX[change['pool']]}-{re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:24].strip('-')}"
            item_id, n = base, 2
            while item_id in taken:
                item_id, n = f"{base}-{n}", n + 1
            taken.add(item_id)
            items.append({"id": item_id, "text": text, "domains": change["domains"], "tags": change["tags"]})
            log.append(f"+ {change['pool']} {item_id}: {text}")

    if confirmed:
        log.append(f"= {len(confirmed)} bullet(s) unchanged, now credited to {slug}: {', '.join(confirmed)}")
    return profile, entries, versions, log
