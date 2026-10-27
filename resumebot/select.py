"""Content selection: job description in, structured plan out.

This module makes the only API call in a tailoring run. It picks *what* goes on the
resume; it never decides how anything is laid out — that is render.py's job.
"""

from __future__ import annotations

import json

import anthropic

from . import corpus

DEFAULT_MODEL = "claude-opus-5"

SYSTEM = """You are a resume editor working from a fixed corpus of a candidate's real \
accomplishments. Given a job description, you select which of those accomplishments \
appear on a one-page resume, and in what order.

Hard rules:
- Only use the ids listed in the corpus. Never invent, reword, merge, or embellish a \
bullet — you are selecting, not writing. The renderer prints the corpus text verbatim.
- Bullets sharing an ALT-GROUP are different phrasings of the same accomplishment. \
Choose at most one per group.
- Only reference bullets that belong to the entry you attached them to.

Domains — this candidate applies into three tracks: `finance`, `national-security`, and \
`data-science`. Every corpus item is labelled with the domains it is evidence for; \
`domains: any` means the item is domain-neutral and stands on its own merit.

- In `domains`, name every track this job description actually draws on. Many postings \
draw on two (a defense-tech analytics role is national-security *and* data-science; a \
sanctions or financial-crime role is national-security *and* finance).
- The resume must carry real evidence for each domain you name. If you call a job \
finance, at least one selected bullet must be finance-labelled.
- Judge each domain on the job's needs, not on how much material the corpus happens to \
hold. Some tracks have more bullets than others; that is an artifact of which resume was \
written most recently, not a signal about the candidate. Do not let a deep bucket crowd \
out a shallow one that the posting cares about more.
- Domain-neutral entries (lifeguarding, sailing, marketing) carry the leadership, \
turnaround, and high-pressure evidence. Use them for those qualities in any domain — but \
they should not outnumber the on-domain material.

Freshness — each bullet is marked `current`, `recent`, or `archive`, meaning how \
recently that exact wording appeared on a resume the candidate actually sent.

- Prefer `current` phrasing. It reflects the candidate's latest, best-edited language.
- This is a tie-breaker between ALT-GROUP members, not a filter. Many `archive` bullets \
state a fact no current bullet states; select those freely when they fit the job.
- When an ALT-GROUP offers both, take the `current` phrasing unless the older wording is \
distinctly better for this posting (e.g. it names a detail the employer cares about).

Selection guidance:
- The resume must fill one page — not overflow it, but not leave the bottom third empty \
either. Aim for 4-6 entries and 15-18 bullets total. The renderer sizes the type to the \
content and trims from the bottom of your priority ranking if it still overflows, so err \
toward including one more good bullet and rank honestly.
- Put the most relevant entry first within its section, and order sections so the \
strongest evidence for *this* job is highest on the page.
- An entry's `default_section` is a suggestion. Move an entry to a different section \
when that framing serves the job better (e.g. a research fellowship reads as \
"Professional Experience" for a research role and as "Leadership & Experience" for a \
banking role). Every section you emit must contain at least one entry; omit empty ones.
- priority 1 = essential to this application, 2 = strong support, 3 = filler that may be \
cut for space. Give each entry at least one priority-1 bullet.
- Pick the credential line, coursework, skills, activities, and interests that a \
recruiter for this specific role would find relevant. Interests are the least important \
line; treat them as low-stakes personality.
- Order coursework, skills, activities, and interests most-important-first. Those four \
lines must each fit on one line, and the renderer drops items from the *end* when one \
runs long — so anything you would be sorry to lose belongs near the front.

In `notes`, explain your framing in 3-5 sentences: what you decided this employer is \
buying, what you led with, what you deliberately cut, and — if the job spans more than \
one domain — how you balanced them. This is for the candidate's terminal, not the PDF.

In `requirements`, list what the posting asks for: 6-12 skills, tools, or qualifications, \
most important first. Give each a short canonical label you would use for the same thing \
in any posting ("SQL", not "strong SQL skills"; "security clearance", not "ability to \
obtain a clearance"), so labels can be tallied across applications. In `evidence_ids`, \
list every corpus id (bullet, skill, course, activity, or credential), selected or not, \
that genuinely demonstrates it; leave it empty when nothing in the corpus does. An empty \
list is how the candidate learns what to build or write next, and a stretched match hides \
that. Like `notes`, this never appears on the resume.

Output shape: `selections` is one flat row per bullet, in the exact order it should appear \
on the page. Sections and entries are grouped by first appearance, so keep every row for \
an entry consecutive, and every entry for a section consecutive.

You are producing a real resume for a real application, not an example of the output \
format. Every array must be populated with actual ids from the corpus above. An empty \
`selections` list, an empty `skill_ids`, or filler text like "placeholder" in `notes` is a \
failed response.

A correctly filled fragment (ids copied verbatim from the corpus):

{
  "target": "Anduril — Data Analyst Intern, Mission Systems",
  "domains": ["national-security", "data-science"],
  "credential_id": "cred-ds-genai",
  "coursework_ids": ["course-aml", "course-econometrics"],
  "selections": [
    {"section": "Projects & Research", "entry_id": "vins-fellow", "bullet_id": "vins-03", "priority": 1},
    {"section": "Projects & Research", "entry_id": "vins-fellow", "bullet_id": "vins-01", "priority": 1},
    {"section": "Projects & Research", "entry_id": "vins-fellow", "bullet_id": "vins-04", "priority": 2},
    {"section": "Projects & Research", "entry_id": "sentinel", "bullet_id": "sentinel-01", "priority": 1},
    {"section": "Leadership & Experience", "entry_id": "cme", "bullet_id": "cme-02", "priority": 3}
  ],
  "skill_ids": ["skill-python", "skill-sql"],
  "activity_ids": ["act-honor"],
  "interest_ids": ["int-camino"],
  "notes": "Anduril is buying end-to-end work on messy geospatial data, so I led with ...",
  "requirements": [
    {"label": "Python", "importance": "required", "evidence_ids": ["skill-python", "vins-03"]},
    {"label": "security clearance", "importance": "preferred", "evidence_ids": []}
  ]
}"""

# One flat row per bullet. Deeper nesting (sections -> entries -> bullets) proved
# unreliable to generate: the innermost array came back empty while every flat field
# filled in correctly. Grouping happens in corpus.resolve() instead.
_SELECTION = {
    "type": "object",
    "properties": {
        "section": {"type": "string", "enum": corpus.SECTION_TITLES},
        "entry_id": {"type": "string"},
        "bullet_id": {"type": "string"},
        "priority": {"type": "integer", "enum": [1, 2, 3]},
    },
    "required": ["section", "entry_id", "bullet_id", "priority"],
    "additionalProperties": False,
}

SCHEMA = {
    "type": "object",
    "properties": {
        "target": {
            "type": "string",
            "description": "Short label for this application, e.g. 'Jane Street — Quant Trading Intern'.",
        },
        "domains": {
            "type": "array",
            "description": "Every career track this posting actually draws on. The "
            "selection must contain real evidence for each one named here.",
            "items": {"type": "string", "enum": corpus.DOMAINS},
        },
        "credential_id": {"type": "string"},
        "coursework_ids": {"type": "array", "items": {"type": "string"}},
        "selections": {
            "type": "array",
            "description": "One row per bullet, in the exact order they should appear on "
            "the page. Sections and entries are grouped by first appearance, so keep all "
            "rows for an entry together and all entries for a section together.",
            "items": _SELECTION,
        },
        "skill_ids": {"type": "array", "items": {"type": "string"}},
        "activity_ids": {"type": "array", "items": {"type": "string"}},
        "interest_ids": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
        "requirements": {
            "type": "array",
            "description": "What the posting asks for, most important first, with the "
            "corpus ids that demonstrate each. Terminal-only; never printed on the page.",
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "1-4 words, canonical across postings, e.g. 'SQL', "
                        "'Tableau', 'A/B testing', 'security clearance'.",
                    },
                    "importance": {"type": "string", "enum": ["required", "preferred"]},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["label", "importance", "evidence_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "target",
        "domains",
        "credential_id",
        "coursework_ids",
        "selections",
        "skill_ids",
        "activity_ids",
        "interest_ids",
        "notes",
        "requirements",
    ],
    "additionalProperties": False,
}


def validate_plan(
    plan: dict, entries: list[dict] | None = None, profile: dict | None = None
) -> list[str]:
    """Structural checks the JSON schema can't express (it forbids minItems)."""
    problems: list[str] = []

    selections = plan.get("selections") or []
    if not selections:
        problems.append("`selections` is empty — no experience was selected")
    elif len(selections) < 6:
        problems.append(f"only {len(selections)} bullets selected — too sparse for a page")

    seen = set()
    for sel in selections:
        key = sel.get("bullet_id")
        if key in seen:
            problems.append(f"bullet {key!r} selected more than once")
        seen.add(key)

    if entries and profile:
        # An unknown id would otherwise surface in corpus.resolve(), after the call is
        # paid for and with no retry. Catching it here sends it back to the model.
        idx = corpus.index(profile, entries)
        for sel in selections:
            owner = idx["bullets"].get(sel.get("bullet_id"), (None,))[0]
            if owner is None:
                problems.append(f"bullet_id {sel.get('bullet_id')!r} is not in the corpus")
            elif owner["id"] != sel.get("entry_id"):
                problems.append(
                    f"bullet {sel['bullet_id']!r} belongs to entry {owner['id']!r}, "
                    f"not {sel.get('entry_id')!r}"
                )
        pool_ids = [plan.get("credential_id")] + [
            i for key in ("coursework_ids", "skill_ids", "activity_ids", "interest_ids")
            for i in plan.get(key) or []
        ]
        for item_id in pool_ids:
            if item_id not in idx["pools"]:
                problems.append(f"{item_id!r} is not a credential/course/skill/activity/interest id")

    if entries and selections:
        # Two phrasings of the same accomplishment must not both appear.
        groups: dict[str, str] = {}
        by_bullet = {b["id"]: b for e in entries for b in e["bullets"]}
        for sel in selections:
            bullet = by_bullet.get(sel.get("bullet_id"))
            group = (bullet or {}).get("alt_group")
            if not group:
                continue
            if group in groups:
                problems.append(
                    f"bullets {groups[group]!r} and {sel['bullet_id']!r} are alternate "
                    f"phrasings of the same accomplishment ({group}) — pick one"
                )
            groups[group] = sel["bullet_id"]

    if entries and selections:
        # A domain the model claims the job needs must be backed by real evidence,
        # otherwise "this is a finance role" is an assertion the page doesn't support.
        by_bullet = {b["id"]: b for e in entries for b in e["bullets"]}
        covered = {
            d
            for sel in selections
            for d in (by_bullet.get(sel.get("bullet_id"), {}).get("domains") or [])
        }
        for domain in plan.get("domains") or []:
            if domain not in covered:
                problems.append(
                    f"you named {domain!r} as a domain for this job but selected no "
                    f"{domain}-labelled bullet"
                )

    if not plan.get("skill_ids"):
        problems.append("`skill_ids` is empty")
    if (plan.get("notes") or "").strip().lower() in {"", "placeholder", "n/a", "none"}:
        problems.append("`notes` was not filled in")

    return problems


def phrasing_notes(plan: dict, entries: list[dict], versions: dict) -> list[str]:
    """Where the plan took older wording over a newer alternative.

    Advisory only — sometimes the archive phrasing really is the better fit, so this
    reports rather than fails.
    """
    by_bullet = {b["id"]: (e, b) for e in entries for b in e["bullets"]}
    tier_rank = [name for name, _ in corpus.FRESHNESS_TIERS]
    notes: list[str] = []

    for sel in plan.get("selections") or []:
        found = by_bullet.get(sel.get("bullet_id"))
        if found is None:
            continue
        owner, bullet = found
        group = bullet.get("alt_group")
        if not group:
            continue
        chosen_tier = corpus.freshness(bullet, versions)[0]
        fresher = [
            alt["id"]
            for alt in owner["bullets"]
            if alt.get("alt_group") == group
            and tier_rank.index(corpus.freshness(alt, versions)[0])
            < tier_rank.index(chosen_tier)
        ]
        if fresher:
            notes.append(
                f"{bullet['id']} ({chosen_tier}) chosen over newer {'/'.join(fresher)}"
            )
    return notes


def build_plan(
    job_description: str,
    profile: dict,
    entries: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    effort: str = "high",
    client: anthropic.Anthropic | None = None,
    max_attempts: int = 3,
    debug_path=None,
) -> dict:
    client = client or anthropic.Anthropic()

    user_content = (
        "# CORPUS\n\n"
        f"{corpus.menu(profile, entries)}\n\n"
        "# JOB DESCRIPTION\n\n"
        f"{job_description.strip()}\n\n"
        "Select the content for a one-page resume targeting this role. Fill every field "
        "with real ids from the corpus above: 15-18 selection rows across 4-6 entries, "
        "and every pool list populated."
    )
    messages: list[dict] = [{"role": "user", "content": user_content}]

    problems: list[str] = []
    for attempt in range(1, max_attempts + 1):
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
            messages=messages,
        )

        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined this request.")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("Selection was truncated — raise max_tokens and retry.")

        text = next(b.text for b in response.content if b.type == "text")
        if debug_path:
            debug_path.write_text(text)
        plan = json.loads(text)
        problems = validate_plan(plan, entries, profile)

        if not problems:
            plan["_usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_read_input_tokens": getattr(
                    response.usage, "cache_read_input_tokens", 0
                ),
                "model": response.model,
                "attempts": attempt,
            }
            return plan

        # Show the model its own bad output and ask for the real thing. Rebuilt from the
        # original user turn each time so failed attempts don't accumulate and anchor.
        messages = messages[:1] + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": (
                    "That response is not usable:\n- "
                    + "\n- ".join(problems)
                    + "\n\nRedo it completely. `selections` must contain 15-18 rows "
                    "covering 4-6 different entries, each row naming a real section, "
                    "entry_id, and bullet_id from the corpus above. Every domain you "
                    "list in `domains` must have at least one bullet labelled with it. "
                    "Write out the full list — a short `selections` array produces a "
                    "half-empty page, which is worse than a debatable bullet choice."
                ),
            },
        ]

    raise RuntimeError(
        "The model returned an unusable plan after "
        f"{max_attempts} attempts:\n  - " + "\n  - ".join(problems)
    )
