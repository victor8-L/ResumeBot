"""Rebuild the corpus from raw resume files in data/.

Run this after you drop a new resume into data/. It extracts text, hands every
version to the model at once, and writes corpus/entries.json + corpus/profile.json.
The shipped corpus was built this way from the existing data/ files, so you only
need this when the source material changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

SYSTEM = """You convert a person's past resumes into a structured corpus of reusable atoms.

You will receive the extracted text of several resume versions of the same person. Merge \
them into one corpus. Rules:

- Every distinct accomplishment becomes one bullet, with the exact wording from the source \
(fix only obvious PDF extraction artifacts like stray line breaks or a typo such as "string \
rapport" -> "strong rapport").
- When two versions phrase the SAME accomplishment differently, keep both as separate \
bullets and give them a shared `alt_group` value so a later step knows to pick only one.
- Group bullets under the role/org they belong to. Use the most complete and most recent \
form of the org name, role title, location, and dates.
- `default_section` is one of "professional", "projects", "leadership".
- Tags are lowercase kebab-case topic labels for retrieval (e.g. "machine-learning", \
"investment-banking", "osint", "leadership"). 3-6 per bullet.
- `domains` is the subset of ["finance", "national-security", "data-science"] that the \
entry or bullet is real evidence for. Be strict: an empty list is correct for material \
that is domain-neutral (lifeguarding, sailing, general marketing), and is better than a \
stretched label. An entry's domains are the union of its bullets'.
- `resume_versions` lists short slugs for the source files a bullet appeared in; reuse \
the slugs already registered in corpus/versions.json where the file matches.
- Order each entry's bullets newest-phrasing-first: a bullet that appears in the most \
recent source file comes before one that only appears in older files.
- Ignore non-resume documents (transcripts, cover letters, essays) entirely.
- Do not invent accomplishments, numbers, employers, or dates."""

_BULLET = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "text": {"type": "string"},
        "domains": {
            "type": "array",
            "items": {"type": "string", "enum": ["finance", "national-security", "data-science"]},
        },
        "tags": {"type": "array", "items": {"type": "string"}},
        "alt_group": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "resume_versions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["id", "text", "domains", "tags", "alt_group", "resume_versions"],
    "additionalProperties": False,
}

_ENTRY = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "org": {"type": "string"},
        "role": {"type": "string"},
        "location": {"type": "string"},
        "dates": {"type": "string"},
        "default_section": {
            "type": "string",
            "enum": ["professional", "projects", "leadership"],
        },
        "domains": {
            "type": "array",
            "items": {"type": "string", "enum": ["finance", "national-security", "data-science"]},
        },
        "tags": {"type": "array", "items": {"type": "string"}},
        "bullets": {"type": "array", "items": _BULLET},
    },
    "required": [
        "id", "org", "role", "location", "dates",
        "default_section", "domains", "tags", "bullets",
    ],
    "additionalProperties": False,
}

SCHEMA = {
    "type": "object",
    "properties": {"entries": {"type": "array", "items": _ENTRY}},
    "required": ["entries"],
    "additionalProperties": False,
}


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    if path.suffix.lower() == ".docx":
        import docx

        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(errors="replace")
    raise ValueError(f"unsupported file type: {path.name}")


def collect(data_dir: Path = DATA_DIR) -> dict[str, str]:
    docs = {}
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".docx", ".txt", ".md"}:
            continue
        try:
            text = extract_text(path)
        except Exception as exc:  # a malformed file shouldn't sink the whole run
            print(f"  ! skipped {path.name}: {exc}")
            continue
        if text.strip():
            docs[path.name] = text
    return docs


def build_entries(
    docs: dict[str, str],
    *,
    model: str = "claude-opus-5",
    client: anthropic.Anthropic | None = None,
) -> list[dict]:
    client = client or anthropic.Anthropic()
    joined = "\n\n".join(
        f"===== FILE: {name} =====\n{text.strip()}" for name, text in docs.items()
    )
    response = client.messages.create(
        model=model,
        max_tokens=32000,
        system=SYSTEM,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": joined}],
    )
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Ingest was truncated — raise max_tokens and retry.")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["entries"]
