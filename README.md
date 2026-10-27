# resumeBot

Job description in, tailored one-page resume PDF out.

Your accomplishments live in a JSON corpus. For each application, Claude picks *which*
of them belong on the page and in what order; a deterministic template turns that
selection into a PDF and guarantees it fits on exactly one page. The model never writes a
word of the resume — it only chooses from bullets you already wrote, so nothing on the
page is invented.

![Sample output rendered from the example corpus](docs/sample-resume.png)

*Rendered from the fictional corpus in [`corpus.example/`](corpus.example/).*

## Why it's built this way

- **Selection, not generation.** The LLM returns ids, never prose. The renderer prints
  corpus text verbatim, which rules out hallucinated metrics or employers by construction.
- **Layout is measured, not predicted.** The fitter renders with headless Chrome, extracts
  text back out of the PDF, and checks what actually wrapped or overflowed — then searches
  for the largest type and loosest spacing that still fit.
- **Structured output that validates.** Every plan is checked (real ids, no duplicate
  phrasings, every claimed domain backed by evidence, enough bullets). A bad plan is sent
  back to the model with the errors, up to three attempts.
- **Hand-editable.** Each run saves its plan as JSON. Reorder or swap bullets and
  re-render with no API call.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then paste your Anthropic API key into it
```

Get a key at <https://console.anthropic.com/settings/keys>. `.env` is gitignored; an
`ANTHROPIC_API_KEY` exported in your shell takes precedence over the file.

Rendering uses headless Google Chrome. Set `CHROME_PATH` if yours lives somewhere unusual.

### Try it without a key

A fresh clone uses the example corpus automatically. `render`, `show`, `audit`, `lint`,
and `gaps` make no API calls:

```bash
python -m resumebot.cli render corpus.example/sample.plan.json --out sample.pdf --open
python -m resumebot.cli audit     # evidence per domain, freshness breakdown
python -m resumebot.cli lint      # corpus text that would print wrong
```

### Use your own material

```bash
cp -r corpus.example corpus       # corpus/ is gitignored, so your data stays local
```

Edit `corpus/*.json` by hand, or drop your past resumes (PDF/DOCX) into `data/` and run
`python -m resumebot.cli ingest` to have Claude extract every bullet into the corpus.
`ingest` is for the first build; after that, add each new resume with
[`merge`](#adding-a-new-resume), which keeps the tags and alt-groups you've tuned.

## Use

```bash
# copy the posting, then:
pbpaste | python -m resumebot.cli tailor --open

# from a file
python -m resumebot.cli tailor jobs/example-defense-tech.txt --open

# inline
python -m resumebot.cli tailor --text "Analyst, Sanctions & Financial Intelligence..."
```

**Prefer `pbpaste |` over pasting into the prompt.** Terminal line discipline cuts any
single pasted line at about 1 KB, and job postings are often long unwrapped paragraphs, so
text can vanish silently. Every run prints how much text it received and warns when a line
hits exactly the truncation limit.

Each run writes three files to `out/`:

- `2026-07-28-acme-data-analyst-intern.pdf` — the resume
- `….plan.json` — the exact selection, with the model's reasoning in `notes`
- `….jd.txt` — the posting as received, written *before* the API call, so a failed run
  never costs you the paste

Don't like a choice? Edit the plan and re-render:

```bash
python -m resumebot.cli render out/2026-07-28-acme-data-analyst-intern.plan.json --open
```

Other commands: `show` prints the corpus exactly as the model sees it; `audit` reports
domain coverage and freshness; `lint` and `merge` are covered under
[Keeping the corpus current](#keeping-the-corpus-current).

### What the posting asks for

Alongside the selection, the model lists the posting's requirements as short canonical
labels (`SQL`, `security clearance`) with the corpus ids that demonstrate each. Coverage is
then computed in code against what survived the fitter, not taken on the model's word:

```
  What the posting asks for:
    on page   Python, data pipelines, geospatial tooling, ML model evaluation, SQL
    off page  Excel [skill-excel]
    MISSING   security clearance
```

`off page` names the ids to add if you hand-edit the plan. Across applications,
`python -m resumebot.cli gaps` tallies every saved plan, so a skill four postings in a row
asked for and the corpus can't show becomes a ranked to-do instead of a hunch. Like
`notes`, none of this reaches the PDF.

## Keeping the corpus current

### Adding a new resume

```bash
python -m resumebot.cli merge "data/Resume(Finance).docx"
```

`ingest` rebuilds the corpus from scratch; `merge` folds in one resume and keeps
everything else. Claude classifies each bullet against the corpus: **same**, **reword**
(retires the old wording), **alternate** (joins its alt-group), or **new**. It also
reports entry changes (an end date replacing "Present", a renamed employer), the GPA line,
footer pools, and every fact that contradicts the corpus. Nothing is written yet: the
proposal is saved to `out/merge-<slug>.json` and printed as a change list. Fix any wrong
call in the JSON, then:

```bash
python -m resumebot.cli merge --apply out/merge-finance-v1.json
```

Applying makes no API call and backs up the corpus first. Code, not the model, has the
last word. A bullet labelled "same" whose text changed is applied as a reword. A "new"
bullet already in the corpus verbatim is credited instead of duplicated. A resume older
than the wording it would replace can't overwrite it. A footer item that differs only by a
qualifier ("Excel" vs "Excel (Financial Modeling)") is never added as a second item. The
resume is registered in `versions.json`, dated from the file or `--date`.

### Lint

`python -m resumebot.cli lint` checks the corpus for text that would print wrong:

- present tense under a role that has ended (`Designing…` under `– May 2026`)
- wording that dates itself (`upcoming`, `incoming`, `currently`)
- two bullets in one entry that disagree on a quantity (`40 clients` vs
  `35+ client accounts`; `112%` vs `2.3x` improvement)
- near-identical bullets without a shared alt-group, which could print together
- a title that differs between an entry and the activities line
- `– Present` entries that no recent resume has used
- broken references: unregistered resume versions, duplicate ids, one-member alt-groups

`!` findings print wrong as written; `?` findings deserve a look. `tailor` and `render`
repeat any finding about content on the page they just built, `audit` ends with a count,
and `lint` exits non-zero on errors.

## How it works

### The plan format

`selections` is a flat list, one row per bullet, in page order. Sections and entries are
grouped by first appearance, so to reorder anything, move rows:

```json
"selections": [
  {"section": "Professional Experience", "entry_id": "acme-analytics", "bullet_id": "acme-01", "priority": 1},
  {"section": "Projects & Research",     "entry_id": "osint-lab",      "bullet_id": "osint-01", "priority": 1},
  {"section": "Leadership & Experience", "entry_id": "dining-hall",    "bullet_id": "dining-02", "priority": 3}
]
```

It's flat rather than nested for a practical reason: with a nested
`sections → entries → bullets` schema, the model reliably filled every top-level field and
then returned an *empty* innermost array. Flattening to one row per bullet fixed it
outright. `priority` (1 = keep, 3 = cut first) is what the trim loop cuts by, so trimming
follows the model's judgment about this specific job.

### Fitting one page

`render.py` never guesses at layout — it renders, measures the actual PDF, and reacts, on
two independent axes:

1. **Horizontal — no wrapped bullets.** Binary-search a ladder of margin/type-size
   combinations for the *largest* type at which no bullet wraps. Wrapping is detected by
   extracting text from the rendered PDF and checking that each bullet's head and tail
   land on the same line (ligature- and quote-normalized, since some fonts render
   `fi`/`fl` as single glyphs). Footer lines (skills, activities, interests) are checked
   the same way; one that still wraps drops its last item instead of shrinking the page.
2. **Vertical — fill the page.** With type size fixed, binary-search leading and gaps for
   the *loosest* spacing that still lands on one page, so the bottom third never goes empty.
3. **Only then, cut.** If even the tightest spacing overflows, drop one piece at a time and
   re-measure: lowest-priority bullets first (never emptying an entry), then the interests
   line, the activities line, then whole entries from the bottom. Then re-loosen spacing.

A run takes about eight renders, roughly 15 seconds.

### Domains

The corpus targets three career tracks: `finance`, `national-security`, and
`data-science`. Every entry, bullet, course, skill, and activity is labeled with the
tracks it is real evidence for; domain-neutral material (a campus job, a sport) carries an
empty list and is chosen on merit.

The selector must declare which tracks a posting draws on, and the plan is rejected if it
names a track without selecting evidence for it. `audit` counts evidence per track, so a
thin area is a number rather than a hunch — and the fix is writing bullets, not tuning the
prompt.

### Freshness

`versions.json` records when each past resume was written. A bullet's `resume_versions`
resolves to `current`, `recent`, or `archive`, which the model sees next to the text. Newer
phrasing wins ties between alternate wordings of the same accomplishment (`alt_group`), but
freshness is only a tie-breaker — older bullets that say something unique stay selectable.

## The corpus

- **`entries.json`** — role blocks, each with bullets carrying `domains`, retrieval
  `tags`, `resume_versions`, and an optional `alt_group`. `default_section` is a
  suggestion; the model may move an entry between sections when a different framing
  serves the job better.
- **`profile.json`** — the fixed header plus the pools the model picks from: credential
  line variants, coursework, skills, activities, interests. Footer pools are ordered
  most-important-first, since the fitter drops from the end.
- **`versions.json`** — the date of each past resume, driving freshness.

See [`corpus.example/`](corpus.example/) for a complete working example.

## Layout

`templates/resume.html.jinja` is the single source of truth for how the page looks; the
density steps in `render.py` feed it margins, font size, and spacing. Nothing in the model
path knows anything about layout. Fonts: Times New Roman by default, plus `--font calibri`
(via the metric-compatible Carlito), `narrow`, and `helvetica`.

## Files

```
corpus.example/          fictional sample corpus + a hand-written plan (shipped)
corpus/                  your real corpus (gitignored; falls back to corpus.example/)
templates/               HTML/CSS resume template
resumebot/select.py      job description -> structured plan (the one API call)
resumebot/render.py      plan -> HTML -> PDF, one-page fit loop
resumebot/corpus.py      loading, id validation, plan -> view model, audit, saving
resumebot/ingest.py      data/*.pdf|docx -> corpus (first build)
resumebot/merge.py       one new resume -> reviewable proposal -> corpus
resumebot/lint.py        corpus text that would print wrong (no API call)
resumebot/gaps.py        requirement coverage per run and across saved plans
resumebot/cli.py         command line
data/                    your source resumes (gitignored)
out/                     generated PDFs + plans (gitignored)
```
