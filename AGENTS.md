# document-lab

Documents (PDF first, HTML tables second) to typed data that carries its provenance. The pipeline, one stage a
module: `capture` makes a PDF of page images from the screen (the long screenshot); `ocr` lays a Tesseract text layer
over it; `pdf` reads any PDF into the IR; `layout` makes a page's lines and reading order, on which `markdown`
writes it as markdown and `regions` finds its figures, tables and equations from its text alone; `tables` reads a
page's ruled tables cell by cell from its rules and text; `per_page`
(`document-lab pages`) and `paper` lay a document out -- page by page (standards, manuals) or by section (papers). Generic and domain-free: no
consumer's vocabulary here -- schemas live in consumers. Modules and usage: `README.md`.

## Rules here

- Dependency arrow: `paper`, `ai` -> `extract`; `paper` -> `markdown`, `regions`, `pdf`; `per_page` -> `markdown`, `regions`,
  `tables`, `ink`, `pdf`, `png`; `markdown`, `regions`, `tables` -> `layout` -> `ir`; `ink` -> `ir` (`regions` takes its drawings, `tables` a scan's rulings, as boxes, never pixels);
  `extract` -> `plot`, `pdf`, `html` -> `numbers`, `ir`; `plot` -> `glyphs` -> `strokefont`; `strokes` -> `plot`, `ir`; `ocr` -> `ink`, `pdf`
  (a picture measured: its skew; a codec decoded to be turned); `capture` is apart (`cli` -> `session` -> `stitch`; `cli` -> `desktop`, `export`, `ocr`, `files` ->
  `win32`, `export`; `desktop` -> `win32` | `quartz` | `x11` (one per OS, chosen at run time) -> `screen`; `join` ->
  `export`, `stitch`; `export` -> `png`), imported only by `__main__`; `png` imports nothing of ours. PDF bytes are
  `_pdf`'s (the Rust crates): `pdf` reads through it, `capture` (`export`, `join`, `files`) and `ocr` read and write
  through it; PyMuPDF only renders pixels (`pdf.read_raster`, `paper`, `glyphs`). OCR is `ocr`'s alone: no other
  module runs the `tesseract` executable, which is the user's to install, never bundled. The base imports
  without `ai_lab` (`tests/unit/test_ai.py` proves it in a subprocess).
- A test that needs expected values builds a SYNTHETIC PDF (`tests/synthetic.py`); no real PDF is ever committed
  (the repo is public). Three real papers, held locally (`DOCUMENT_LAB_PAPERS=<dir>`, the `papers` part), are checked
  STRUCTURALLY against what each PDF says (`tests/paper/structure.py`), every check paired with a planted wrong truth
  it must report.
- `windows`, `macos`, `linux` (`tests/<os>`: a real window, `tests/capture_viewer.py`, scrolled and captured; one
  body, `tests/capture_part.py`) are OS parts: CI runs each on its hosted runner (`linux` and `macos` after a setup
  giving the runner's interpreter Tk, `linux`'s also starting Xvfb and openbox: `[tool.dev_lab.parts] hosted`). `gpu` (marker-pdf), `vendor` (`DOCUMENT_LAB_VENDOR_PDF=<pdf>`), `papers`
  (`DOCUMENT_LAB_PAPERS=<dir>`), `reference` (`DOCUMENT_LAB_REFERENCE_PDFS=<pdfs or dirs>`: the reader accepted
  against PyMuPDF, the previous implementation, within declared tolerances) and `ocr` (Tesseract) are declared
  resource parts, unproduced on hosted runners: base never selects them, and they FAIL where their resource is
  missing, never skip. `corpus` (`tests/corpus`) is produced on hosted runners: only CC BY 4.0 or US public-domain
  PDFs, fetched from their immutable public URLs, pinned by sha256, never committed (`tests/corpus/sources.py`).
- A value the document does not state is absent, a raster plot raises, an ambiguous match refuses: never a
  silent default.
- No pymupdf4llm: markdown is written from the IR (`markdown.py`). `pymupdf<2` because 2.x requires
  pymupdf4llm and onnxruntime (no cp315 wheel); the venv is the family's Python 3.15.
- Verdicts: `.venv/Scripts/python.exe -m dev_lab.dev.verify -- <paths>`. Family files are rendered
  (`tests/architecture/_famconfig.py`, `python -m dev_lab.dev.famfiles`, `python -m dev_lab.dev.ciworkflows render`).
- Design record: `.claude/memory/2026/11/05/`.

<!-- BEGIN dev-lab family section -->

## Family principles

_Rendered by `dev_lab.dev.agentsection` from `dev_lab.dev.principles`; a hand edit here is refused._

- **First principles** (`first-principles`): Derive from the actual system and current needs, not habit or precedent;
  state shaping assumptions.
- **Discard historical baggage** (`no-historical-baggage`): Remove obsolete structures, duplicates and dead paths no
  current need uses -- never a capability.
- **Occam's razor** (`occams-razor`): Simplest design meeting the need; nothing hypothetical; never a capability.
- **Single source of truth** (`single-source-of-truth`): Each fact lives in one place; all else derives from it.
- **Layered decoupling** (`layered-decoupling`): Depend one way through narrow interfaces; never into internals or
  upward.
- **Names match what they do** (`names-match-behavior`): A name says what the thing does now.
- **No backward compatibility** (`no-backward-compatibility`): No shims, aliases or dual paths for old callers; callers
  move with the change.
- **Progressive disclosure** (`progressive-disclosure`): An entry point states what is needed first and points to
  detail; no duplication across levels.
- **Keep every valuable parameter and option** (`keep-valuable-options`): Generality is a requirement: keep every
  parameter and option a real case uses or plausibly needs. A reference match ADDS a selectable option and never
  replaces or deletes the alternative; numerical method choices are options even with no current caller. Every
  retirement is classified (renamed, baggage, or capability-dropped) and only a user ruling drops a capability.
- **Every wall teaches** (`every-wall-teaches`): Every refusal or failure an agent hits names its cause and its door;
  one that does not is a mechanism gap, and closing it (a hook row, a remedy, a test) comes before the task. Never route
  around a wall or hand the user a command. The loop: docs-src/dev/feedback.md (dev-lab).
- **Fan out without a barrier** (`no-barrier-fan-out`): Parallel work is independent items, each landed the moment it is
  done; nothing waits for the slowest sibling -- no barrier-style orchestration (e.g. a dynamic workflow that joins all
  agents) unless a step truly needs every result. How: docs-src/dev/fanout.md (dev-lab).
- **Deliver an image by its absolute path** (`images-by-absolute-path`): An image handed to the user is named by its
  ABSOLUTE path, never a relative one (IMAGES-BY-ABSOLUTE-PATH).
- **A worktree is named for its task** (`worktree-names-its-task`): A worktree's name has a physical meaning tied to its
  task; a hash-like name (`agent-<hex>`, an appended hash) is forbidden (WORKTREE-NAMES-ITS-TASK).
- **Native first** (`native-first`): Python declares and orchestrates, Rust computes. A stage Rust owns is written in
  Rust from its first line; a Python version of it is only a scratch prototype, never a second implementation;
  acceptance is against references (analytic, the vendor ladder, the previous implementation during a migration), never
  against a Python twin.
- **Everything in its place** (`everything-in-its-place`): Each thing lives in the one owner responsible for it; a
  misplaced thing moves, never leaving a pointer or a copy.
- **No detour** (`no-detour`): Every hop must add something; a rule that forces a detour is the defect. Complexity is
  measured, not guessed.

## Family workflow

_Each rule is cited by ID; its full statement and refusing mechanism live in `dev_lab.dev.rules`._

- `SHARED-CHECKOUT`: A box holds a SHARED, possibly concurrent checkout: there is NO single-agent mode, the remote IS
  the only shared medium, pushing is the obligation rather than the last step of landing, and a verdict cannot be
  delegated to someone who cannot see the code.
- `ONE-BRANCH-PER-SESSION`: The integration branch is the only long-lived branch a session works into: a pushed head is
  `pr/<task-slug>/<feature>`, one feature per PR, and every other branch is merged and then deleted, never left behind;
  a repo`s long-lived branches are DECLARED (the trunk plus the integration branch) so anything else on origin or on
  this box is debt, and unpushed work is reported, never deleted silently.
- `PRS-TARGET-INTEGRATION`: Every pull request targets the integration branch, one opened against the trunk is
  retargeted with a comment saying why, and each child branch carries ONE integration comment (or, without a PR, one
  `integration` commit status) updated in place by `forge integration`, never a new one per tick.
- `DOWNSTREAM-SYNCS-INTEGRATION`: A child branch (a lane, a feat branch, a pull-request head) merges ONLY the
  integration branch, so another child`s work reaches it once, through integration, unless the owner opts in to watch
  named refs recorded in the checkout`s own git config (`branch.<child>.labSyncFrom`), never in a file, and `forge sync`
  merges integration first, then the watched refs, refusing any other ref by name.
- `SUBAGENT-NO-PUSH`: A subagent never publishes (no push, no PR opened or readied, no skipped hook): it commits on its
  lane and hands the SHA back to the main session, which merges and pushes, the forge produces the verdict
  (VERDICTS-FROM-THE-FORGE), and a targeted measure of the tests it touched stays its own.
- `VERDICTS-FROM-THE-FORGE`: A verdict is a fact about (tree, env key, part), produced ONCE after the push by the forge
  serve host that claims its `tests/<part>` part and read from the forge: no pre-push hook runs a test, a full tier
  refuses outside the producer naming the remedy (push, then read the part), a targeted measure needs no claim, and a
  producer cites a stored PASS of its key rather than re-run.
- `TRUNK-MOVES-BY-FAST-FORWARD`: The trunk moves only by fast-forward to an integration commit whose every required
  `tests/<part>` part stands PASS at scope full, through `integrate promote`; it refuses with the missing parts named
  and the emergency door, and refuses a trunk that cannot fast-forward rather than merging into it.
- `MERGE-OR-DELETE`: A merged branch is DELETED and an unmerged one is MERGED, no third state: an `archive/*` tag or
  untracked rename hides work and is debt -- `branchset` lists it, deletes it when a declared branch holds its tip, else
  names it to merge.
- `AGENT-GUARD`: A hand-written test invocation, a bare network git verb and a stash are refused by the shared deny
  machinery rather than by prose, so the refusal is identical in every repo that adopts it.
- `NETWORK-RETRY-THEN-REPORT`: The forge is a REMOTE service: a network verb goes through the retry wrapper rather than
  being called directly, and a write that still fails is REPORTED with its diagnosis instead of hammered.
- `PRODUCTION-ENTRY-POINT`: Reproduce a behavior through the PRODUCTION entry point, never an ad-hoc script handed out
  beside it, and treat a behavior not reachable from there as a gap to FIX, the boundary being the SUBJECT, not the tree
  the code happens to live in.

<!-- END dev-lab family section -->
