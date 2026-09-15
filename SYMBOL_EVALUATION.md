# Symbol-catalogue pipeline evaluation — 2026-09-05

**This replacement passed its quality gate and is the pipeline wired into the website today.**
It supersedes both approaches documented in [OUTLINE_EVALUATION.md](OUTLINE_EVALUATION.md):
SD 1.5 / ControlNet silhouette generation, and free-form LLM-drawn vector outlines. Both
failed for the same underlying reason: the *shape source* was unreliable (raster silhouettes
were often cropped or patterned; LLM-drawn vector points were often geometrically invalid or
unrecognizable once valid). This approach removes that failure mode by construction — the
model never draws a shape, it only *selects* one from a hand-vetted library of real icon
outlines. The street-search/routing stage that those two evaluations already validated
(5/6 valid routes on hand-drawn fixtures) is unchanged.

## What was implemented (previous session, not yet measured broadly before this evaluation)

- `src/symbols.py` (`SymbolGenerator`): normalized-prompt exact/alias lookup (instant, free,
  no model call) with a Modal-hosted LLM fallback for everything else — Qwen3-4B-Instruct-2507
  via vLLM on one L4, guided/structured JSON decoding constrained to
  `{symbol: enum[151 names + none], tilt: {-20,0,20}, aspect: {normal,tall,wide}}`. The model
  cannot emit a novel or invalid shape; it can only pick a catalogue key. Selections are
  persisted to a disk cache keyed by prompt + catalogue version.
- `scripts/prepare_symbols.py`: compiles pinned Material Design Icons SVGs (v7.4.47) into
  `src/assets/symbols.json` — outer boundary only, simplified to 8–48 vertices, validated as a
  simple closed polygon via shapely, small gaps joined by buffering. 151 of the requested
  names survived (a few excluded for having no simple closed outer path).
- `src/generation.py` / `src/web.py`: rewired to call `SymbolGenerator` → `StreetSearch`
  directly. The SD 1.5 / ControlNet image pipeline is no longer imported by `src/` at all
  (moved to `outputs/legacy-diffusion-source/` for reference).
- `src/street_search.py` (`street-search-v6`, unchanged by this evaluation): grid search over
  position × rotation × scale, then joint cyclic DP matching of directed street edges to the
  outline perimeter.

## Measured results (this evaluation, 2026-09-05)

1. **Full catalogue sweep** — every one of the 151 symbols in `src/assets/symbols.json`, at
   its default orientation, run through `StreetSearch.load_prepared().search()` against the
   real prepared SF bike graph (15,701 nodes / 38,573 edges; 59,985 subdivided search
   locations after 50 m edge subdivision). **151/151 (100%) produced at least one valid
   3–15 km closed route** within the 8-second search budget; median search time 3.8 s on a
   cold cache pass, effectively instant on a warm one. Zero failures.
   Reproduce: `.venv/bin/python -m scripts.symbol_benchmark`.
   Raw data: [`outputs/symbol-routability-v1/results.json`](outputs/symbol-routability-v1/results.json),
   [`summary.json`](outputs/symbol-routability-v1/summary.json).
2. **Real free-text prompts through the deployed model** — 6 descriptions that are *not*
   exact catalogue keys, sent to the live `route-sculptor-symbols` Modal app (workspace
   `nyro-robotics`). **6/6 produced a valid route**, including a deliberately nonsensical
   prompt ("quantum entanglement between two electrons") that the model mapped to a loosely
   related catalogue symbol (infinity) rather than crashing. Cold start was 67 s total (model
   load ≈36 s + generation ≈7 s); warm calls took 0.7–0.9 s.
   Raw data: [`outputs/symbol-routability-v1/e2e_prompts.json`](outputs/symbol-routability-v1/e2e_prompts.json).
3. **Real browser end-to-end check** — typed "a wild horse galloping" into the running site
   (`python -m src.web`) and submitted through the actual Gradio UI, not a mocked test.
   Result: interpreted as "horse", a genuinely horse-shaped 10.2 km loop placed in the
   Richmond/Presidio, a working elevation profile, zero browser console errors, and a
   downloaded GPX that parses as valid XML with track points.
   Screenshot: [`outputs/symbol-routability-v1/browser-horse-galloping.png`](outputs/symbol-routability-v1/browser-horse-galloping.png).
   GPX: [`browser-horse-galloping.gpx`](outputs/symbol-routability-v1/browser-horse-galloping.gpx).

## What this does not establish

- "Valid route" means the same objective routing gate `OUTLINE_EVALUATION.md` used (closed,
  3–15 km, ≤20% repeated street, inside the map, not collapsed) — it is not a systematic,
  independently-scored recognizability study like the one that failed the previous approach.
  The horse screenshot is one qualitative data point, not a replacement for that kind of review.
- The LLM's symbol choice is sometimes associative rather than literal — e.g. "a scary
  dinosaur roaring" selected `skull` even though `dinosaur` exists in the catalogue. The schema
  was satisfied and the shape routed, so this isn't a defect in this evaluation's terms, but
  it's a prompt-tuning opportunity if literal matches matter more than thematic ones.
- Concurrency, cost, and uptime were not load-tested. The symbol worker allows only one
  container (`max_containers=1`) and scales to zero after 5 minutes idle, so concurrent users
  queue for interpretation and the first request after idle pays a ~30–90 s cold start. The
  CPU website itself (`scripts/modal_app.py`, Modal app `route-sculptor`) is defined but not
  currently deployed; the site only runs locally today.
- While checking these outputs, the offline debug-SVG renderer used by this benchmark
  (`scripts/outline_benchmark.py`'s `svg()`/`contact_sheet()`) was found to anchor shapes to
  the top-left corner of its canvas instead of centering them, making non-square shapes look
  smaller and off-center than necessary. Fixed in this evaluation; it never affected the live
  website's own map rendering (`src/map_view.py`), which uses different code and was already
  verified correctly oriented via the browser screenshot above.

## Reproduction

```bash
.venv/bin/python -m scripts.symbol_benchmark --output-dir outputs/new-symbol-routability
GRADIO_ANALYTICS_ENABLED=False .venv/bin/python -m pytest -q
```

The free-text and browser checks above were exploratory (not scripted into a rerunnable
command); `scripts/outline_benchmark.py`'s `DESCRIPTIONS` list plus `scripts/modal_symbols.py`'s
local entrypoint cover the same ground for a full 30-prompt run against the deployed model:

```bash
MODAL_PROFILE=nyro-robotics .venv/bin/modal run scripts/modal_symbols.py --output-dir outputs/new-symbol-benchmark
.venv/bin/python -m scripts.outline_benchmark --routes --output-dir outputs/new-symbol-benchmark
```
