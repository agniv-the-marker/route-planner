# Route Sculptor — Project Context

## What This Project Does
Generates rideable GPX bike routes in San Francisco that trace the shape of a text concept
(e.g., "a wild horse galloping" → horse-shaped bike route on real streets). Lives as a
Gradio site mounted on FastAPI (`src/web.py`), runnable locally or on Modal.

## Current Pipeline (rewritten 2026-09-04/05 — this is the third approach; see below)
```
Text prompt
  → SymbolGenerator (src/symbols.py): exact/alias catalogue match (instant, free), else a
    remote call to Qwen3-4B-Instruct-2507 (vLLM, deployed on Modal as `route-sculptor-symbols`)
    that SELECTS one of ~151 pre-vetted vector silhouettes via guided/schema-constrained JSON.
    The model never draws a shape — it can only pick a catalogue key (or "none") + a small
    tilt/aspect tweak — so it can't return broken geometry.
  → OutlineSpec (src/outlines.py): validated data-only contract — closed, simple, 8-48 points,
    normalized [0,1]. Both the symbol catalogue and the (now unused) freeform LLM path go
    through this same gate.
  → StreetSearch (src/street_search.py): grid search over position × rotation × scale against
    the real 10×10km SF bike graph, then joint cyclic dynamic-programming matching of directed
    street edges to the outline's perimeter. Rejects non-loops, excessive detours, >20%
    repeated street length, out-of-map and collapsed loops. Returns up to 3 alternative valid
    routes, 3-15km, within an 8s budget.
  → GPX export + SVG map overlay + elevation profile (src/generation.py, src/map_view.py,
    src/terrain.py)
```

## Validated Results (2026-09-05 — see SYMBOL_EVALUATION.md for full evidence)
- **151/151 (100%)** of the curated symbol catalogue produced ≥1 valid 3-15km closed route
  against the real prepared SF bike graph (median search 3.8s, budget 8s). Reproduce:
  `.venv/bin/python -m scripts.symbol_benchmark`.
- **6/6** real free-text prompts (not exact catalogue keys) round-tripped through the actually
  deployed Modal LLM (`route-sculptor-symbols`, workspace `nyro-robotics`) to a valid route,
  including a deliberately nonsensical prompt that gracefully resolved to a loosely-related
  symbol instead of erroring.
  Interpretation was slightly "creative" once: "a scary dinosaur roaring" → `skull` rather
  than the more literal `dinosaur` (which does exist in the catalogue) — schema was satisfied
  and the route worked, so not a bug, but a possible prompt-tuning target.
- **Real browser end-to-end check**: typed "a wild horse galloping" into the live site,
  got back a genuinely horse-shaped 10.2km loop in the Richmond/Presidio, working elevation
  profile, GPX download that parses as valid XML, zero console errors. Screenshot at
  `outputs/symbol-routability-v1/browser-horse-galloping.png`.
- All 51 pytest tests pass (`GRADIO_ANALYTICS_ENABLED=False .venv/bin/python -m pytest -q`).

Net: the core "text → recognizable bike route" loop that was previously broken (see History
below) now works reliably. The main remaining gaps are operational, not correctness bugs —
see Known Issues.

## Known Issues / Limitations
- **Hard dependency on the deployed Modal LLM** for any prompt that isn't an exact catalogue
  name — there is no CPU-only fallback. If `route-sculptor-symbols` is stopped or Modal
  credentials change, all non-exact prompts fail with "shape interpreter is unavailable"
  (the exact-match path still works offline).
- **`max_containers=1`** on the symbol worker means concurrent users queue for LLM
  interpretation; fine for a demo, not for real traffic. Idle scale-down is 5 minutes, so the
  first request after a lull pays a ~30-90s cold start.
- **The CPU website itself is not deployed to Modal** — `scripts/modal_app.py` defines the
  `route-sculptor` app but it has never been `modal deploy`ed. The site currently only runs
  locally (`python -m src.web`).
- **Fixed catalogue of ~151 concepts** (`src/assets/symbols.json`, from Material Design Icons
  v7.4.47). Very unusual descriptions get mapped loosely or to "none" (tested, graceful).
- **Bike graph is conservative but not exhaustive**: no turn-restriction relations, no live
  closures.
- Two earlier, now-superseded approaches are intentionally kept as evidence, not live code:
  `outputs/legacy-diffusion-source/` (SD1.5/ControlNet), `src/outline_inference.py` +
  `scripts/modal_outlines.py` (freeform LLM-vector-JSON outlines — see OUTLINE_EVALUATION.md).
  `scripts/outline_benchmark.py` is NOT dead code despite being part of that era — it's shared
  benchmark/test infrastructure still used by `tests/test_generation.py`,
  `scripts/modal_symbols.py`, and `scripts/symbol_benchmark.py`. Don't delete it.

## History: three approaches were tried; this is the one that worked
1. **SD 1.5 (+ optional Canny ControlNet) silhouette generation.** Raster image → binary
   threshold → contour → fixed placement → route. Failed: 0 valid routes across ~48 evaluated
   attempts (incomplete/cropped outlines, connectivity failures, excessive detours). Code
   archived in `outputs/legacy-diffusion-source/`, no longer imported by `src/`.
2. **Freeform LLM-drawn vector outlines.** Qwen3-4B asked to emit 8-48 `[x,y]` points
   directly, validated by `OutlineSpec`, placed by a street-aware search (`street_search.py`).
   The search/routing stage worked (5/6 hand-drawn fixtures routed validly), but the model's
   own drawings were unreliable: 7/11 evaluated outputs were geometrically invalid, and even
   valid ones weren't recognizable as their intended object. Failed its quality gate — see
   `OUTLINE_EVALUATION.md`. Code kept at `src/outline_inference.py` / `scripts/modal_outlines.py`.
3. **Curated symbol catalogue + model-picks-not-draws (current).** Same street-search engine
   as #2, but the shape source is now a fixed library of real, pre-validated icon outlines;
   the model only selects among them. This sidesteps the failure mode of both #1 and #2 by
   construction. See "Validated Results" above and `SYMBOL_EVALUATION.md`.

## Key Implementation Details (don't reintroduce old bugs)
- One shared projected coordinate frame (`src/geo.py`, EPSG:32610 UTM, meters) for pixels,
  streets, routes, and terrain.
- `StreetSearch.transform`: normalized outline space is x-right/y-down; it's flipped to y-up
  exactly once when converting to projected meters, then rotated counterclockwise. Covered by
  `test_rotation_matches_north_up_map_without_mirroring` — don't add a second flip elsewhere.
- `OutlineSpec.parse` is the single validity gate (closed, simple, non-self-intersecting,
  8-48 points, normalized 0-1).
- Route validity in `StreetSearch.match`: closed loop, 3-15km, ≤20% repeated physical street
  length, inside map bounds, not collapsed — rejection reasons are tallied in diagnostics
  rather than silently dropped.
- `StreetSearch.load_prepared()` caches a derived index (`data/search-index/*.npz`,
  gitignored) keyed by graph hash + `VERSION` + map bounds; bump `VERSION` in
  `street_search.py` if the index format changes. Route results are separately cached in
  `route_cache/routes/*.json` (also gitignored).
- The offline debug-SVG helper in `scripts/outline_benchmark.py` (`svg()`, `contact_sheet()`)
  anchors shapes to the canvas center as of 2026-09-05 (previously anchored top-left, making
  non-square shapes look small/off-center in evidence images — cosmetic only, never affected
  the live site's own map rendering in `src/map_view.py`).

## Infrastructure
- **Modal** (workspace `nyro-robotics` — always pass `MODAL_PROFILE=nyro-robotics`; the
  default `agni-spare` profile is unrelated to this project and should stay empty):
  - `route-sculptor-symbols` (`scripts/modal_symbols.py`): **deployed**. One L4, Qwen3-4B-
    Instruct-2507 via vLLM, scales to zero after 5 min idle.
  - `route-sculptor` (`scripts/modal_app.py`): CPU web host, defined but **not deployed**.
- **Data**: `data/sf.graphml` + `sf.json` (15,701-node/38,573-edge SF bike graph, via
  `scripts/prepare_map.py`), `data/sf-terrain.*` (via `scripts/prepare_terrain.py`),
  `src/assets/symbols.json` (151 curated MDI-derived outlines via `scripts/prepare_symbols.py`).
  `data/search-index/` and `route_cache/` are derived/gitignored and rebuild automatically.
- **Local run**: `GRADIO_ANALYTICS_ENABLED=False .venv/bin/python -m src.web` → http://127.0.0.1:7860
- **Tests**: `GRADIO_ANALYTICS_ENABLED=False .venv/bin/python -m pytest -q` (51 passed)
- **Evidence docs**: `OUTLINE_EVALUATION.md` (approach #2, rejected), `SYMBOL_EVALUATION.md`
  (approach #3, current, passed), `PICKUP.md` (2026-09-04 session handoff notes — predates
  the symbol catalogue pivot in places; SYMBOL_EVALUATION.md and this file supersede it where
  they conflict).

## Next Steps (genuinely open, none are correctness bugs)
1. Decide whether to deploy the website publicly (`MODAL_PROFILE=nyro-robotics modal deploy
   scripts/modal_app.py`) — outward-facing and costs ongoing money, so a deliberate choice
   rather than a default.
2. Optional: a CPU-only symbol-matching fallback (e.g. keyword/embedding match against
   catalogue interpretations) so the site degrades gracefully instead of failing outright if
   the Modal LLM is unreachable.
3. Optional: tune `selection_prompt()` in `src/symbols.py` so literal catalogue matches (e.g.
   "dinosaur") aren't overridden by looser associations, if literal matching matters more than
   thematic matching for product quality.
4. Optional housekeeping: `src/outline_inference.py` and `scripts/modal_outlines.py` (the
   rejected approach #2) could be deleted now that #3 has cleared the bar they failed;
   currently kept as evidence per the project's existing convention of preserving history.
