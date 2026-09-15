# Route Sculptor pickup

## 2026-09-14 website repair

The live website uses Z-Image-Turbo → raster contour → historical SF street search for text, and the directed stroke matcher for canvas drawings. Both interactive modes stay on the historical SF graph; the Peninsula research graph is not the website default. Earlier catalogue-only descriptions below are historical.

Fixed a streamed text crash caused by passing an unplaced OutlineSpec into the map renderer, restored cached road rendering, restored feasible canvas placement sizes, and added route-bound zoom. Visible distances are miles and elevation is feet. `/about` now cites the actual diffusion model and drawing paper instead of claiming text uses catalogue icons. The shape preview shows the extracted contour used for routing; raw generation remains in debug.

Real Chromium verification: canvas square returned three candidates with a 6.0-second search and a 556-point GPX. Text `fish` and `heart` completed through the actual deployed diffusion worker and real SF search with 917-point and 687-point GPX exports respectively. Screenshots, GPX and measured browser checks are in `outputs/website-repair-20260914/`; reproduce with `python -m scripts.verify_website --drawing` or `--prompt fish` (uncached text invokes paid inference). These runs establish working interactions/export, not general prompt-fidelity or recognition guarantees.

Uncached diffusion now reserves bounded inference cost before dispatch and journals Modal call IDs for reconciliation. Six fresh candidate calls were run for these two prompts; their reservations remain at full value until actual billing is reconciled. Cached repeats do not dispatch inference. Use `MODAL_PROFILE=nyro-robotics` explicitly.

Drawing result maps now support click/keyboard selection, drag pan, wheel/button/keyboard zoom, and reset to the fitted route. Real Chromium exercised these controls and downloaded GPX; `drawing-interaction.png` records the selected zoomed card. The full suite passed 109 tests during repair; focused checks were rerun after the final extraction changes. The local server runs at `http://127.0.0.1:7860/`.

## 2026-09-13 interactive drawing addition

Local drawing preview now exists at `/draw`, linked from the website header. It supports drawing directly on the SF map and drawing on a canvas with automatic placement, source overlays, explicit transfers, alternatives and GPX download. See `docs/DRAWING_PREVIEW.md` for server instructions, measured verification and limits. All 102 tests passed; real Chromium map/canvas flows and mobile layout were verified. No paid inference or recognition comparison was run. The experimental StrokeRouter is reused with independent interactive settings; historical runner settings and outputs remain unchanged.

Code inspection also found the current `RouteService` defaults to `DiffusionGenerator`, superseding the older catalogue-only description below. This existing text-handler change was preserved. Source acceptance status remains version-specific: the v2 artifact records acceptance, while older prose below and in the milestone is stale.

Updated 2026-09-05. This file supersedes old recognition/reliability claims. Prior pickup and agent context are preserved in `docs/history/`; all existing uncommitted rewrite changes and historical artifacts were retained.

## Current direction and evidence

Routing first, generation second: recognizable, feature-preserving continuous bike art on SF and the Peninsula through Palo Alto, including coastal/bayside roads. Compare 15/30/50/80 km. Keep ±15° upright without mirroring. Open rides, crossings, internal strokes and retracing are permitted; every transfer must be drawn and exported. Review 8–12 examples at a time. Quality precedes latency.

The website remains the 151-silhouette catalogue demo. Its 151/151 route-validity result does not establish recognition or requested pose fidelity. The separate generated-vector baseline compiled 11/120 and produced zero qualifying routes. Neither arbitrary-text capability nor the proposed six-of-eight human-recognition gate has been established. See `docs/history/INDEX.md` for runs and versions.

## Architecture and completed implementation

- Historical website: `src/web.py`, `src/symbols.py`, `src/street_search.py`, fixed historical `data/sf.graphml`. Historical router settings remain unchanged.
- New experiment: `src/strokes.py` (ordered strokes, explicit connectors), `src/stroke_router.py` (directed shape-cost path search, joint beam transitions, placement/refinement, route edges and line/vertex/footprint metrics).
- Config: `configs/routing-v1.json`. Expanded shortest-path objective and paper-inspired objective share all settings and deterministic work ceilings; each gets a separate 120 s wall limit. Historical variant has its original 8 s/3–15 km settings and excludes incompatible drawings. The expanded matcher differs from the historical cyclic matcher to support open strokes; do not call it bit-for-bit reproduction.
- Region: `src/regions.py`, `scripts/prepare_region.py`. Active `sf-palo-alto-v2` clips out the East Bay. Prepared graph: **114,226 nodes, 268,649 directed edges**; SHA-256 `97e8180a8ccad50f29226ab22e4676450bd20a40ac6c103424e523766b552410`. Metadata and geographic SVG beside the graph. `v1` is the preserved broad bounding-box parent. Historical SF graph remains separate.
- Journal: `src/experiment_journal.py`, versioned raw/stage/artifact records, atomic writes, explicit states, source and graph identities, uncertain-call reconciliation and spend reservations. New map record is in `experiments/routing-first-v1/`. Generation cache hits recompile immutable raw text; benchmark re-evaluations use versioned outputs rather than existing filenames and stale thumbnails.
- References: `configs/routing-references-v2.json`, revised proposed sources under `outputs/routing-first-v2/`. IDs 01–04 tune parameters (heart, house, bicycle, running person); 05–12 compare (star, fish, cat, sailboat, umbrella, butterfly, flower, mountains/sun). Initial ordering draft and reviewed v1 are retained as historical artifacts.
- Runner: `scripts/routing_comparison.py`; rejects unapproved/changed source drawings, freezes comparison parameters, keeps paired outputs and explicit incomplete states, exports anonymous routes/overlays/maps/GPX.
- CPU review: `src/review_preview.py`, `scripts/modal_review.py`, persistent SQLite reviews on Modal volume `route-sculptor-routing-review`. Sources start unreviewed; pass/fail/uncertain labels are explicit. Preview labels do not automatically accept the experiment sources.

## Required next input and exact resume

**The user accepted all v2 source drawings** (“yeah those all pass for me”), recorded in `outputs/routing-first-v2/references.json`. The original v1 review rejected seven drawings because component joins were implied rather than drawn; v2 replaces those with intentional continuous/retraced paths. No eight-drawing comparison has run.

After actual acceptance:

```bash
.venv/bin/python -m scripts.routing_comparison accept --evidence 'ACTUAL USER ACCEPTANCE MESSAGE'
.venv/bin/python -m scripts.routing_comparison tuning
# Tune only on 01–04, then freeze:
.venv/bin/python -m scripts.routing_comparison freeze
.venv/bin/python -m scripts.routing_comparison comparison
```

Preserve partial results as incomplete. Inspect plain routes before revealing subjects; judge distinguishing details afterward. Require at least six of eight comparison drawings to pass recognition and feature review; report uncertainty/ties explicitly. Further generator/correction-loop/diffusion/RL research stays gated on credible routing evidence. Do not replace failed generated subjects with catalogue icons.

Useful commands:

```bash
GRADIO_ANALYTICS_ENABLED=False .venv/bin/python -m pytest -q
GRADIO_ANALYTICS_ENABLED=False .venv/bin/python -m src.web
# Region already prepared: this validates it without downloading:
.venv/bin/python -m scripts.prepare_region
# Reproduce v2 from preserved v1 only in a fresh output/version:
.venv/bin/python -m scripts.prepare_region --source data/sf-palo-alto-v1.graphml
# New preview allocation requires its own recorded bounded reservation:
MODAL_PROFILE=nyro-robotics .venv/bin/modal serve --timeout 3600 scripts/modal_review.py
```

## Budget and running preview

New research budget: $100 total, $10 protected for preview/overhead. `experiments/budget.json` reserves **$7**: $2 across two bounded preview sessions and $5 for the bounded L4 outline SFT (including setup/overhead uncertainty); **$93 unallocated**, of which at most $90 is research. The SFT is the first paid training experiment; actual spend remains unmeasured. Prior experiment charges remain historical and are not silently estimated as zero.

Workspace: **nyro-robotics**, explicitly selected by `MODAL_PROFILE`; leave default `agni-spare` untouched. Current temporary preview:

- URL: https://nyro-robotics--route-sculptor-routing-review-web-dev.modal.run
- Run: https://modal.com/apps/nyro-robotics/main/ap-mvuYbt8s8K9NsELHPbpxWg
- Lifecycle: native `modal serve --timeout 3600`, launched during this implementation; automatic shutdown one hour after startup. Verify current state before promising availability. It is a temporary experimental release, not indefinite deployment.
- CPU-only, one container, hard limit 0.25 cores/256 MiB; maximum continuous one-hour compute approximately $0.014 at prices checked 2026-09-05. $1 reservation includes overhead. Persistent review volume survives shutdown. The `/routes` page now exposes saved map drawings, overlays, GPX files, and incomplete attempts.
- Health passed with 12 sources and `inference_enabled: false`. Initial missing import was fixed; preview reloads source changes, so use `/health` for current release identity. Release/health records are stored under `experiments/`.
- Rollback target: no prior review deployment; stop this preview. Historical website unchanged. Shutdown status: timer armed, not yet verified stopped.

## Verification and limitations

Full suite: **88 passed** outside sandbox. It covers the longer shape-following path beating a shortcut, directed/parallel/curved edges, overpasses without junctions, access filtering, open rides and retracing, GPX agreement, visible connectors, orientation, distance/boundary limits, partial searches, source approval, stale compiler caches, atomic artifacts, resume, spend protection, and persistent review labels. FastAPI/Gradio TestClient tests stall in the sandbox; run them outside it. A headless Firefox screenshot timed out with a graphics error; do not claim successful browser visual verification. A generated plain reference contact sheet was inspected locally.

No real-street recognition results exist for the new solver. Whole-edge anchors, beam pruning and local arc-distance bounds are heuristic limitations. Vertex-to-route distance does not establish semantic feature preservation. Human source approval and routing review remain necessary. The regional map conservatively filters access/direction but does not implement turn-restriction relations or live closures.

## Dated implementation journal

- **2026-09-05 — reconciliation:** preserved existing dirty rewrite and outputs, archived superseded pickup/context, audited shared benchmark imports, corrected recognition claims. No legacy implementations moved.
- **2026-09-05 — routing implementation:** added stroke contract, shape-aware directed search, versioned settings and paired runner. Synthetic shortcut regression chose 300 m shape-following route over 100 m shortcut. Recognition remains unmeasured.
- **2026-09-05 — source decision:** prepared 12 proposed drawings and source-acceptance gate; separated four tuning references and eight comparison references before any routing. User accepted the revised v2 sources.
- **2026-09-05 — map preparation:** downloaded OSM bicycle graph separately; preserved broad v1 parent and clipped v2 to Peninsula geography. Final 114,226 nodes/268,649 edges. Zero paid inference.
- **2026-09-05 — preview:** reserved $1, launched CPU-only one-hour review preview in nyro-robotics, fixed remote import initialization, verified health. Labels persist on volume. Shutdown timer armed; final billing reconciliation pending.
- **2026-09-05 — verification:** 88 tests passed; 33 focused regression tests passed after the final journal/cache edits. No source acceptance, paired comparison, automated-judge calibration, or later model research completed.
- **2026-09-05 — source revision v2:** reviewed local `Source drawing approval.html`; rebuilt house, bicycle, runner, sailboat, umbrella, flower and mountains/sun so all joins are source geometry and retracing is explicit. Preserved v1 review evidence and published a new bounded CPU preview. User accepted all v2 sources.
- **2026-09-06 — tuning evidence:** began the four-drawing × four-distance paired tuning run. Historical heart/house produced three routes at each budget; historical multi-stroke drawings were explicitly excluded. Expanded and paper variants repeatedly reached their recorded work/wall limits without a qualifying route. 51 completed/excluded journal stages are preserved; the second run was stopped after a corridor optimization still left heart incomplete. Do not freeze or run the held-out comparison until the shape-aware solver shows credible tuning routes.
- **2026-09-06 — model direction:** accepted the diagnosis that text-to-good-outline is the current bottleneck. Built `experiments/outline-sft-v2` (604 licensed MDI examples, six held-out families), added explicit QLoRA/SFT and held-out evaluation runners, and reserved $5. Current Modal training run: `ap-9iYjWdAmvmgGMGwEhUgulB`; early loss improved, but no recognition claim is permitted before held-out compile, route and human review gates.
- **2026-09-06 — model training completed:** the bounded QLoRA run finished 300/300 steps on `ap-9iYjWdAmvmgGMGwEhUgulB` and committed `/checkpoints/outline-sft-v4`; final training loss was 0.6497. Held-out inference generated all 24 prompts, but **0/24 compiled** because every output exceeded the compiler's 32-command cap (`Expected 4–32 commands`). This is a failed baseline, not a recognition result. Raw rows remain on the Modal volume at `outline-sft-v4-heldout.json`; the next iteration must constrain decoding or train against shorter programs before routing.
