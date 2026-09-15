# Drawing modes and text-to-route research plan

Research date: 2026-09-13. Proposal only; no website implementation, model execution, or paid allocation was performed in this planning pass.

Preserve the current visual design. Rework the functionality around a shared drawing-to-route boundary, so a human drawing can test routing independently of text generation. Arbitrary text-to-recognizable-GPX is an experimental capability until measured.

## Current evidence

- `src/web.py` uses Gradio and a custom SVG map (`src/assets/map.js`). The map currently handles pan/zoom, not stroke capture. This permits an incremental drawing prototype without a frontend rewrite.
- The historical website's 151-shape catalogue is not arbitrary text generation. Its route-validity count does not establish recognition or requested pose fidelity. Implementation inspection on the next turn found that the current `RouteService` now defaults to `DiffusionGenerator` (SD 1.5 raster generation), superseding the older pickup's description of the active text handler. This existing change is preserved; it is not evidence that text-to-route quality has passed.
- `src/strokes.py` already defines normalized ordered strokes, feature references, explicit transfers, and constrained placement. `src/stroke_router.py` supplies a separate directed shape-aware experimental router. Reuse the contracts and graph infrastructure; do not assume the solver is ready for the website.
- `PICKUP.md` records tuning searches reaching work/time ceilings without qualifying experimental routes. The held-out comparison has not passed. It also records the trained outline baseline's 0/24 compile result from exceeding its command cap; training loss is not quality evidence.
- Source-acceptance documentation conflicts: AGENTS.md and the older milestone say no acceptance; the newer pickup and `outputs/routing-first-v2/references.json` record acceptance of v2 with the actual message. Preserve all records and reconcile version-specific status before executing the experiment; do not ask for acceptance again merely because the older prose is stale.

## Product proposal

Keep the existing typography, palette, map treatment and page structure. Add a compact Text / Draw switch; Draw contains On map / On canvas. All modes share region selection, 15/30/50/80 km maximum distance, source overlay, route alternatives and GPX download.

**On map — Best fit.** Draw where the ride should go. Preserve geographic location and scale by default; snap the drawing to a continuous directed road walk. An explicit adjustment control can allow limited translation/scale refinement. Do not silently relocate it across the region. Marking a few important points or locking endpoints is a useful later refinement. This is the simpler routing problem because placement is largely supplied.

**On canvas — Fit to map.** Draw in normalized coordinates, select a supported search region, then search translation, uniform scale and rotation within ±15°, without mirroring. Return up to three distinct valid candidates with length and source overlay. The result means best found within the stated search budget, not a proven global optimum. Manual move/resize should reuse the fixed-placement operation.

Both drawing modes need mouse/touch input, explicit draw/pan tools, undo/redo, clear, stroke deletion, and persistence of the original stroke samples. Preserve aspect ratio during normalization. Simplification must protect corners and marked features. A pen lift does not teleport the cyclist: preview the proposed transfer and count/export its entire routed geometry. Prefer a continuous first sketch, while retaining the existing multi-stroke contract.

**Text.** Generate a small batch of editable source drawings. Show the actual proposed subject and pose before routing, permit selection/editing, then invoke the same canvas placement service. Once reliability is demonstrated, source review could become optional. Never substitute a catalogue subject for failed generation. A good route for the wrong subject is a failure.

## Shared architecture and solver work

Proposed boundary:

`text or pointer strokes -> Drawing + source metadata -> fixed placement or placement search -> directed route -> verified GPX`

Add a placement request alongside Drawing: geographic transform for on-map input, allowed transform ranges for canvas input, region, distance cap, protected features, optional endpoints. Preserve the EPSG:32610 frame and exactly one screen-y flip. Convert pointer coordinates through the SVG transform so zoom/pan does not corrupt strokes.

Split the solver into independently measurable fixed-placement matching and automatic placement. Proposed approach, not a proven improvement:

1. Fix one human-recognizable drawing at one location. Profile queue expansion, candidate construction and geometry costs. Improve this until it produces credible real-street examples before increasing regional search.
2. Use ordered anchors and joint transition selection with geometric deviation inside path cost. Add tangent/corner or ordered-curve comparison where it resolves observed shortcuts. Plain nearest-road snapping and shortest paths between sparse waypoints can omit essential details.
3. Support partial-edge attachment on long streets using temporary split points that preserve directed edges and parallel-edge identity. Preserve bridge/underpass topology; image crossings are not necessarily junctions.
4. For canvas search, cheaply rank location/scale/angle proposals using road distance and orientation fields, then run the exact graph traversal checks on a diverse shortlist. Raster agreement is a proposal heuristic only. Add structural graph retrieval as a later competing proposal method.
5. Keep completed candidates when the search budget expires, record search incompleteness separately, and distinguish timeout, no candidate found and proven constraint violation. A bounded failed search does not prove infeasibility.

Hard validity checks: directed continuity, allowed edges, distance, orientation, complete transfer geometry, and GPX agreement with the traversed edge sequence. Rank valid candidates by ordered shape agreement and protected feature preservation, then length/retracing and geographic preference. Evaluate tangent/Fréchet-style measures as supplements: average distance alone can conceal a missing head, tail or wheel.

The current graphs omit turn-restriction relations and live closures. Before a public claim of navigation-ready reliability, add turn-aware routing and appropriate access validation. GPX geometry alone is not proof of all real-world riding constraints.

## What the literature changes

**Direct continuous curves are the most relevant generation representation.** Magne et al., *Single Line Drawing Generation via Semantics-Driven Optimization* (2026), use diffusion guidance to optimize a rational B-spline that is continuous by construction. Their formulation includes simplicity/style losses. This directly targets the disconnected-stroke problem, but does not establish road fit or cycling feasibility. Our proposed adaptation is to simplify under protected-feature constraints and pass the curve to the graph solver; retracing may need different treatment than their artistic objective. [Paper](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.70502).

The official SLDgen repository provides code and a Docker route, lists at least 24 GB GPU VRAM, and requires access to Stable Diffusion 3.5 Medium. Dependency/model access and applicable licenses need checking before scheduling a pilot. These are reproduction prerequisites, not a runtime or cost estimate. [Official implementation](https://github.com/tanguymagne/SLDgen).

**Image-to-vector is a useful pose-preserving comparison.** CLIPasso converts an object image into a small set of optimized vector strokes, with abstraction controlled through stroke count. It is not text generation or a continuous ride planner. [CLIPasso](https://arxiv.org/abs/2202.05822). SwiftSketch generates vector sketches conditioned on an image; its companion ControlSketch uses depth-conditioned diffusion guidance during sketch optimization. This suggests selecting an accurate reference image first, then extracting an abstract vector sketch. Continuity and transfers still require explicit handling. [Project and paper](https://swiftsketch.github.io/).

**Text-guided vector optimization is an established alternative to edge extraction.** DiffSketcher optimizes vector sketches using a pretrained diffusion model. It is worth comparing if the continuous-curve approach loses important subject details, but multiple independent curves do not automatically form an acceptable ride. [DiffSketcher](https://arxiv.org/abs/2306.14685).

**Fixed-position shape routing has direct precedent.** Waschk and Krüger (2019) use shape-aware path optimization for GPS art. The repository already contains a directed adaptation, not a reproduction; our observed incomplete runs show that citing the method does not validate this implementation. [Paper](https://duepublico2.uni-due.de/servlets/MCRFileNodeServlet/duepublico_derivate_00072443/Waschk_et_al_Automatic_Route_Planning.pdf).

**Ordered Fréchet matching is a stronger fit metric than nearest-road averages.** Chen et al. describe approximate graph matching under strong Fréchet distance, preserving curve order and continuity; their urban experiments report practical speedups over exact methods. A staged solver can use a fast corridor pass, then an ordered free-space or discrete-Fréchet check while retaining directed edges and complete transfers. [Chen et al.](https://geometry.stanford.edu/paper/cdgnw-ammrfd-11/cdgnw-ammrfd-11.pdf).

**Automatic placement also has relevant newer work.** Li and Fu (2026) retrieve graphics from road networks using turning angles, length ratios and graph matching, including one-to-many segment correspondences. This is a candidate source of placement proposals rather than simply scanning every pixel location. Applicability to our directed continuous rides, fixed orientation and distance caps needs separate evaluation. [Paper](https://www.mdpi.com/2220-9964/15/3/98).

**ControlNet provides visual conditioning, not graph feasibility.** The original work supports spatial conditions such as edges, depth and pose. It does not establish that road-image conditioning produces legal routes. [ControlNet](https://arxiv.org/abs/2302.05543). My inference: a dense road map may compete with the subject, and a raster road image discards direction/access and grade-separated topology. Try road-aware refinement only after a recognizable source and plausible placement exist; finish every candidate with graph routing and validation.

## Why Sobel is an insufficient core representation

Sobel measures local image gradients, not the semantic importance of a line. Applied to rendered strokes it can trace both boundaries, and texture/shading can create irrelevant fragments. It supplies neither stroke order nor a connected traversal. A stronger prompt does not impose those structural constraints.

Keep a raster baseline for comparison. For a filled silhouette, segment the foreground and trace its boundary. For a line drawing, use centerline thinning, prune spurs and explicitly construct ordered paths; do not skeletonize a filled silhouette when its exterior is the desired shape. Both still require human feature review and visible connections. These are proposed engineering baselines, not claims of reliable arbitrary-text performance.

My preferred eventual pipeline is text -> faithful reference/continuous vector -> editable accepted drawing -> placement -> graph route. Road-aware vector deformation can be a later alternating loop between routing and small feature-preserving source edits, with explicit limits on semantic drift. It must never change the requested subject to make routing easier.

## Implementation and experimental sequence

1. Preserve the aesthetic; add the input contract and on-map stroke editor behind an experimental mode. Save/reload drawings and expose fixed-placement matching. Verify pointer transforms, transfers and GPX geometry. No GPU required.
2. Diagnose the solver on the four existing tuning references and hand-drawn fixed placements. Keep the eight held-out references untouched. Profile and fix observed failures rather than merely extending timeouts.
3. Add the canvas editor and automatic placement once fixed-placement routing is credible. Use the same drawings to separate placement failures from matching failures. Freeze parameters, then run the held-out comparison under the existing 15/30/50/80 km contract. Require the existing six-of-eight human recognition and feature gate, reporting outcomes by distance and all incomplete attempts.
4. After routing/recognition gates, benchmark existing pretrained generation methods before new training: constrained short vector programs and the simple raster baseline as controls; SLDgen as the priority research candidate; image -> ControlSketch as the pose-preserving alternative. Evaluate in batches of 8–12, using frozen prompts/seeds and equal recorded budgets. Do not train diffusion or RL yet.
5. Integrate the winner into Text using the same source editor and route service. Add road conditioning only if recorded failures suggest it will improve fit without harming recognition.

Generation evaluation must separate compile/parse success, unprompted source recognition, requested feature/pose accuracy, routing completion, and unprompted final-route recognition. Preserve failures in denominators. Use pose-sensitive prompts alongside simple subjects; predeclare which traits matter. An automated visual judge may assist triage but cannot replace uncalibrated human evaluation. For the failed SFT baseline, fix command-budget compatibility before interpreting semantic performance; do not silently raise the cap and claim the old result succeeded.

Modal credits make bounded inference comparisons practical but do not change these gates. Keep `MODAL_PROFILE=nyro-robotics` explicit and leave `agni-spare` untouched. The existing $100 cap and $10 preview reserve remain controlling; the ledger has $7 reserved and actual charges need reconciliation. Check current prices and a runtime/resource ceiling before each new allocation. No credentials belong in this document or the repo. No credit balance was verified in this pass.
