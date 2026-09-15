# Routing-first experiment, version 1

The website remains a 151-silhouette catalogue demonstration. Routability is not recognition. The generated-vector baseline compiled 11 of 120 attempts and produced zero qualifying routes (`outputs/generative-v1-final/summary.json`). No arbitrary-text capability or six-of-eight recognition gate has been established.

The new experimental contract in `src/strokes.py` permits open/intersecting strokes and internal features. Stroke order is explicit. Disconnected component endpoints are joined with visible target connectors, which are routed as part of the ride. Placement flips screen y once, rotates within ±15°, and never mirrors. Original directed edges, including curved and parallel geometries, generate the complete route and GPX.

## Solvers and comparison

- **historical**: unmodified `street-search-v8`, historical SF graph and 3–15 km closed-loop settings, 8 seconds. Shapes outside its contract are explicitly excluded. This is contextual evidence, not a budget-matched contestant.
- **expanded**: the existing shortest-path-then-deviation objective adapted to the new ordered stroke/beam matcher and regional placement. This is not a bit-for-bit expansion of the historical cyclic solver: the shared matcher must support open strokes. All expanded settings match the paper variant.
- **paper**: nonnegative length × (1 + weighted sampled distance / normalization) inside directed Dijkstra search. Corner-preserving anchor layers and beam search choose connected transitions jointly; nearby placement candidates are re-matched to refine local paths.

This is an adaptation of [Waschk and Krüger (2019)](https://duepublico2.uni-due.de/servlets/MCRFileNodeServlet/duepublico_derivate_00072443/Waschk_et_al_Automatic_Route_Planning.pdf), not an exact reproduction. Their graph is bidirectional and automatic placement remains future work. Here, overlapping coarse placement neighborhoods use a full directed regional graph, so refinement is not trapped at artificial window boundaries.

Each expanded run receives 600,000 deterministic work units and a separate 120-second ceiling. Work units count placement candidates, path queue pops, edge relaxations, and layer transitions. No warm path caches affect work. The ceiling can yield partial routes marked **incomplete**, which must not be reported as a completed paired comparison. Work parity does not imply identical CPU instruction counts. The bounded beam and arc-distance cap are heuristics, not proofs of feasibility or optimality. Whole-edge anchors can miss useful placements along long streets; this limitation is retained explicitly.

Scores measure symmetric sampled line distance, distance of every source vertex to the route, physical retracing, width/height and convex-hull footprint. Vertex distance is a geometric feature proxy, not a semantic judgment or a guarantee of corner-angle preservation. The rank groups line and vertex error into 10 m bins, then prefers shorter rides and lower retracing. No polygon-overlap recognition claim is made.

## Human gate and resume

`configs/routing-references-v1.json` preserves the proposed 12 drawings. Anonymous sources and continuous traversal previews are in `outputs/routing-first-v1/`. Four sources tune parameters; eight are held for comparison. Obtain explicit source acceptance before routing either split. No acceptance has been recorded.

```bash
# Only after receiving source acceptance, record the actual message:
.venv/bin/python -m scripts.routing_comparison accept --evidence 'ACTUAL USER ACCEPTANCE'
.venv/bin/python -m scripts.routing_comparison tuning
# Adjust configs/routing-v1.json using only tuning results, then freeze:
.venv/bin/python -m scripts.routing_comparison freeze
.venv/bin/python -m scripts.routing_comparison comparison
```

Every drawing runs at 15/30/50/80 km. Evaluation identities include source revision and dirty-source hash, graph identity, configuration, variant, raw source identity and deterministic seed. Completed stage artifacts are hash-checked before reuse. Uncertain remote stages require reconciliation before retry. Raw attempt records are immutable; evaluation revisions use distinct identities. Historical generation cache hits now recompile raw text instead of trusting stale validation. Legacy scripts remain reproduction tools; use the journal for new experiments.

Plain routes precede prompt/detail reveal. Labels are explicit unreviewed/pass/fail/uncertain. Recognition and requested-feature judgments are separate from route validity. At least six of eight comparison drawings must pass human recognition and feature review; ties/uncertainty remain explicit. Automated judges are not calibrated and must not optimize this experiment. Subsequent generator, correction-loop, diffusion and RL research remains gated on credible routing results.

## Budget and preview

The new experiment has a $100 cap, with $90 available to research and $10 protected for preview/overhead. Historical experiment charges are not silently counted as zero or charged to this new allocation. `experiments/budget.json` records full reservations until actual charges are reconciled.

The CPU preview uses one container limited to 0.25 physical cores and 256 MiB, with a native one-hour `modal serve --timeout 3600` lifecycle and a persistent Modal volume for review labels. [Modal pricing](https://modal.com/pricing), checked 2026-09-05: $0.0000131/core-second and $0.00000222/GiB-second. One-hour maximum continuous compute is approximately $0.014; a $1 reservation includes build/storage/overhead uncertainty. No GPU is invoked. This is a temporary experimental release, not an indefinitely deployed production site.

```bash
MODAL_PROFILE=nyro-robotics .venv/bin/modal serve --timeout 3600 scripts/modal_review.py
```

The preview serves saved source artifacts and stores reviews in SQLite on `route-sculptor-routing-review`. Preview review labels do not implicitly authorize source acceptance or model spending. Reconcile storage and runtime charges before further paid reservations.
