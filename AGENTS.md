# Route Sculptor context

Read `PICKUP.md` for current state and `docs/ROUTING_MILESTONE.md` for the research contract. Preserve the existing uncommitted rewrite and historical outputs; do not restore deleted legacy code just because Git is dirty. This implementation added a separate experimental workflow and did not replace the website.

The current website (`src/web.py`) selects from 151 catalogue silhouettes via `src/symbols.py`, then uses historical `src/street_search.py` on the fixed SF map. 151/151 valid routes is **routability evidence only**. It does not prove recognition, requested pose preservation, or arbitrary-text capability. The generated-vector baseline compiled 11/120 and produced zero qualifying routes. A skull substituted for a dinosaur is a prompt-fidelity failure, regardless of route validity.

New research: `src/strokes.py`, `src/stroke_router.py`, `src/regions.py`, `src/experiment_journal.py`; runners and preview under `scripts/routing_*`, `scripts/prepare_region.py`, `scripts/modal_review.py`. Quality precedes generation research. Human source acceptance is required before routing the 12 proposed references; four tune parameters and eight are frozen for comparison. No acceptance exists yet. Keep ±15° upright, no mirroring, visible transfers, directed continuity, 15/30/50/80 km ride limits, and explicit incomplete results. Never substitute catalogue shapes for failed generated subjects.

`experiments/budget.json` protects a $100 new experiment cap, including $10 preview overhead. Reserve every paid allocation using checked prices and a runtime bound. Do not repeat uncertain remote inference without reconciliation. Use `MODAL_PROFILE=nyro-robotics` explicitly; leave `agni-spare` untouched. A CPU preview serves saved artifacts only; GPUs remain behind experiment runners.

`src/geo.py` is the common EPSG:32610 coordinate frame. Flip screen y exactly once. Preserve curved/parallel edge geometry and topology at overpasses. The regional graph is separate from historical `data/sf.graphml`; neither graph implements turn-restriction relations or live closures. `scripts/outline_benchmark.py` is shared fixture infrastructure, not dead code.

Prior context is archived under `docs/history/`. Use measured tests and recorded runs, not the old reliability claims. Do not start diffusion or RL training until the routing and human recognition gates have been evaluated.
