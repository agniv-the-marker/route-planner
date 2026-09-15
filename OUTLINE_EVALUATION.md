# Structured outline / street search evaluation — 2026-09-04

**The replacement failed its quality gate and was not enabled in the website.**
The existing UI, terrain, popups, GPX download handling, and legacy generation
pipeline remain unchanged. The legacy diffusion dependencies and comparison scripts
were deliberately retained: the approved plan requires passing the replacement's
checks before removing them. No training or further model experiment was started.

## What was implemented

- `src/outlines.py`: data-only `OutlineSpec`, normalized coordinates, 8–48 points,
  explicit closure, positive area, and rejection of self-intersections. Versioned
  normalized-prompt keys and a constrained JSON schema; no generated code or SVG input.
- `src/outline_inference.py`: injectable JSON completion, one geometry repair maximum,
  and a bounded cache of successful outputs. Transport failures remain distinct from
  invalid geometry.
- `src/street_search.py`: directed edge subdivision at at most 50 m, retaining actual
  curves, parallel edges, and original connectivity. No junctions at geometric crossings.
  A 50 m distance field scores 500 m translations, 30° rotations, and 3/5/8/11/14 km
  perimeters. The best 24 centers at least 500 m apart get local refinement.
  Matching uses 32–64 anchors, up to four locations within 300 m, bounded cached
  shortest paths, and joint cyclic dynamic programming in both traversal directions.
  It rejects excessive arc detours, disconnected/short/long/collapsed/out-of-map loops
  and more than 20% repeated physical street length. Scores include normalized
  outline distance, silhouette overlap, and repeated length. Up to three results
  retain chosen transforms, original directed edges, subdivided edge IDs, and coordinates.
  Search is limited to eight seconds, with explicit failures and retained outlines.
- `scripts/modal_outlines.py`: separate Qwen3-4B-Instruct-2507 worker, vLLM 0.10.2,
  Transformers 4.55.4, one L4 maximum, five-minute idle scale-down, temperature 0,
  seed 0, and 768 tokens per attempt. The CPU website does not import this worker.
- Frozen fixtures/descriptions, CPU route evaluation, exact curved GPX exports,
  numbered review sheets, overlays, stage timings, and evaluation summaries under
  `scripts/outline_benchmark.py`, `scripts/review_outlines.py`, and
  `scripts/summarize_outline_benchmark.py`.

This is an evaluation implementation, not a production replacement. In particular,
`StreetSearch` instances must be used serially; the index is currently constructed
in memory at process startup, and caches are bounded process-local caches. A durable
prepared index/cache, CPU website integration, immediate outline streaming, “another
route,” and replacing diffusion controls/hosting are deferred by the failed gate.
The model repository revision was not pinned in this run; package/model/prompt names
and raw responses are recorded, but exact future model reproduction is not guaranteed.

## Measured results

| Check | Result |
| --- | --- |
| Clean vector routing proof | 5/6 have qualifying routes; heart has none |
| Fixture route recognizability | Only circle clearly recognizable in agent review |
| Fixture CPU search median | 6.64 s, excluding one-time index construction |
| Fixture index construction | 77.37 s on the local machine |
| Frozen model descriptions | 30: six familiar shapes, 12 objects, 12 variations |
| Model descriptions evaluated | 11; stopped once seven were invalid |
| Valid model outlines | 4/11, after at most one repair each |
| Absolute upper bound for full outline gate | 23/30, below required 24/30 |
| Valid complete street routes | 3/11 evaluated descriptions |
| Recognizable model outlines / routes | 0 / 0 in preliminary agent review |
| Warm model RPC median / p95 | 33.04 s / 47.41 s, ten measured warm calls |
| First model RPC wall time | 118.76 s |
| Model loading inside first worker | 68.21 s |
| CPU search on four valid model outlines, median / p95 | 1.39 s / 1.54 s |
| Successful route-cache retrievals | 0.97–3.27 ms, three measurements |
| GPU outline-cache latency | Not measured before the early stop |

The model benchmark stopped after 11 recorded outputs; the threshold became
unreachable at output 10 and another output completed before the stop. The remaining
19 descriptions are **unevaluated**, not failures. Even perfect remaining outputs
cannot reach the required 24 recognizable outlines because seven evaluated outputs
are geometrically invalid. This objective failure does not depend on subjective
recognizability ratings. The 18/30 route target was not demonstrated.

Timings are separate Modal model RPC and local CPU stages, not a deployed end-to-end
benchmark. Image build time is outside the first RPC. Failed calls and repairs are
included in the warm model sample. Full 30-case latency percentiles are not available.
Visual review displayed only numeric IDs, but the reviewing agent had previous
knowledge of the prompt set; this is not an independent blinded human study.

Across failed attempts, validation recorded eight empty/self-intersecting outlines,
seven unclosed outlines, and one consecutive duplicate vertex. The four valid
outlines were generic polygons with little resemblance to their intended objects.
High geometric overlap on a street loop therefore does not establish recognizability.

## Rotation and the reported street-distance error

The asymmetric fish regression confirms that normalized screen coordinates survive
conversion to projected meters and back without reflection, and that +90° rotates
counterclockwise on the north-up map. The map's pan/zoom code adds no rotation.
All-angle placement can deliberately select a sideways symbol. Map overlays retain
both the transformed outline and route plus the chosen angle, so this is inspectable.

The reported “Part of this shape is too far from bicycle-accessible streets” message
comes from `BikeRouter.route` in the **legacy fixed-placement** pipeline. The local
server was still running that pipeline. The new search tries placements across SF
and reports a bounded-search failure instead of that immediate fixed-placement
rejection. It was not connected to the UI because its quality gate failed. This
work does **not** claim the legacy website error is fixed.

## Verification

51 Python tests passed, including directed/parallel edges, crossings without junctions,
closure, curved GPX coordinates, bounded paths, timeouts, hash invalidation, rotation,
one-repair behavior, model failure, and the existing web/download/terrain tests.
Compilation and `git diff --check` passed. One existing requests dependency warning
remains. Browser controls were not changed; the prior desktop/mobile checks were not
rerun and are not presented as new browser verification.

## Evidence and reproduction

- [Fixture route sheet](outputs/vector-routing-proof-v2/routes.png),
  [fixture outline sheet](outputs/vector-routing-proof-v2/outlines.png), and
  [fish map alignment](outputs/vector-routing-proof-v2/03-map.png).
- [Model outline sheet](outputs/outline-benchmark-v1/outlines.png),
  [model route sheet](outputs/outline-benchmark-v1/routes.png),
  [raw model results](outputs/outline-benchmark-v1/results.json),
  [route details](outputs/outline-benchmark-v1/routes.json),
  [summary](outputs/outline-benchmark-v1/summary.json), and
  [qualitative review](outputs/outline-benchmark-v1/review.json).
- Initial slower placement proof is preserved in `outputs/vector-routing-proof/`;
  optimized proof is in `outputs/vector-routing-proof-v2/`. Historical diffusion
  outputs were not touched. Outputs remain local, in the existing ignored directory.
- [Modal benchmark run](https://modal.com/apps/nyro-robotics/main/ap-NqnteO86XFkSe4RYIijowq)
  is stopped with zero tasks. A first build failed because changing `__pycache__`
  files were mounted; the retry mounted only the required source files. No inference
  ran in that first failed build. Compatible Transformers dependencies were pinned
  before the retry.
- The supplied credential is configured as the named `nyro-robotics` profile in
  the user's private Modal configuration, mode 0600, outside this repository.
  The user's default profile was not changed. All Modal commands explicitly selected
  `nyro-robotics`.

Use fresh output directories to preserve evidence. These are reproduction commands,
not a recommendation to repeat a failed model experiment:

```bash
.venv/bin/python -m scripts.outline_benchmark --output-dir outputs/new-fixture-proof
MODAL_PROFILE=nyro-robotics .venv/bin/modal run scripts/modal_outlines.py --output-dir outputs/new-outline-benchmark
.venv/bin/python -m scripts.outline_benchmark --routes --output-dir outputs/new-outline-benchmark
GRADIO_ANALYTICS_ENABLED=False .venv/bin/python -m pytest -q
```

The benchmark runner now automatically stops as soon as seven outlines fail, and
probes the first successful outline's cache before continuing. These control-flow
improvements do not change the generation prompt or settings used in this run.

Implementation references: [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
[vLLM 0.10.2 structured outputs](https://docs.vllm.ai/en/v0.10.2/features/structured_outputs.html),
[Modal cold starts](https://modal.com/docs/guide/cold-start), and
[Newson–Krumm map matching](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/map-matching-ACM-GIS-camera-ready.pdf).
