# Historical experiment index

All existing outputs and the uncommitted rewrite were retained. Reports describe the implementation used at the time; their conclusions are not current recognition claims.

| Run | Implementation | Evidence |
| --- | --- | --- |
| controlnet-comparison-20260904-194047 | archived SD1.5/ControlNet, fixed 7 km map | 0/36 valid routes; `PICKUP-20260904.md` |
| controlnet-comparison-20260904-200116 | revised prompt, 10 km map | 0/12; `PICKUP-20260904.md` |
| controlnet-comparison-20260904-200839 | paired prompt audit | 0/12, includes repeated baseline seeds |
| outline-benchmark-v1 | Qwen outline / historical StreetSearch revisions | `OUTLINE_EVALUATION.md`; 7/11 invalid outputs |
| vector-routing-proof through v5 | evolving historical StreetSearch | Preserve each output manifest; do not pool across revisions |
| symbol-routability-v1, symbol-benchmark-v1 | catalogue / street-search-v8 | `SYMBOL_EVALUATION.md`; 151/151 route validity, recognition unestablished |
| generative-v1, generative-v1-run2, generative-v1-final | vector-program compiler / street-search-v8 | `GENERATIVE_MILESTONE.md`; final 120 attempts, 11 compiled, zero qualifying routes; do not count rerouted records as new inference |
| routing-first-draft1 | initial proposed source ordering | Superseded draft; never routed or human-approved |
| routing-first-v1 | stroke-router-v1 / routing-v1 | Source approval pending; see `docs/ROUTING_MILESTONE.md` |

Import audit: `scripts/outline_benchmark.py` remains shared by production-adjacent scripts and tests, including fixtures used in the new references. No obsolete module was moved or deleted during this implementation. `outputs/legacy-diffusion-source/` retains the archived pipeline. `PICKUP-20260904.md` and `CLAUDE-before-routing-milestone.md` preserve superseded context verbatim.
