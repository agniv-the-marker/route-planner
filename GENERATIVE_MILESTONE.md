# Generative drawing milestone

The public UI remains a **catalogue demo**. The experimental generator creates geometry from text; it never calls the symbol catalogue or repairs an invalid sample by replacing it with an icon. No training has been run. Recognition and human reward calibration are separate acceptance gates, not inferred from route validity.

## Implemented path

`VectorGenerator(complete, judge=None, cache_dir=None)` preserves the generator injection seam in `RouteService`. The completion callback receives messages and fixed sampling parameters. Four independent candidates retain the exact prompt, raw JSON attempt, parsed vector program, model/revision, seed, validation result, outline, selected street route, scores and timings. The service streams each valid outline before its street search. With a configured `VisionJudge`, candidate selection uses original-prompt route fidelity followed by street fit. Without a judge, results are explicitly experimental and ranked only by street fit; this is insufficient for promotion.

The baseline is `Qwen/Qwen2.5-Coder-7B-Instruct`, revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`, verified from the [publisher model metadata](https://huggingface.co/api/models/Qwen/Qwen2.5-Coder-7B-Instruct). Sampling uses seeds 17/29/43/71, temperature 0.7, top-p 0.95 and 1,536 tokens. The compiler accepts only integer M/L/Q/C/Z commands, one closed path and at most 32 commands. Curve subdivision has a 0.25-grid-unit flatness limit; the simplification error allowance subtracts that amount from 0.5% of bounding-box width. Invalid geometry and outlines requiring more than 48 points are rejected. Short outlines gain collinear midpoint vertices to satisfy the eight-point minimum.

Street search retains every compiled vertex as an anchor, subdivides the largest remaining arcs up to 64 anchors, and applies each arc's own path-length limit. Directed graph connectivity, curved and parallel edges, 3–15 km closure and 20% repeated physical length checks remain. Work budgets count search checks and clear the shortest-path cache to remove cache-warmth dependence. A wall timeout is a separate safety ceiling; incomplete wall-time searches are not cached. Completed successes and failures have graph/configuration-versioned cache keys. Generative service routes use a separate cache directory.

Map route fitting changes only the viewport. North stays up, with a full-SF reset.

## Reproduction and review

```bash
MODAL_PROFILE=nyro-robotics .venv/bin/modal run scripts/modal_vectors.py --output-dir outputs/NEW-DIRECTORY
# Resume/export CPU routing from preserved raw records without further inference:
.venv/bin/python -m scripts.generative_benchmark outputs/NEW-DIRECTORY
.venv/bin/python -m scripts.summarize_vectors outputs/NEW-DIRECTORY --reviews human-reviews.json --comparisons comparisons.json
.venv/bin/python -m pytest -q
```

The first command freezes 30 prompts before inference, then runs a single capped L4 function with no automatic model escalation or sample repair. Its GPU function has a 7,100-second hard timeout and a 6,800-second soft generation cutoff, with at most 120 attempts. Choose a new output directory for every run; previous evidence is never overwritten by inference. CPU routing starts after GPU generation finishes. Files include raw candidate records, evaluated records, plain-canvas outline/route images, numbered contact sheets, GPX downloads, manifest and review HTML. The review page exports explicit human recognition labels, including requested distinguishing features. Summary reports first-sample and best-of-four separately; absent labels never count as recognizable.

`VisionJudge` accepts an injected separate vision-service callback and requires a model and frozen revision. It stores its frozen prompt, original user prompt, raw responses and deterministic balanced outline/route presentation order. Judge failures are excluded from reward. No vision service is configured by default, so judgments remain pending. `scripts.judge_vectors.judge_saved(output, judge)` applies a configured judge to saved candidates without new inference; versioned success/failure caches exclude infrastructure errors. `scripts.vector_comparisons` exports a same-prompt, balanced left/right review page; `VisionJudge.compare` records frozen pair judgments. `balanced_pairs` also produces distinct comparison pairs with balanced display order; `calibration` requires 50 usable human/judge comparisons at 80% agreement. Human reviews cannot be fabricated by the harness.

Reward remains provisional: invalid geometry −1; no qualifying route 0; valid routes 70% final-route prompt fidelity, 20% silhouette overlap, 10% low repetition. Outline prompt fidelity and outline distortion remain separate diagnostics. Test fixtures exercise low semantic scores for wrong subjects, poses, mirrored directional subjects and lost features; a real judge still needs those adversarial evaluations before optimization.

`configs/vector-training.json` and `scripts/vector_training.py` provide SFT/GRPO LoRA configuration and licensed-data validation with subject-family splits. These are scaffolding, not a runnable training job. Training is disabled. Do not propose an RL pilot until the recognition gates and human calibration pass.

## Executed baseline

The completed run is in `outputs/generative-v1-final`; see its `REPORT.md` and `review.html`. All 120 attempts were retained. Eleven attempts compiled, covering at most 10/30 prompts best-of-four (1/30 first-sample). No qualifying street routes were found. The 24/30 outline and 18/30 route acceptance gates failed even before human recognition review. Recorded model loading plus generation took 17.45 minutes on the single-L4 run; no training or model escalation occurred. The finalized implementation passed 75 tests. Human recognition and the 50-comparison reward calibration remain unreviewed.

After reviewing the artifacts, the decoder was tightened for the next run: it now extracts one JSON command array from Markdown fencing or trailing prose while retaining the original completion verbatim. This is syntax normalization only; geometry validation still rejects malformed, self-intersecting, collapsed or feature-losing programs. The recorded baseline numbers above predate this parser change and are intentionally unchanged.

## Text-to-outline training follow-up — 2026-09-06

The routing-first review confirmed that clean catalogue outlines can route while generated outlines remain the bottleneck. A licensed bootstrap corpus is now frozen at `experiments/outline-sft-v2`: 604 examples (four prompt templates × 151 MDI-derived silhouettes), six held-out subject families (`heart`, `dinosaur`, `horse`, `cat`, `fish`, `butterfly`), source URLs and hashes, and zero compiler errors after bounded polygon simplification. The corpus is training data, not recognition evidence; the held-out prompts are evaluated separately.

`scripts/modal_outline_training.py` runs a 300-step QLoRA/SFT job on one L4 using the pinned Qwen revision. It requires an explicit training flag in `scripts/train_outline.py`, writes to a versioned Modal volume checkpoint, and is capped by the recorded `$5` research reservation. The first three launches failed before optimizer steps due to dataset/TRL API issues; those errors are retained in the Modal run logs. The current run `ap-9iYjWdAmvmgGMGwEhUgulB` has completed optimizer steps and is active; early loss fell from 1.65 at step 10 to 0.78 at step 40, which is a training signal only.

After completion, `scripts/modal_outline_eval.py` will generate 24 deterministic held-out prompts and report compiler validity. Valid geometry is necessary but insufficient: inspect plain outlines, then run route fitting and human recognition/feature review. Do not promote the adapter based on loss or compiler validity alone. Route-aware preference optimization remains gated on held-out recognition and calibrated human/judge agreement.
