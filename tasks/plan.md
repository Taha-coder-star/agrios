# Implementation Plan: Pre-Symptomatic Wheat Stem Rust Detection

Spec: `SPEC.md` (v2, approved) · Research schedule: `PLAN.md` · Quality bar: `CONSTRAINTS.md`
Task list: `tasks/todo.md` (the default target; no external tracker)

## Overview
Build a leak-free pipeline that measures whether stem rust is detectable at DPI 4–5 from
hyperspectral images. It runs in this order: data → evaluation → classical baseline → band
selection → deep models → headline CV and cross-experiment results → interpretability and
sensor simulation → demo → report. The tasks are ordered so the guard code (splits, evaluation,
leakage-safe preprocessing) is written and tested on **synthetic data before the download
finishes**. Every later result then runs through code that already enforces the hard rules.

## Architecture decisions
- **Guards first, on synthetic data.** `splits`, `evaluation` and `preprocess` are pure
  functions tested against synthetic indexes and cubes. Real data never enters tests, so
  `pytest` runs on local Windows in seconds. Tests that need real data use a
  `requires_data` marker that skips with a stated reason when `data/` is absent.
- **One source of truth for pots.** Every fold, split and role comes from `data/splits.json`,
  keyed by `(experiment, pot_id)`. No module recomputes splits.
- **Out-of-fold predictions are the currency.** Every model writes per-image out-of-fold
  predictions (`oof.csv`: experiment, pot, dpi, label, fold, prob). Metrics, thresholds, lead
  time, the demo and the report all read those files, so there are no fresh numbers anywhere.
- **Fitted transforms are objects with `fit(train_idx)`.** Normalisers and the outlier filter
  can only see training indices (CONSTRAINTS R2/R3). Tests assert this.
- **Classical models are the guaranteed result.** They run on CPU and cover the full nested CV
  and cross-experiment tests. Deep models are added on top, and their compute can flex.
- **Git:** one branch per day (`day-NN-<topic>`), merged into `main` after each checkpoint.

## Dependency graph
```
T1 env ─┬─ T2 splits ──┐
        ├─ T3 evaluation┤
        └─ T4 preprocess┘──(checkpoint A)
T5 download ─ T6 index ─ T7 confound/EDA ──(checkpoint B: pot IDs + confound verdict)
            T8 loader+masks ─ T9 caches ─ T10 real splits ──(checkpoint C)
                    ┌──────────────┴──────────────┐
          T11 spectra preproc              T16 torch harness
          T12 classical nested CV          T17 SimpleCNN + 1D-CNN
          T13 classical report             (checkpoint E)
          (checkpoint D)                   T18 HybridSN + weighted loss + ablations
          T14 indices                      (checkpoint F)
          T15 in-fold band selection
                    └──────────────┬──────────────┘
                 T19 deep CV + cross-experiment A↔B
                 T20 thresholds, lead time, curves, stats ──(checkpoint G)
                 T21 gradients+SHAP · T22 simulated sensors · T23 pixel maps ──(checkpoint H)
                 T24 predict/export ─ T25 Gradio + MODEL_CARD
                 T26 figures+README ─ T27 report+slides ─ T28 fresh-eyes audit ──(final)
```

## Task index (details in `tasks/todo.md`)

| Phase | Tasks | PLAN.md day | Branch |
|---|---|---|---|
| 1 Foundation (no data needed) | T1–T4 | Day 0–1 | `day-01-setup` |
| 2 Data | T5–T10 | Day 0–2 | `day-01-setup`, `day-02-pipeline` |
| 3 Classical | T11–T13 | Day 3 | `day-03-classical` |
| 4 Bands | T14–T15 | Day 4 | `day-04-bands` |
| 5 Deep | T16–T18 | Day 5–6 | `day-05-deep`, `day-06-deep2` |
| 6 Headline evaluation (protected) | T19–T20 | Day 7 | `day-07-cv` |
| 7 Explain and sensors | T21–T23 | Day 8 | `day-08-explain` |
| 8 Demo | T24–T25 | Day 9 | `day-09-demo` |
| 9 Report (protected) | T26–T28 | Day 10 | `day-10-report` |

## Parallelisation
- T2, T3 and T4 are independent once T1 exists.
- T5 (download) runs in the background while T1–T4 are built.
- After checkpoint C, the classical/bands track (T11–T15) and the deep track (T16–T18) are
  independent until T19.
- T21, T22 and T23 are independent.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Pot IDs not recoverable (T6) | High: stops the project | Found at checkpoint B, before any modelling. Check the paper and contact the authors |
| Treatment confound (no mock inoculation) | High: changes the claim | T7 temporal test + author email. Becomes the lead limitation if unresolved |
| Drive quota / disk (62 GB) | Med | Stream-and-cache in T5. Experiment B is needed only by T19 |
| GPU quota runs out | Med | Classical track is complete on its own. Cut list in PLAN.md |
| Results swing across folds (6 infected pots) | Med | Pot cluster bootstrap; report intervals, not points |
| Selection overfitting in band search | Med | In-fold selection only (T15) |

## Open questions
1. Are pot IDs recoverable? Resolved at checkpoint B.
2. Will the authors reply about mock inoculation? @taha sends the email drafted in T7.
