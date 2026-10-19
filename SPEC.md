# Spec: Pre-Symptomatic Wheat Stem Rust Detection

Status: **v2, approved 2026-10-19** · @taha · Tasks: `tasks/plan.md`, `tasks/todo.md`
Source plan: `PLAN.md` (corrected 10-day plan). Hard rules: `CLAUDE.md`. Quality bar:
`CONSTRAINTS.md`.

## Assumptions (correct these before approval)
1. The dataset is the public Fedotov et al. 2025 Google Drive release: two independent
   experiments (A, B) of 864 uint16 106-channel TIFFs each, ~62 GB total. Read with `tifffile`.
2. Pot IDs can be recovered from file names or metadata. If not, the project stops (hard rule 8).
   This can't be confirmed until the download.
3. Compute is the free Colab/Kaggle GPU tier. Local Windows is for code, tests and classical
   models only.
4. This is an FYP research codebase, not a production service. "Ship" means a reproducible repo,
   a report and a Gradio demo, not a deployment.
5. Python 3.12. Dependencies are pinned in `requirements.txt`, dev tools in
   `requirements-dev.txt`.
6. The files are already dark/white calibrated by the camera software. Verify on Day 1.
7. Whether the controls were mock-inoculated is **unknown**. The paper describes only the
   infected treatment. It is handled as a confound risk (Day 1 temporal test, email to the
   authors, a limitation in the report), not as a blocker.

## Objective
Show whether wheat stem rust infection can be detected at DPI 4–5, before visible symptoms at
DPI 6–7, from hyperspectral images, using leak-free pot-level evaluation. Identify the
wavelengths responsible and the minimum band count a cheap sensor would need.

Users: the student (@taha), the FYP panel and examiners, and future readers reproducing the
results.

## Capability map

| Module id | Responsibility | Depends on |
|---|---|---|
| `data` | Download/stream, TIFF loader, calibration check, NDVI + K-Means masks, train-only outlier filter, caches, `index.csv`, `splits.json` (outer folds + inner val, A/B experiments) | — |
| `evaluation` | Per-DPI metrics, pot-level accuracy, pot cluster bootstrap, thresholding, lead time, cross-experiment A↔B | `data` |
| `classical` | Preprocessing (SG/SNV/derivative) + LR/SVM/RF/XGB/PLS-DA in nested CV | `data`, `evaluation` |
| `bands` | Vegetation indices, in-fold 2-band search, forward selection, top-5 bands | `classical` |
| `deep` | Training harness, SimpleCNN, 1D-CNN, HybridSN, weighted loss, deep CV | `data`, `evaluation` |
| `explain` | Gradients, SHAP, simulated RGB/multispectral sensors, pixel maps | `deep`, `bands` |
| `demo` | `predict.py`, model export, Gradio app, MODEL_CARD | `deep`, `evaluation` |
| `report` | Figure regeneration script, report, README, audit | all |

Build order: `data` → `evaluation` → `classical` → `bands` / `deep` → `explain` → `demo` →
`report`. This matches the PLAN.md days: Day 0–2 for `data`/`evaluation`, Day 3 for
`classical`, and so on.

## Tech stack
Python 3.12 · NumPy · SciPy · scikit-learn · XGBoost · PyTorch · tifffile · matplotlib · SHAP ·
Gradio · PyYAML. Dev: pytest · pytest-cov · ruff.

## Commands
```
Setup:     python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt -r requirements-dev.txt
Lint:      ruff check src tests
Test:      pytest -q
Coverage:  pytest --cov=src --cov-report=lcov
Index:     python -m src.data.build_index --raw data/raw --out data/index.csv
Cache:     python -m src.data.cache --index data/index.csv --out data/cache
Splits:    python -m src.data.splits --index data/index.csv --out data/splits.json --seed 42
Train:     python -m src.train --config configs/<experiment>.yaml
Evaluate:  python -m src.evaluate --results results/<experiment>
Explain:   python -m src.explain --config configs/<experiment>.yaml
Figures:   python -m src.figures            # regenerates every report figure from results/
Demo:      python app/app.py
```

## Project structure
See `CLAUDE.md § Repo structure`. Other locations:
- `tasks/plan.md` and `tasks/todo.md` hold the /plan output. They are generated after this spec
  is approved.
- `configs/` holds one YAML per experiment. Every run in `results/` names its config.

## Code style
- Plain functions over classes, except PyTorch `nn.Module`/`Dataset`.
- Type hints on public functions. Arrays stay NumPy `float32` in the platform's shape (H, W, B).
- Every experiment entry point takes `--config` and writes `config.json` + `metrics.json` +
  `per_dpi.csv` to `results/<experiment>/`.
- Pot IDs and folds always come from `splits.json`. They are never recomputed inline.

```python
def per_dpi_f1(y_true: np.ndarray, y_pred: np.ndarray, dpi: np.ndarray) -> dict[int, float]:
    """Image-level F1 for each DPI."""
    return {int(d): f1_score(y_true[dpi == d], y_pred[dpi == d]) for d in np.unique(dpi)}
```

## Testing strategy
- pytest, with tests in `tests/` and one file per module (`test_data.py`, `test_evaluation.py`, …).
- **Must-have tests** (these guard the hard rules; IDs match CONSTRAINTS.md):
  - R1: no pot (keyed by `(experiment, pot_id)`) appears in two roles within any fold.
  - Class balance per DPI per split.
  - R2: normalisation statistics are fitted only on training-pot indices. No per-day
    normalisation over all images.
  - R3: the outlier filter sees training indices only, and reports how many images it removed.
  - Pot cluster bootstrap resamples pots, not images. Test it on a synthetic case with a known
    answer.
  - Cache shapes are correct and contain no NaN/inf.
- Tests use small synthetic cubes and never need the real dataset, so they run on local Windows.
- Model code: a smoke test runs one forward/backward step on a tiny tensor for each architecture.
- Sanity experiment, not a unit test: the shuffled-label run must give F1 ≈ 0.5.

## Boundaries
- **Always:** follow the `CLAUDE.md` hard rules and `CONSTRAINTS.md`; run `ruff check` and
  `pytest -q` before each commit; work on a day branch (`day-NN-<topic>`) and merge it into
  `main` after the checklist passes; write
  results to `results/<experiment>/`; add a PROGRESS.md entry each day.
- **Ask first:** adding a dependency not listed above; changing `splits.json` once results
  exist; changing a hard rule; anything that touches test pots outside the final evaluation;
  deleting raw data.
- **Never:** split by image; tune on test pots; commit `data/raw/` or large caches; report a
  number that doesn't trace to a results file; delete or skip a failing test to get to green.

## Success criteria
1. `pytest -q` passes, including every must-have test listed above.
2. A nested pot-level CV over all 12 pots gives per-DPI F1 + pot-level accuracy with pot
   bootstrap 95% CIs, for the best classical model and the final deep model.
2b. Cross-experiment results (train A → test B, and B → A), per DPI, for the same two models.
3. Lead time stated as "flagged by DPI 4 in k/6 inoculated pots, at false-alarm rate Y".
4. Out-of-fold DPI 4 F1 vs number of bands curve (1, 2, 3, 5, 10, 106) and a band-selection
   stability table.
5. Simulated-RGB, top-5 multispectral and full-HSI DPI 4/5 F1 compared in one table.
6. The confound check is documented in DATA_CARD.md: the temporal difference-spectrum test
   (DPI 4–9), imaging order if known, and the author email's status. It is listed as a
   limitation in the report.
7. `python app/app.py` runs from a clean env and shows only out-of-fold predictions.
8. `python -m src.figures` regenerates every report figure, and every number in the report
   traces to a results file.

Note: there is **no F1 target.** The goal is an honest measurement, not a number to hit.

## Decisions (2026-10-19)
- Mock inoculation: undocumented in the paper. Treated as a confound risk, not a blocker (see
  Assumption 7).
- Evaluation: the leave-pot-out nested CV **and** the cross-experiment A↔B test are both
  headline results.
- Background removal: try NDVI and K-Means on Day 2, and choose on preview quality.
- Outlier filter and normalisation: training data only (CONSTRAINTS R2, R3).
- Quality bar: `CONSTRAINTS.md`.
- Git: a project-local repo with `main` as the trunk and one branch per day. The old `master`
  belonged to a repo at the `F:\` drive root, not to this project.
- Schedule: 10 days fixed, Days 6–8 flex, Days 7 and 10 protected.

## Open questions
1. Are pot IDs recoverable? This can't be answered until Day 0/1.
2. Will the authors reply about the mock inoculation? @taha emails them on Day 1.
