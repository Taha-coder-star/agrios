# Task list

Every task also clears the standing bar: `ruff check src tests` clean, `pytest -q` green,
CONSTRAINTS floor respected, results written to `results/<experiment>/` with `config.json`.
Sizes: S = 1–2 files, M = 3–5 files.

---

## Phase 1: Foundation (no data needed) · `day-01-setup`

- [x] **T1: Environment and skeleton** (S/M · deps: none)
  - Acceptance: `requirements.txt` (pinned), `requirements-dev.txt` (pytest, pytest-cov,
    ruff), `pyproject.toml` with ruff config (`I`, `B`) and the `requires_data` marker;
    `src/` package dirs per CLAUDE.md; `tests/conftest.py` with a synthetic index and a small
    cube fixture; empty `PROGRESS.md`.
  - Verify: `pip install -r requirements.txt -r requirements-dev.txt`, `ruff check src tests`
    and `pytest -q` all succeed.
  - Files: `requirements*.txt`, `pyproject.toml`, `src/**/__init__.py`, `tests/conftest.py`,
    `PROGRESS.md`

- [x] **T2: Pot-level splits** (S · deps: T1)
  - Acceptance: `make_splits(index, seed)` returns a single 8/2/2 split + 6 outer folds, each
    with an inner validation pair (1 inoculated + 1 control), keyed by `(experiment, pot_id)`
    and written to JSON. CLI `python -m src.data.splits`.
  - Verify: `pytest tests/test_splits.py`, covering R1 (no pot in two roles in any fold), class
    balance per DPI, every pot appearing as test exactly once, and determinism under a seed.
  - Files: `src/data/splits.py`, `tests/test_splits.py`

- [x] **T3: Evaluation core** (S · deps: T1)
  - Acceptance: `per_dpi_f1`, `per_dpi_pot_accuracy`, `pot_bootstrap_ci(oof, metric, n=2000)`
    (resamples pots, not images), and a `metrics.json`/`per_dpi.csv` writer.
  - Verify: `pytest tests/test_evaluation.py`, covering R5 (output has every DPI) and R6
    (synthetic case: duplicate images within a pot don't narrow the CI; known-answer check).
  - Files: `src/evaluation.py`, `tests/test_evaluation.py`

- [x] **T4: Leakage-safe preprocessing** (S · deps: T1)
  - Acceptance: a `BandScaler` with `fit(X[train_idx])`/`transform`, and an
    `OutlierFilter(threshold=0.85)` that fits on training spectra only and reports
    `n_removed`. No API accepts a whole dataset without indices.
  - Verify: `pytest tests/test_leakage.py`, covering R2 (scaler stats don't change when
    test-pot values change) and R3 (filter never drops test indices; count reported).
  - Files: `src/features/preprocess.py`, `tests/test_leakage.py`

### Checkpoint A
- [x] `ruff` + `pytest -q` green; record project coverage and test wall time in CONSTRAINTS
  "Measured" table
- [x] Commit on `day-01-setup`

## Phase 2: Data · `day-01-setup` → `day-02-pipeline`

- [ ] **T5: Download / stream (Day 0)** (S · deps: none, can run alongside T1–T4)
  - Acceptance: a script fetches experiment A (B can follow later) into `data/raw/`, or streams
    each file → cache → delete. Sizes logged to `data/DOWNLOAD_LOG.md`.
  - Verify: manual check that the file count matches 864 for experiment A.
  - Files: `src/data/download.py`, `data/DOWNLOAD_LOG.md`

- [ ] **T6: Inspect files and build the index** (M · deps: T1, T5) **high risk, do early**
  - Acceptance: `DATA_CARD.md` records format, dtype, shape, the wavelength list from the file,
    and calibration status. `data/index.csv` has path, experiment, dpi, label, pot_id,
    acq_order (if any). **If pot IDs aren't recoverable, stop and ask.**
  - Verify: `pytest -m requires_data tests/test_index.py`, checking 864 per experiment, 144 per
    DPI, 72/72 per DPI, and 12 pots per experiment.
  - Files: `src/data/build_index.py`, `data/DATA_CARD.md`, `tests/test_index.py`

- [ ] **T7: Confound check and EDA** (M · deps: T6)
  - Acceptance: `notebooks/01_eda.ipynb` → `results/eda/`. It contains true-colour images at
    DPI 4/6/8; mean ± std spectrum per class and DPI; the difference spectrum per DPI; the
    **difference magnitude vs DPI 4–9** (growing or shrinking verdict written down); and mean
    brightness by class at DPI 4. Also produce a DATA_CARD "Confound" section and an email draft
    to the authors in `docs/author_email.md`.
  - Verify: manual review that the Day 1 checklist in PLAN.md passes.
  - Files: `notebooks/01_eda.ipynb`, `data/DATA_CARD.md`, `docs/author_email.md`

### Checkpoint B (human review, required)
- [ ] Pot IDs confirmed, otherwise stop
- [ ] Confound verdict recorded; @taha has sent the author email
- [ ] PROGRESS.md entry; merge `day-01-setup` → `main`

- [ ] **T8: Loader and background masks** (M · deps: T6)
  - Acceptance: `load_cube(path) -> float32 (H, W, 106)` via tifffile, with calibration applied
    only if DATA_CARD says the data is raw. NDVI > 0.3 and 2-cluster K-Means masks. 10
    previews of each (including DPI 8–9 infected) in `results/masks/`. Mask choice recorded in
    DATA_CARD, based on previews, not F1.
  - Verify: `pytest tests/test_data.py` (synthetic: shapes, dtype, the mask picks the
    "vegetation" region), then a manual preview review.
  - Files: `src/data/loader.py`, `src/data/masking.py`, `tests/test_data.py`

- [ ] **T9: Caches** (M · deps: T8)
  - Acceptance: `data/cache/spectra.npz` (mean ± std, masked + unmasked, mask pixel count) and
    4×-pooled cubes. A full cache load takes under 1 minute. Mask area by class at DPI 4 noted
    in DATA_CARD.
  - Verify: `pytest tests/test_data.py` (no NaN/inf, shapes; real-data cases marked
    `requires_data`), and time the load.
  - Files: `src/data/cache.py`, `tests/test_data.py`

- [ ] **T10: Real splits** (XS · deps: T2, T6)
  - Acceptance: `data/splits.json` generated from the real index with seed 42 and committed.
  - Verify: `pytest tests/test_splits.py -m requires_data` runs the R1 checks on the real file.
  - Files: `data/splits.json`

### Checkpoint C
- [ ] Day 2 checklist in PLAN.md passes; PROGRESS.md entry; merge `day-02-pipeline` → `main`

## Phase 3: Classical baseline · `day-03-classical`

- [ ] **T11: Spectral preprocessing variants** (S · deps: T4)
  - Acceptance: `raw | sg | sg_snv | sg_snv_d1` pipelines built on T4 objects (SNV is
    per-sample, so it needs no fit).
  - Verify: `pytest tests/test_preprocess.py` (shapes; the derivative of a linear spectrum is
    constant; SNV gives mean 0 and std 1).
  - Files: `src/features/preprocess.py`, `tests/test_preprocess.py`

- [ ] **T12: Classical nested-CV runner** (M · deps: T3, T9, T10, T11)
  - Acceptance: `python -m src.train --config configs/classical.yaml` runs LR/SVM/RF/XGB/PLS-DA
    × 4 preprocessings × 2 regimes (all DPIs; DPI 4–5 only) across 6 outer folds, tuning on
    each fold's inner validation pair. It writes `oof.csv`, `per_dpi.csv` and `config.json`
    (including `outliers_removed`).
  - Verify: `pytest tests/test_train_classical.py` (a synthetic 2-fold run gives `oof.csv` with
    every image exactly once), then a real run completes.
  - Files: `src/models/classical.py`, `src/train.py`, `configs/classical.yaml`,
    `tests/test_train_classical.py`

- [ ] **T13: Classical report** (S · deps: T12)
  - Acceptance: a model × preprocessing × DPI table (F1 + pot accuracy + pot-bootstrap CI), a
    per-DPI F1 chart and a DPI 4 confusion matrix in `results/classical/`.
  - Verify: `python -m src.evaluate --results results/classical` regenerates them. Manual
    check: if DPI 4 is near 1.0, re-check splits and the confound notes.
  - Files: `src/evaluate.py`

### Checkpoint D
- [ ] Day 3 checklist passes; no method change made after comparing with the paper; merge

## Phase 4: Bands · `day-04-bands`

- [ ] **T14: Vegetation indices** (S · deps: T9)
  - Acceptance: NDVI, GNDVI, PRI, ARI/ARI2, GARI, red-edge and MGRVI using the nearest real
    wavelengths. The MGRVI definition is checked against Zhu and the source noted in a comment.
  - Verify: `pytest tests/test_indices.py` (a synthetic spectrum with known reflectances gives
    the hand-computed values).
  - Files: `src/features/indices.py`, `tests/test_indices.py`

- [ ] **T15: In-fold band selection** (M · deps: T12, T14)
  - Acceptance: a 5,565-pair normalised-difference search and forward selection (k = 1, 2, 3,
    5, 10), **within each outer fold**, ranked on inner-val DPI 4–5. Outputs: out-of-fold F1 vs
    number of bands, a selection-frequency table, the band-importance plot and
    `results/bands/top5.json`.
  - Verify: `pytest tests/test_bands.py` (on synthetic data with a planted informative band
    pair, the search recovers it; selection never reads test indices), then a real run.
  - Files: `src/features/bands.py`, `configs/bands.yaml`, `tests/test_bands.py`

## Phase 5: Deep · `day-05-deep`, `day-06-deep2`

- [ ] **T16: Torch dataset and training harness** (M · deps: T4, T9, T10)
  - Acceptance: a dataset over pooled cubes with a train-fit scaler (T4); a YAML-driven loop
    (Adam 1e-4, wd 1e-5, ≤ 50 epochs, early stopping on inner-val loss, patience 10, per-epoch
    CSV); deterministic algorithms; a seeds list; out-of-fold output in the same `oof.csv`
    schema.
  - Verify: `pytest tests/test_harness.py` (2 epochs on a tiny synthetic set run end to end; the
    same seed gives the same loss).
  - Files: `src/data/torch_dataset.py`, `src/train.py`, `configs/dl_base.yaml`,
    `tests/test_harness.py`

- [ ] **T17: SimpleCNN and 1D-CNN** (M · deps: T16)
  - Acceptance: SimpleCNN (GroupNorm) and 1D-CNN (mean spectrum + a sampled-pixel variant). A
    3-seed screening run on the single split and a shuffled-label run (F1 ≈ 0.5) go to
    `results/dl_v1/`.
  - Verify: `pytest tests/test_models.py` (forward/backward smoke test per model), plus the
    shuffled-label result.
  - Files: `src/models/simple_cnn.py`, `src/models/cnn1d.py`, `tests/test_models.py`,
    `configs/dl_v1_*.yaml`

### Checkpoint E
- [ ] Day 5 checklist passes; GPU hours used so far logged in PROGRESS.md

- [ ] **T18: HybridSN, weighted loss, ablations, final model choice** (M · deps: T17)
  - Acceptance: HybridSN (SpectralAttentionNet only if ahead of schedule); DPI 4–5 loss weight
    W ∈ {1.0, 1.2, 1.5}; ablations (unmasked / mask / mask + d1) with 3 seeds; the final model
    chosen **on inner-val results only**, with the reason written in `results/dl_v2/CHOICE.md`.
  - Verify: `pytest tests/test_models.py` (HybridSN smoke test, and the weighted loss equals the
    plain loss when W = 1). Manual: CHOICE.md cites no test numbers.
  - Files: `src/models/hybridsn.py`, `src/train.py`, `configs/dl_v2_*.yaml`,
    `results/dl_v2/CHOICE.md`

### Checkpoint F
- [ ] Day 6 checklist passes; merge

## Phase 6: Headline evaluation (protected) · `day-07-cv`

- [ ] **T19: Deep nested CV and cross-experiment A↔B** (M · deps: T12, T18; experiment B
  downloaded)
  - Acceptance: the final deep model across 6 outer folds (1–2 seeds per fold). The best
    classical and final deep models trained on A and tested on B, and the reverse, each tuned on
    inner-val pots from the training experiment only. Out-of-fold files in `results/cv/` and
    `results/cross_exp/`.
  - Verify: `pytest tests/test_splits.py` (a cross-experiment split never mixes A and B
    between train and test), then real runs complete.
  - Files: `src/train.py`, `configs/cv_*.yaml`, `configs/cross_exp_*.yaml`

- [ ] **T20: Thresholds, lead time, curves, statistics** (M · deps: T3, T19)
  - Acceptance: a per-fold threshold at 95% specificity on inner-val pots; lead time as "k/6
    inoculated pots flagged by DPI 4, m/6 by DPI 5, false-alarm rate Y"; ROC/PR per DPI; a
    reliability plot; pot-bootstrap CIs; Wilcoxon classical vs deep with effect size and the
    n = 6 caveat. All in `results/cv/summary/`.
  - Verify: `pytest tests/test_evaluation.py` (the threshold is computed only from inner-val
    rows; lead time on a synthetic pot series is correct).
  - Files: `src/evaluation.py`, `src/evaluate.py`, `tests/test_evaluation.py`

### Checkpoint G
- [ ] Day 7 checklist passes; CV vs single split vs cross-experiment gaps discussed in
  PROGRESS.md; merge

## Phase 7: Explain and sensors · `day-08-explain`

- [ ] **T21: Gradients, SHAP, combined figure** (S · deps: T18)
  - Acceptance: input-gradient band importance (overall and per DPI); SHAP GradientExplainer
    with a 50-image training background (2D models); one figure that overlays gradients, SHAP
    and the T15 band frequencies.
  - Verify: `pytest tests/test_explain.py` (on a toy linear model, gradient importance matches
    the weights), then manual review of the figure.
  - Files: `src/explain.py`, `tests/test_explain.py`

- [ ] **T22: Simulated sensors** (M · deps: T15, T16)
  - Acceptance: broad Gaussian RGB responses (~80–100 nm FWHM) and a 10–20 nm filter on each of
    the top-5 bands, compared with all 106 bands, each retrained (3 seeds) → DPI 4/5 F1 table in
    `results/sensors/`.
  - Verify: `pytest tests/test_sensors.py` (response weights sum to 1; a flat spectrum maps to
    the same flat value).
  - Files: `src/features/sensors.py`, `configs/sensors_*.yaml`, `tests/test_sensors.py`

- [ ] **T23: Pixel maps (illustrative)** (S · deps: T17) · *on the cut list*
  - Acceptance: the per-pixel 1D-CNN applied to plant pixels for a few DPI 4/8 out-of-fold
    samples, with overlays captioned "illustrative; no pixel labels".
  - Verify: manual review.
  - Files: `src/explain.py`

### Checkpoint H
- [ ] Day 8 checklist passes; merge

## Phase 8: Demo · `day-09-demo`

- [ ] **T24: Predict and export** (S · deps: T18, T20)
  - Acceptance: `python -m src.predict <cube>` → probability, verdict at the T20 threshold and
    pixel map. TorchScript/ONNX export with CPU time per cube logged.
  - Verify: `pytest tests/test_predict.py` (the exported model gives the same output as the
    eager model on a synthetic cube, within 1e-5).
  - Files: `src/predict.py`, `tests/test_predict.py`

- [ ] **T25: Gradio app and model card** (M · deps: T24)
  - Acceptance: `python app/app.py` serves **out-of-fold samples only**: true-colour image,
    heatmap, verdict/probability/true label, and the per-DPI error rate shown next to them. A
    second tab shows band importance + the sensor table. `MODEL_CARD.md` with limits and the
    confound status.
  - Verify: the app launches in a clean venv; spot-check that 5 samples match `oof.csv`.
  - Files: `app/app.py`, `MODEL_CARD.md`

## Phase 9: Report (protected) · `day-10-report`

- [ ] **T26: Figure regeneration and README** (S · deps: T13, T15, T20, T22)
  - Acceptance: `python -m src.figures` rebuilds every report figure from `results/`.
    `README.md` gives fresh-clone reproduction steps.
  - Verify: delete `report/figures/`, rerun, and get identical files.
  - Files: `src/figures.py`, `README.md`

- [ ] **T27: Report, slides, viva** (M · deps: T26)
  - Acceptance: an 8–12 page report following the PLAN.md Day 10 outline; 8–10 slides; 10 viva
    Q&As; references with verified DOIs.
  - Verify: every number in the report has a `results/` path next to it in a traceability
    table.
  - Files: `report/*`

- [ ] **T28: Fresh-eyes audit** (S · deps: T27)
  - Acceptance: a new Claude Code session audits for leakage, unbacked claims, mismatched
    numbers and any decision that touched test pots. Findings are fixed or listed as
    limitations in `report/AUDIT.md`.
  - Verify: run `/agent-skills:review` on the full diff from the first commit; no open blockers.
  - Files: `report/AUDIT.md`

### Checkpoint: Final
- [ ] All SPEC.md success criteria met
- [ ] A fresh clone reproduces the main table
- [ ] Final PROGRESS.md entry; merge `day-10-report` → `main`
