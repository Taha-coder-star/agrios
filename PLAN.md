# Pre-Symptomatic Wheat Stem Rust Detection — Corrected 10-Day Plan

Revision of the original PDF plan (Oct 19, 2026). Project context and hard rules are in
`CLAUDE.md`. The changes from the PDF are tagged **[fix #N]**, where N is the problem number in
the review.

## Reference benchmarks (verify against the papers before quoting)

| Method | Overall F1 | DPI 4 F1 | Source |
|---|---|---|---|
| Traditional ML, tuned preprocessing | 0.86–0.94 | up to 0.83 | Fedotov et al. 2025 |
| SimpleCNN, no weighting | 0.814 | 0.937 (±0.13) | Zhu et al. 2026 |
| Best model, weighted loss | 0.912–0.942 | up to 0.99 | Zhu et al. 2026 |

These look internally odd: the DPI 4 F1 is above the overall F1, and the stds are large. That
suggests small or leaky test sets. Use them for context only. **[fix #7]**

---

## Day 0 — Download (start before Day 1, run in background) [fix #19]
- Check the Drive folder size first. There are two experiments (A, B) of 864 uint16 TIFFs each,
  ~36 MB/image, so ~31 GB per experiment and ~62 GB total. Get experiment A first, then B
  before Day 7.
- Gdown often hits "download quota exceeded". Fallbacks: "Add shortcut to My Drive" + mount in
  Colab, or rclone.
- If disk is tight (Kaggle ~20 GB), stream each file: download → write the pooled cube + mean
  spectra → delete the raw file. The pooled cache is ~3.9 GB.

**Checklist:** all 864 files present (or a streaming cache script working), with sizes logged.

## Day 1 — Setup, inspect, confound check, explore
**Tasks**
1. Repo structure, venv, requirements.txt, empty PROGRESS.md.
2. Inspect files: format, dtype, shape, **wavelength list from the file**, raw counts or
   reflectance.
3. Work out how DPI, label and pot are encoded. Build `data/index.csv`.
4. **Confound check [fix #4].** The paper describes the infected treatment only (NOVEC 7100
   spray, 24 h dark at 100% RH). Controls are undocumented, so treat the confound as a real risk:
   - **Temporal test:** plot the infected-minus-healthy difference (its magnitude, e.g. L2 over
     bands, plus the per-band curve) for DPI 4–9. A disease signal grows over time. A treatment
     artefact stays constant or fades. Large at DPI 4 and then shrinking is a red flag: record
     it.
   - Imaging order and session per pot, if the metadata has it.
   - Per-image mean brightness by class at DPI 4.
   - Draft a short email to the corresponding author (Alexander Fedotov). Ask whether controls
     were mock-inoculated and given the same dark incubation. @taha sends it.
   - Record all of it in DATA_CARD.md. It goes in the report's limitations either way.
   - Confirm the files are calibrated reflectance (values roughly in 0–1 after scaling, white
     reference ≈ 1). The paper says calibration happened in the camera software.
5. `notebooks/01_eda.ipynb`: true-colour images (nearest bands to 660/550/470) at DPI 4/6/8;
   mean ± std spectrum per class per DPI; difference spectrum per DPI.

**Checklist**
- [ ] 864 images, 144/DPI, 72/72 per DPI, exactly 12 pot IDs
- [ ] Mock inoculation and imaging order documented (if unknown, stop and ask, or list as a
      limitation)
- [ ] DPI 4 true-colour healthy vs infected look alike
- [ ] Difference spectrum visibly non-zero by DPI 6
- [ ] PROGRESS.md entry

## Day 2 — Data pipeline and leak-free splits
**Tasks**
1. Loader → float32 (H, W, 106). Apply the Fedotov calibration if the data is raw counts.
2. Background removal, **two variants**: NDVI > 0.3 (Zhu) and 2-cluster K-Means (Fedotov).
   Save 10 previews of each, **including DPI 8–9 infected**, to check pustule tissue isn't
   masked out. Log mask pixel count per image. Choose one on preview quality, not on F1.
   **[fix #13]**
2b. Outlier filter (Fedotov): flag images whose mean spectrum has correlation < 0.85 with the
   training mean. **Apply it to training pots only** (CONSTRAINTS R3) and log how many images it
   removes.
2c. Normalisation: fitted on training pots only. **Never Fedotov's per-day normalisation over
   all images**, which uses test statistics (CONSTRAINTS R2).
3. Caches: mean ± std spectra **masked and unmasked**, and 4× pooled cubes (~103×103×106).
   **[fix #13]**
4. `splits.json`: single 8/2/2 split + **6 outer folds, each with an inner validation pair**
   (1 inoculated + 1 control) taken from the 10 non-test pots. **[fix #1]**
5. Tests (see CONSTRAINTS.md): `tests/test_splits.py` (no pot in two roles, class balance per
   DPI); `tests/test_leakage.py` (normalisation and outlier filter use training indices only);
   `tests/test_data.py` (shapes, no NaN/inf). Pots are keyed by `(experiment, pot_id)`.

**Checklist**
- [ ] pytest passes
- [ ] Masks cover leaves (including diseased leaves), not the pot rim or soil
- [ ] Mask area by class at DPI 4 compared. If it differs a lot, note that it's a possible
      shortcut feature
- [ ] Cached dataset loads in under a minute
- [ ] PROGRESS.md entry

## Day 3 — Classical baseline (inside CV from the start)
**Tasks**
1. Preprocessing variants: raw; Savitzky–Golay; SG + SNV; SG + SNV + 1st derivative.
2. Models: logistic regression, SVM-RBF, Random Forest, XGBoost, PLS-DA.
3. Run in **all 6 outer folds**. Tune each fold with a small grid on its inner validation pair.
   These models run on CPU, so this is cheap. **[fix #1, #20]**
4. Two regimes: train on all DPIs; train on DPI 4–5 only.
5. Save out-of-fold predictions for every image, plus a model × preprocessing × DPI table and
   a per-DPI F1 chart.

**Checklist**
- [ ] Per-DPI F1 (image-level) + pot-level accuracy for every combination
- [ ] No method change was made because of comparing with the paper **[fix #2]**
- [ ] DPI 4 near 1.0 → re-check splits and the confound notes
- [ ] DPI 4 confusion matrix saved
- [ ] PROGRESS.md entry

## Day 4 — Vegetation indices and band selection (inside CV) [fix #5]
**Tasks**
1. Standard indices using the nearest bands: NDVI, GNDVI, PRI, ARI/ARI2, GARI, red-edge, MGRVI.
   **Check Zhu's MGRVI definition**: the standard one is green² vs red². **[fix #14]**
2. Exhaustive 2-band normalised-difference search (5,565 pairs), **run per outer fold** and
   ranked on the inner validation pair at DPI 4–5. Report **out-of-fold** F1 for each fold's
   chosen pair, and how often each band gets selected across folds.
3. Permutation importance + forward selection to k = 1, 2, 3, 5, 10 within each fold.
4. Band-importance plot with 480–550 and 750–870 nm shaded. Out-of-fold DPI 4 F1 vs number of
   bands.
5. Top-5 bands = the bands selected most often across folds → `results/bands/top5.json`.

**Checklist**
- [ ] F1 vs number of bands curve is out-of-fold
- [ ] Band selection stability across folds reported
- [ ] Top bands compared with 480–550 nm
- [ ] PROGRESS.md entry

## Day 5 — Deep learning I: harness, SimpleCNN, 1D-CNN
**Tasks**
1. Dataset over pooled cubes; per-band MinMax fitted on training pots only.
2. Training loop from YAML: Adam, lr 1e-4, wd 1e-5, up to 50 epochs, early stopping on
   inner-val loss (patience 10), CSV log per epoch. **Use GroupNorm, or batch ≥ 16 on pooled
   cubes; avoid BatchNorm with batch 4. [fix #17]** Turn on deterministic algorithms.
   **[fix #18]**
3. SimpleCNN (pooled cubes); 1D-CNN (mean spectra + a variant on sampled plant pixels).
4. Screening on the single split with 3 seeds. Report per-DPI F1 as mean ± seed-std, labelled
   "seed variance only". **[fix #15]**
5. Shuffled-label sanity run → F1 ≈ 0.5.

**Checklist**
- [ ] Curves sane; per-DPI table; shuffled labels ≈ 0.5
- [ ] Compared with the Day 3 out-of-fold baseline at DPI 4
- [ ] PROGRESS.md entry

## Day 6 — Deep learning II: weighted loss, ablations, choose the model
**Tasks**
1. HybridSN (and SpectralAttentionNet if there's time, since it's on the cut list).
2. **Weighted loss on DPI 4–5** (the project goal), W ∈ {1.0, 1.2, 1.5}, chosen on inner-val
   DPI 4 F1. **[fix #6]**
3. Ablations on the best model, 3 seeds: unmasked vs NDVI mask vs mask + 1st derivative.
4. Choose the final model **on validation results only**, with a written reason: DPI 4 val F1,
   seed std, model size. **[fix #2]**

**Checklist**
- [ ] Master table (model × setting × DPI) on validation pots
- [ ] Final model chosen without test-pot results
- [ ] Any per-DPI drop caused by weighting noted
- [ ] PROGRESS.md entry

## Day 7 — Cross-validated headline results, cross-experiment test, lead time
Protected day: never cut.

**Tasks**
0. **Cross-experiment test (second headline):** train on experiment A (tune on inner-val pots
   from A), test once on all of experiment B. Then the reverse. Do this for the best classical
   model and the final deep model. This is a stronger test than a pot split, because it checks
   generalisation to a separate run of the experiment.
1. Run the final deep model through the 6 outer folds. Each fold tunes or early-stops on its own
   inner validation pair, with 1–2 seeds per fold. **[fix #1, #20]** The classical out-of-fold
   results come from Day 3.
2. **Pot-level cluster bootstrap** 95% CIs for DPI 4 and DPI 5 F1. **[fix #3]**
3. Alert threshold: in each fold, choose the threshold that gives 95% specificity on that fold's
   inner validation pots, then apply it to that fold's test pots. **[fix #10]**
4. Lead time: "infection flagged by DPI 4 in k/6 inoculated pots, by DPI 5 in m/6, at
   false-alarm rate Y on healthy pots". The data starts at DPI 4, so detection earlier than that
   can't be shown. **[fix #9]**
5. ROC/PR per DPI; reliability plot. Wilcoxon classical vs deep across the 6 folds, reported
   with the effect size and the note "n = 6, minimum p = 0.031, indicative only". **[fix #16]**

**Checklist**
- [ ] Cross-experiment A→B and B→A per-DPI F1 reported next to the CV result
- [ ] CV mean + pot-bootstrap CI reported; gap vs single split discussed
- [ ] Lead time stated as k/6 pots with the false-alarm rate
- [ ] Classical vs deep difference stated with its caveat
- [ ] PROGRESS.md entry

## Day 8 — Interpretability and the sensor experiment
**Tasks**
1. Input-gradient band importance for the final model, overall and per DPI.
2. SHAP (GradientExplainer, 50 training images as background) for 2D models.
3. **Simulated sensors [fix #8]**, each retrained in CV (or on the single split with 3 seeds if
   time is short):
   a. Phone RGB: integrate bands with broad Gaussian responses (~470/540/610 nm centres,
      ~80–100 nm FWHM, overlapping), or published camera curves.
   b. Multispectral: the top-5 bands from Day 4, each integrated with a 10–20 nm FWHM filter.
   c. All 106 bands.
   Report DPI 4/5 F1 for each.
4. Per-pixel maps from the **per-pixel-trained** 1D-CNN, captioned "illustrative; no pixel
   labels". **[fix #11]**
5. Combined figure: gradients, SHAP and Day 4 band frequency on one wavelength axis.

**Checklist**
- [ ] Simulated-RGB DPI 4 F1 recorded
- [ ] Multispectral (top-5) DPI 4 F1 recorded
- [ ] Gradient/SHAP/index agreement written down
- [ ] PROGRESS.md entry

## Day 9 — Inference package and demo
**Tasks**
1. `src/predict.py`: cube path → probability, verdict at the Day 7 threshold, pixel map.
2. Export to TorchScript/ONNX and measure CPU time per cube.
3. Gradio app: samples come **only from out-of-fold predictions**. Show true-colour, heatmap,
   verdict, probability and true label, plus that DPI's out-of-fold error rate. **[fix #12]**
   Second tab: band importance + sensor table.
4. MODEL_CARD.md: data, intended use, CV performance, limits (one cultivar, darkroom, 12 pots,
   confound notes).
5. Optional: a short note on fusion with a weather-based risk layer.

**Checklist**
- [ ] App runs from a clean env with one command
- [ ] App predictions match the out-of-fold outputs
- [ ] Error rate is visible next to the showcase samples
- [ ] PROGRESS.md entry

## Day 10 — Report, figures, audit
**Tasks**
1. One script regenerates every figure from `results/`.
2. Report (8–12 pages): intro; related work; data + **pot-level nested-CV protocol**; methods;
   results (CV headline, per-DPI, lead time as k/6, false alarms); interpretability + sensor
   comparison; limitations (one cultivar, darkroom, 12 pots, confound status, no DPI 0–3 data);
   future work.
3. README with reproduction steps from a fresh clone.
4. Fresh-eyes audit in a new Claude Code session: leakage, unbacked claims, report/results
   mismatches, and any decision that touched test pots.
5. 8–10 slides; 10 viva Q&As. **References with DOIs/links, verified. [fix #22]**

**Checklist**
- [ ] Every number traces to `results/`
- [ ] Fresh clone reproduces the main table
- [ ] Audit findings fixed or listed as limitations
- [ ] Final PROGRESS.md entry

---

## Risks and fallbacks

| Risk | Fallback |
|---|---|
| Drive quota / disk too small | Day 0 streaming cache; delete raw files after caching |
| Pot IDs not recoverable | Stop; check the Fedotov data description or contact the authors |
| Controls not mock-inoculated (undocumented in the paper) | Day 1 temporal test + email to the authors; state it as the lead limitation; a large DPI 4 signal that shrinks later means "treatment detection" can't be ruled out |
| Raw counts, not reflectance | Fedotov calibration; record it in DATA_CARD.md |
| Results swing between folds | Expected with 6 infected pots; report pot-bootstrap CIs |
| GPU time runs out | Classical + 1D-CNN (CPU) become the main CV result |

## Schedule policy
The 10-day total is fixed, and Days 6–8 can flex. If GPU time runs short, work down the cut
list. Day 7 and Day 10 are protected. If deep learning stalls entirely, the classical models
(CPU) are a complete result on their own, and Fedotov's paper is built entirely on them.

## Cut list (drop in this order) [fix #21]
1. Day 9 weather note
2. SpectralAttentionNet
3. HybridSN (keep SimpleCNN + 1D-CNN)
4. Gradio second tab
5. Per-pixel maps
6. Calibration plot

**Never cut:** pot-level split, nested CV with inner validation, the confound check, per-DPI
metrics, pot-level CIs, the simulated-sensor experiment, band importance.

## References (add DOIs/links)
- Fedotov et al. 2025: hyperspectral processing and ML for early wheat stem rust detection
- Zhu et al. 2026: pre-symptomatic wheat stem rust detection, HSI + deep learning
- Terentev et al. 2025: cross-crop transferability, wheat and barley
- Zhang et al. 2024: HSI early detection of tomato bacterial leaf spot
- PSNet 2026: low-cost hyperspectral + RGB fusion
- Dataset: Google Drive folder (URL TODO)
