# Pre-Symptomatic Wheat Stem Rust Detection

Goal: detect wheat stem rust 4–5 days after inoculation (DPI 4–5), before visible symptoms
(DPI 6–7), from hyperspectral images. Then explain which wavelengths make that possible and how
few bands a cheap sensor would need.

Read CONSTRAINTS.md before writing code. Do not weaken it to make a change pass.

## Workflow (addyosmani/agent-skills)
`/agent-skills:spec` → `/agent-skills:plan` → `/agent-skills:build` → `/agent-skills:test` →
`/agent-skills:review` → `/agent-skills:ship`. Each phase waits for human approval.
- `SPEC.md` is the source of truth: objective, commands, boundaries, success criteria.
  Update it first whenever a decision changes.
- `PLAN.md` is the day-by-day research plan. `/plan` turns it into `tasks/plan.md` +
  `tasks/todo.md`.
- Build one task at a time, test-first, one commit per task.
- Each evening, run that day's checklist and add a `PROGRESS.md` entry (done, key numbers,
  problems, next step). Don't start a day until the previous day's checklist passes.
- Git: `main` is the trunk. Use one branch per day (`day-01-setup`, `day-02-pipeline`, …),
  merged into `main` after that day's checklist passes.

## Dataset facts (verify on Day 0/1, correct here if wrong)

| Item | Value |
|---|---|
| Source | Fedotov et al. 2025, public Google Drive folder (URL: TODO) |
| Repeats | **Two independent experiments (A and B), 864 images each.** Fedotov trained on one and tested on the other |
| Format | Multi-channel TIFF, 106 channels, uint16, ~36 MB/image → ~31 GB per experiment, ~62 GB total. Read with `tifffile` |
| Calibration | Dark/white calibration was done in the camera software, so files are likely reflectance already. Verify on Day 1 |
| Inoculation | Infected: airbrushed spore suspension in NOVEC 7100, 24 h dark at 100% RH and 23 °C, then 25 °C. **Controls: not described. Mock inoculation unknown, treated as a confound risk** |
| Crop | Spring wheat, Saratovskaya 74 (highly susceptible) |
| Pots | 12: 6 inoculated, 6 healthy controls |
| Camera | Cubert Ultris 20 snapshot, indoor darkroom |
| Spectral range | 450–874 nm, 106 bands (~4.04 nm step; always use the wavelength list from the files) |
| Image size | 410 × 410 × 106 |
| Time points | DPI 4–9, 144 images/day (72 healthy, 72 infected) = 12 images per pot per day |
| Total | 864 images per experiment (1,728 total) |
| Stages | DPI 4–5 pre-symptomatic · DPI 6–7 slight yellowing · DPI 8–9 pustules |

**Independent units are pots, not images.** There are 12 pots, and only 6 infected pots per DPI.

## Hard rules (never break these)

1. **Split by pot, never by image.** Default single split: 8 train / 2 val / 2 test pots, each
   balanced inoculated/control.
2. **Cross-validation is the headline result, not the single split.** Use 6 outer folds (test =
   1 inoculated + 1 control pot). Each fold has its own inner validation pair taken from the
   remaining 10 pots. Early stopping, hyperparameters, thresholds and band selection all use the
   inner validation pair only.
3. **Test pots are used once.** No decision may depend on test-pot results. That covers models,
   preprocessing, bands, loss weights and thresholds. Comparing against published numbers is not
   a reason to change the method after seeing test results.
4. **Report every metric per DPI.** DPI 4 is the headline. Report pot-level accuracy next to
   image-level F1.
5. **Uncertainty comes from pots.** Confidence intervals use a pot-level cluster bootstrap,
   never an image-level one. Seed std measures initialisation noise only, so label it that way.
6. **Seeds.** Final deep models get 5 seeds (42, 123, 456, 789, 1024). Ablations and screening
   get 3 seeds. Deep CV gets 1–2 seeds per fold. Set `torch.use_deterministic_algorithms(True)`
   where possible.
7. **Traceability.** Every experiment writes its config and results to `results/<experiment>/`
   as JSON + CSV. No number goes in the report unless it traces to a file.
8. **Stop and ask** if pot IDs can't be recovered. The whole evaluation depends on them.
   Mock inoculation is undocumented, so proceed, but run the Day 1 temporal check and report
   the confound as a limitation.
9. **No test-set statistics anywhere.** Normalisation is fitted on training pots only, and never
   per day over all images. The outlier filter (corr < 0.85) runs on training data only, and
   the number of images removed is logged.
10. **Cross-experiment test.** Train on experiment A, test on experiment B (and the reverse).
    This is the second headline evaluation next to the leave-pot-out CV.
11. **Published benchmarks are reference points, not targets.** An honest pot-level result
   below them is not a failure.

## Repo structure

```
CLAUDE.md  PLAN.md  PROGRESS.md  requirements.txt
configs/            # one YAML per experiment
data/raw/           # downloaded cubes (not in git)
data/cache/         # mean spectra (masked + unmasked), pooled cubes
data/index.csv      # path, experiment (A/B), dpi, label, pot_id, acq_order (if known)
data/splits.json    # single split + 6 outer folds with inner val pairs
data/DATA_CARD.md
src/data/  src/features/  src/models/
src/train.py  src/evaluate.py  src/explain.py  src/predict.py
notebooks/          # EDA and figures only
results/  app/  tests/  report/
```

Stack: Python 3.12, NumPy, SciPy, scikit-learn, XGBoost, PyTorch, tifffile, matplotlib, SHAP,
Gradio. Dev: pytest, pytest-cov, ruff. Kaggle/Colab GPU if there's no local GPU.
