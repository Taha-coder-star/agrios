# Progress

## 2026-10-19 · Day 1 (part 1): setup without data · `day-01-setup`

**Done**
- T1: `.venv` on Python 3.12 (3.11 isn't installed; Colab is on 3.12 too). Pinned
  `requirements.txt` / `requirements-dev.txt`, `pyproject.toml` (ruff + `requires_data`
  marker), package skeleton, synthetic fixtures.
- T2: `src/data/splits.py`: a single 8/2/2 split + 6 outer folds with inner validation pairs,
  keyed `experiment:pot_id`. Every pot is test once and val once. CLI `python -m src.data.splits`.
- T3: `src/evaluation.py`: per-DPI F1, pot-level accuracy, a pot cluster bootstrap
  (label-stratified), and the results writer.
- T4: `src/features/preprocess.py`: `BandScaler.fit(X, train_idx)` and a train-only
  `OutlierFilter` (corr < 0.85) that reports `n_removed`.

**Key numbers**
- `pytest -q`: 19 passed, 1 skipped (needs real data), 4.6 s. Coverage 99%. Ruff clean.

**Problems**
- The first bootstrap took 42 s per call because of sklearn `f1_score` overhead × 2,000 draws.
  Replaced it with a NumPy F1 (tested against sklearn), and the test now takes 0.4 s.

**Next**
- @taha downloads experiment A into `data/raw/` (T5).
- Then T6: inspect the TIFFs and build `data/index.csv`. **Stop if pot IDs aren't
  recoverable.**
- Then T7: confound check (difference magnitude vs DPI), EDA, author email draft.
