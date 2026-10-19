# Constraints

Last reviewed: 2026-10-19 by @taha

Interview answers: enforce the project's research-integrity rules plus test coverage and
security; **block** on the floor and the integrity rules, **warn** on everything else until Day
5; measure and hold (no invented numbers); ≤ 90 s for task-end checks.

Status: no code exists yet. Every tool below is installed by the Day 1 venv
(`requirements-dev.txt`). Until then, only the floor applies, through review of `git diff`.

## Floor (always enforced, no setup required)

- No new suppression comments: `# noqa`, `# type: ignore`, `# pragma: no cover`,
  `pytest.mark.skip`/`xfail` without a reason in the commit message
- No unimplemented stubs: `raise NotImplementedError`, `pass` in place of logic, a bare
  `except: pass`
- No skipped or deleted tests without a reason in the commit message
- No secrets in source. No raw data or large caches committed (`.gitignore` covers `data/raw/`,
  `data/cache/`)
- This file is not weakened to make a change pass. Tightening is silent; loosening needs
  @taha's approval in its own commit

## Research integrity (block, because these make or break the result)

| ID | Rule | Checked by | Runs at |
|----|------|-----------|---------|
| R1 | Split by pot. No pot appears in two roles (train/val/test) within any fold or the single split | `pytest tests/test_splits.py` | task end, CI |
| R2 | Normalisation and scaling statistics are fitted on training-pot indices only. **No per-day normalisation over all images** (Fedotov's best-score trick uses test statistics and can't score a single new image) | `pytest tests/test_leakage.py` | task end, CI |
| R3 | The outlier filter (correlation < 0.85 with the mean spectrum) runs on **training data only**. Its threshold is fixed before any test result, and the number of images removed is logged | `pytest tests/test_leakage.py` + `results/*/config.json` field `outliers_removed` | task end, CI |
| R4 | Test pots and the second experiment are used once. No hyperparameter, band, threshold or model choice reads them | review: every `configs/*.yaml` selection field names a val/inner-val source; `/agent-skills:review` checks the diff | review |
| R5 | Every metric is reported per DPI, and DPI 4 is the headline | `pytest tests/test_evaluation.py` (per-DPI output schema) | task end, CI |
| R6 | Confidence intervals resample **pots** (cluster bootstrap), never images | `pytest tests/test_evaluation.py` (synthetic known-answer case) | task end, CI |
| R7 | Final deep results use 5 seeds; ablations 3; deep CV 1–2 per fold. `mean ± std` outside CV is labelled "seed variance" | review of `results/*/config.json` `seeds` field | review |
| R8 | Every number in `report/` traces to a file in `results/` | `python -m src.figures` regenerates every figure; Day 10 fresh-session audit | Day 10, review |

## Enforced with numbers

| Dimension | Rule | Checked by | Runs at |
|-----------|------|-----------|---------|
| Lint | Zero errors (ruff defaults + `I`, `B`) | `ruff check src tests` | every edit |
| Tests | All pass | `pytest -q` | task end, CI |
| Coverage: integrity code | Changed lines in `src/data/` and `src/evaluation.py` ≥ 80% covered | `pytest --cov=src --cov-report=lcov` + `git diff` | task end (warn until Day 5, then block) |
| Secrets | No findings | `gitleaks detect --redact --no-banner` | CI (external check) |
| Dependencies | Nothing at high or above | `osv-scanner scan source -r .` | CI (external check) |

80% on changed lines is high enough to force a test, and low enough to allow a config line.
Notebooks, `app/` and model architectures are exempt from coverage. Their check is a smoke test
(one forward/backward pass per model).

## Measured, not yet enforced

| Metric | Today | Direction |
|--------|-------|-----------|
| Project coverage | n/a (no code) | record on Day 2, must not fall |
| `pytest -q` wall time | n/a | record on Day 2, keep < 90 s |

## Exceptions

| ID | Rule | Path | Reason | Owner | Expires |
|----|------|------|--------|-------|---------|
| — | | | | | |
