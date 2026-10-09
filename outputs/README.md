# Run the astronomy workflow

From the Astromanthan project folder, run in PowerShell:

```powershell
& '.\ML Model\.venvs\astronomy\Scripts\python.exe' -u .\outputs\pipeline.py
```

This reproduces the audit, grouped split, five-fold model comparison, development
diagnostics, final evaluation, model, predictions and six figures. Default inputs
are the unchanged CSV copies in `selected_datasets`. Override with `--data29`,
`--data24`, and `--output`. `--audit-only` stops before training. A full rerun
overwrites the output files; use `--output outputs/reproduction` to retain a run.

The runner imports the astronomy extension under `ML Model/backend/engine/astronomy`.
It uses the existing engine's preprocessing builder, metrics evaluator, champion
selector, champion store, and UnifiedPredictor. Keep `ML Model` beside `outputs`.
The old employee-turnover and house-price artifacts are not astronomy models.

Predict without retraining:

```powershell
& '.\ML Model\.venvs\astronomy\Scripts\python.exe' .\outputs\predict.py --csv '.\selected_datasets\Dataset 29.csv'
```

The structural champion requires the 30 named raw inputs listed in
`champion_manifest.json`. It masks invalid cells, constructs features and imputes
using the training-fitted pipeline. Output probabilities map **0 = GALAXY,
1 = STAR**. This is a binary classifier; QSO and other unknown types will still
be assigned one of those two classes. Dataset24 lacks structural columns and
must use the separate `five_band_comparison.joblib` diagnostic model instead.

`predictions.csv` contains the held-out test predictions; `new_predictions.csv`
is the default destination of the inference command. Only load joblib files
from a trusted source.

For another machine, create a Python 3.12 environment and install the versions in
`requirements-lock.txt`. The local environment is not part of the source repository.
See `REPORT.md`, `01_audit.md`, `selection.json`, and the CSV/JSON companion
artifacts for methodology, exact counts, assumptions, and experiment results.
