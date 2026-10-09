# Cleaned dataset exports

- Dataset_29_cleaned.csv: 89,251 retained sources, 30 numeric measurements and Object_Class.
- Dataset_24_cleaned.csv: 62,128 retained sources, five ugriz magnitudes and Object_Class.

These exports reproduce the existing pipeline's sample selection and fixed
measurement-validity checks. Only STAR/GALAXY remain. Conservative duplicate
groups, conflicting-label groups and unidentifiable rows are handled exactly as
documented in 01_audit.md. Dataset24 excludes all groups linked to Dataset29.
Identifiers, coordinates, redshift, derived spectral/population attributes and
arbitrary notes are excluded. Both files use normalized Dataset29 column names.

Blank cells mean missing, nonnumeric or rejected measurements. No imputer,
scaler, anomaly detector or model was fitted during export. Fit imputation on
training folds only. Engineered features remain part of the saved model pipeline.

Dataset29 contains BOTH development and held-out test rows. Its row order exactly
matches split_assignments.csv; use that file's partition column to preserve the
existing split. Do not train on the whole export and then report the old holdout
as unseen. Metadata is kept separate so it cannot accidentally become a predictor.
Dataset24 has no structural measurements and is for the five-band comparison.
