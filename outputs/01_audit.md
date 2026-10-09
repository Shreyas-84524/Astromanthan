# Data audit

Two files verified by SHA-256 (see audit JSON). Dataset 29: 100,050 rows / 46 columns. Dataset 24: 99,999 / 12.

These are partially overlapping survey samples with different feature coverage, not a ready train/test pair. Keep Dataset29 for the structural model; use nonoverlapping Dataset24 only for a separately trained five-band transfer check. No row concatenation or morphology join. Dataset29 IDs use rounded scientific notation, so cannot be reliable join keys.

Identity rule: link within 1 arcsecond, or exact agreement (5 decimal places) on any three valid ugriz bands, transitively. Full Dataset24 IDs also link records within that file. This is conservative and can merge close sources. Exclude rows with neither valid position nor three valid bands. Ambiguous mixed-label groups are quarantined. It prevents detected duplicates crossing partitions, but unrecognized identities remain a limitation.

Linkage: {"position_pairs": 1912, "photometric_links": 18097, "shared_object_groups": 1819, "dataset24_overlapping_rows": 2734, "unidentifiable_rows": 51, "total_groups": 176021}

Retained partitions: {"primary_rows": 89251, "development_rows": 71401, "test_rows": 17850, "secondary_rows": 62128, "conflicting_groups": 1909, "duplicate_primary_rows_removed": 197}

Labels: {'GALAXY': 52361, 'STAR': 37250, 'QSO': 10439} (29); {'GALAXY': 59445, 'STAR': 21593, 'QSO': 18961} (24). QSO is excluded: the task is binary, not QSO detection.

All fields loaded as strings to protect IDs and expose contamination. Full per-column ranges, missingness, unique counts, dominant fractions, and sentinel occurrences are in 02_feature_decisions.csv. Original dtypes/head/memory are in audit JSON.

Fixed cleaning assumptions: magnitudes [-5,40], angular radii (0,1000], axis ratios (0,1], absolute flux <1e7; nonnumeric/nonfinite values become missing. These deliberately broad bounds reject corruption such as +/-1e12. Units follow SDSS-like names, without an organizer dictionary. Do not blanket-replace -1/99; their meaning is column-dependent. Negative flux is allowed. Bad cells are masked, not entire scientifically unusual rows.

IDs/bookkeeping/position/notes are excluded from predictors. Redshift is diagnostic-only; spectral type and galaxy population have unverified derivation and are excluded. Median imputation and scaling are fitted within training folds. Final test is sealed until selection and development diagnostics finish.
