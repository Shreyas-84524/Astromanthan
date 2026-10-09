"""Export the exact retained samples with deterministic measurement cleaning."""
from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ML Model'))
from backend.engine.astronomy.workflow import normalize, identity_groups
from backend.engine.astronomy.features import RAW, MAG, clean_numeric


def main():
    out = ROOT / 'outputs'
    a = normalize(pd.read_csv(ROOT / 'selected_datasets' / 'Dataset 29.csv', dtype=str), '29')
    b = normalize(pd.read_csv(ROOT / 'selected_datasets' / 'Dataset 24.csv', dtype=str), '24')
    combined, _ = identity_groups(a, b)
    binary = combined.Object_Class.isin(['STAR', 'GALAXY'])
    counts = combined.loc[binary].groupby('object_group').Object_Class.nunique()
    allowed = binary & combined.identifiable & ~combined.object_group.isin(counts[counts > 1].index)
    primary = combined.loc[allowed & (combined.source == '29')].copy()
    primary['_complete'] = clean_numeric(primary[RAW]).notna().sum(axis=1)
    primary = primary.sort_values(['_complete', 'source_row'], ascending=[False, True]).drop_duplicates('object_group').sort_values('source_row').reset_index(drop=True)
    all_primary_groups = set(combined.loc[combined.source == '29', 'object_group'])
    secondary = combined.loc[allowed & (combined.source == '24') & ~combined.object_group.isin(all_primary_groups)].drop_duplicates('object_group').reset_index(drop=True)
    splits = pd.read_csv(out / 'split_assignments.csv')
    np.testing.assert_array_equal(primary.source_row, splits.source_row)
    np.testing.assert_array_equal(primary.object_group, splits.object_group)
    assert not set(primary.object_group) & set(secondary.object_group)
    expected = json.loads((out / 'partition_summary.json').read_text(encoding='utf-8'))
    assert len(primary) == expected['primary_rows']
    assert len(secondary) == expected['secondary_rows']
    for name, frame, features in [('Dataset_29_cleaned.csv', primary, RAW), ('Dataset_24_cleaned.csv', secondary, MAG)]:
        cleaned = clean_numeric(frame[features])
        cleaned['Object_Class'] = frame.Object_Class
        path = out / name
        cleaned.to_csv(path, index=False, na_rep='')
        reread = pd.read_csv(path)
        assert reread.shape == cleaned.shape
        assert reread.Object_Class.isin(['STAR', 'GALAXY']).all()
        assert not np.isinf(reread[features].to_numpy()).any()
        print(f'{name}: {len(cleaned):,} rows, {len(features)} measurements + Object_Class; {int(cleaned.isna().sum().sum()):,} blank cells')
    (out / 'CLEANED_DATASETS.md').write_text('''# Cleaned dataset exports

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
''', encoding='utf-8')


if __name__ == '__main__':
    main()
