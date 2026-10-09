"""Select model-relevant raw columns without cleaning or filtering any values."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ML Model'))
from backend.engine.astronomy.features import RAW, MAG


def main():
    for number in [29, 24]:
        source = ROOT / 'selected_datasets' / f'Dataset {number}.csv'
        original = pd.read_csv(source, dtype=str, keep_default_na=False, na_filter=False)
        if number == 29:
            structured = original[RAW + ['Object_Class']].copy()
        else:
            names = {f'{band}_Magnitude': f'{band}_Band_Magnitude' for band in 'UGRIZ'}
            names['Class'] = 'Object_Class'
            structured = original[[f'{band}_Magnitude' for band in 'UGRIZ'] + ['Class']].rename(columns=names)
        target = ROOT / 'outputs' / f'Dataset_{number}_raw_structured.csv'
        structured.to_csv(target, index=False)
        saved = pd.read_csv(target, dtype=str, keep_default_na=False, na_filter=False)
        pd.testing.assert_frame_equal(saved, structured)
        assert len(saved) == len(original)
        print(f'{target.name}: {len(saved):,} rows, {len(saved.columns)} columns; all selected raw cell values preserved')


if __name__ == '__main__':
    main()
