"""Regenerate the concise UTF-8 report from recorded results; never retrain."""
from pathlib import Path
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ML Model'))
from backend.engine.astronomy.workflow import write_report


def main():
    out=ROOT/'outputs'
    def read(name): return json.loads((out/name).read_text(encoding='utf-8'))
    transfer=read('transfer_metrics.json')
    write_report(out,read('partition_summary.json'),read('identity_linkage.json'),
        pd.read_csv(out/'cv_summary.csv').to_dict('records'),read('selection.json')['winner'],
        read('test_metrics.json'),pd.read_csv(out/'robustness.csv').to_dict('records'),
        pd.read_csv(out/'importance_test_descriptive_only.csv'),transfer['dataset24_nonoverlap'],
        transfer['dataset29_holdout_shared_bands'],pd.read_csv(out/'cleaning_counts.csv',index_col=0),
        pd.read_csv(out/'outlier_counts.csv').to_dict('records'))
    p=out/'REPORT.md'; s=p.read_text(encoding='utf-8')
    s=s.replace('plus full IDs within Dataset24','plus conservative repeated-ID grouping within Dataset24 (IDs are quantized)')
    s=s.replace('Efficient predefined settings','Boosting variants are nearly tied; their tiny CV differences do not establish statistical superiority. Efficient predefined settings')
    s=s.replace('No tuning follows this result.','Dataset24 is substantially fainter: median g=21.12 versus 17.42 in development (all binary Dataset24 rows; domain_shift.csv). No tuning follows this result.')
    p.write_text(s,encoding='utf-8')
    print(f'Report regenerated: {len(s.split())} words')


if __name__=='__main__': main()
