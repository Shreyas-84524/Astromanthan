"""Check saved inference, metrics and identity disjointness without retraining."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ML Model'))
from backend.engine.inference.unified_predictor import UnifiedPredictor
from backend.engine.astronomy.features import MAG,RAW,clean_numeric


def main():
    out=ROOT/'outputs'
    split=pd.read_csv(out/'split_assignments.csv')
    train=set(split.loc[split.partition=='development','object_group'])
    test=set(split.loc[split.partition=='test','object_group'])
    assert not train&test
    assert split.object_group.is_unique
    p=pd.read_csv(out/'predictions.csv')
    truth=(p.Object_Class=='STAR').astype(int); pred=(p.predicted_class=='STAR').astype(int)
    scores=json.loads((out/'test_metrics.json').read_text())
    assert abs(accuracy_score(truth,pred)-scores['accuracy'])<.000051
    assert abs(f1_score(truth,pred,average='macro')-scores['f1_macro'])<.000051
    assert set(p.object_group)==test
    d=pd.read_csv(ROOT/'selected_datasets'/'Dataset 29.csv',dtype=str)
    sample=d.iloc[p.source_row.iloc[:20]].copy()
    loaded=UnifiedPredictor.load(out)
    prediction=loaded.predict_proba(sample)[:,1]
    np.testing.assert_allclose(prediction,p.probability_STAR.iloc[:20],rtol=1e-12,atol=1e-12)
    sample.to_csv(out/'inference_example.csv',index=False)
    # Postfreeze distribution diagnosis; no threshold/model changes follow it.
    dev=d.iloc[split.loc[split.partition=='development','source_row']]
    second=pd.read_csv(ROOT/'selected_datasets'/'Dataset 24.csv',dtype=str).rename(columns={f'{b}_Magnitude':f'{b}_Band_Magnitude' for b in 'UGRIZ'})
    second=second.loc[second.Class.isin(['STAR','GALAXY'])]
    a=clean_numeric(dev[MAG]); b=clean_numeric(second[MAG]); rows=[]
    for c in MAG:
        lo,hi=a[c].quantile([.01,.99]); rows.append(dict(feature=c,development_median=a[c].median(),
            dataset24_binary_median=b[c].median(),development_p01=lo,development_p99=hi,
            dataset24_outside_development_1_99_fraction=float(((b[c]<lo)|(b[c]>hi)).mean())))
    pd.DataFrame(rows).to_csv(out/'domain_shift.csv',index=False)
    result=dict(group_overlap=0,unique_primary_groups=True,prediction_rows=len(p),
        misclassified=int((truth!=pred).sum()),metrics_recomputed=True,
        fresh_process_reload_matches=True,model_size_bytes=(out/'model.joblib').stat().st_size,
        figure_count=len(list((out/'figures').glob('*.png'))))
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
