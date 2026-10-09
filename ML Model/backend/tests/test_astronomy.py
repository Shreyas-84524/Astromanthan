"""Focused checks for leakage boundaries and recoverable inference."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
import pandas as pd
from backend.engine.astronomy.features import AstronomyFeatures, AnomalyFlags, clean_numeric, RAW, MAG
from backend.engine.astronomy.workflow import identity_groups, make_pipeline


def sample(n=50):
    rng=np.random.default_rng(42)
    d=pd.DataFrame({c:rng.normal(18,1,n) for c in RAW})
    for c in d:
        if 'Ratio' in c: d[c]=rng.uniform(.2,1,n)
        elif 'Radius' in c: d[c]=rng.uniform(.5,6,n)
    return d


def test_cleaning_preserves_negative_flux_and_valid_magnitudes():
    d=pd.DataFrame({'U_Band_Magnitude':['-1','1e12','garbage'],
        'Petrosian_Radius_U':['-1','2','0'],'Petrosian_Flux_U':['-1','99','1e12']})
    c=clean_numeric(d)
    assert c.U_Band_Magnitude.iloc[0]==-1
    assert c.U_Band_Magnitude.iloc[1:].isna().all()
    assert c.Petrosian_Radius_U.isna().tolist()==[True,False,True]
    assert c.Petrosian_Flux_U.iloc[:2].tolist()==[-1,99]


def test_no_spectroscopy_or_identifiers_in_main_features():
    d=sample(); d['Redshift']=np.arange(len(d)); d['Object_ID']='cheat'; d['Object_Class']='STAR'
    f=AstronomyFeatures().fit_transform(d)
    assert not set(['Redshift','Object_ID','Object_Class']) & set(f)
    np.testing.assert_allclose(f.colour_U_G,d.U_Band_Magnitude-d.G_Band_Magnitude)
    assert f.shape[1]==60


def test_fold_fitted_imputation_and_anomalies_do_not_change_on_transform(tmp_path):
    x=sample(); x.loc[0,'U_Band_Magnitude']=np.nan
    y=np.arange(len(x))%2
    p=make_pipeline('Logistic',flags=True).fit(x,y)
    med=p.named_steps['preprocessor'].named_transformers_['numeric'].named_steps['imputer'].statistics_.copy()
    center=p.named_steps['anomalies'].scaler_.center_.copy()
    extreme=sample(3); extreme[MAG]=1e12
    pred=p.predict_proba(extreme)
    assert np.isfinite(pred).all()
    np.testing.assert_allclose(med,p.named_steps['preprocessor'].named_transformers_['numeric'].named_steps['imputer'].statistics_)
    np.testing.assert_allclose(center,p.named_steps['anomalies'].scaler_.center_)
    artifact=tmp_path/'model.joblib'; joblib.dump(p,artifact)
    np.testing.assert_allclose(pred,joblib.load(artifact).predict_proba(extreme))


def test_identity_matching_transitive_and_not_rounded_id():
    a=sample(3); b=sample(2)
    for d in [a,b]:
        d['Object_ID']='1.24e+18'; d['Right_Ascension_deg']=[10.,20.,30.][:len(d)]; d['Declination_deg']=0.
    b.loc[0,MAG]=a.loc[0,MAG]
    b.loc[0,'Right_Ascension_deg']=10.0001
    b.loc[1,'Object_ID']='different'; b.loc[1,'Right_Ascension_deg']=90
    b.loc[1,MAG]=a.loc[2,MAG]
    grouped,info=identity_groups(a,b)
    assert grouped.object_group.iloc[0]==grouped.object_group.iloc[3]
    assert grouped.object_group.iloc[2]==grouped.object_group.iloc[4]
    assert grouped.object_group.iloc[0]!=grouped.object_group.iloc[1]
    assert info['shared_object_groups']==2
