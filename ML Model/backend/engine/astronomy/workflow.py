"""Reproducible, bounded Deep Sky Divide experiment using the existing engine.

Training selection uses development CV only. Test is opened once, after all
choices and development robustness experiments are persisted in selection.json.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
    confusion_matrix, f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from backend.engine.astronomy.features import (AstronomyFeatures, AnomalyFlags,
    clean_numeric, BANDS, MAG, PSF, RADIUS, R50, FLUX, AXIS, RAW)
from backend.engine.preprocessing.pipeline_builder import PreprocessingPipelineBuilder
from backend.engine.evaluation.evaluator import MetricsEvaluator
from backend.engine.evaluation.champion_selector import ChampionSelector
from backend.engine.artifacts.champion_store import ChampionStore
from backend.engine.inference.unified_predictor import UnifiedPredictor
from backend.engine.contracts.schemas import ChampionCandidate, FinalTestEvaluationResult, ProblemType

ROOT = Path(__file__).resolve().parents[4]
START = time.monotonic()
SEED = 42


def log(message):
    print(f'[{(time.monotonic()-START)/60:.1f} min] {message}', flush=True)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def category(c):
    if c in ['Class', 'Object_Class']:
        return 'TARGET', 'Target only'
    if c in ['Redshift','Spectral_Type','Galaxy_Population']:
        return 'SPECTROSCOPIC/LABEL-DERIVED', 'Exclude: unavailable/derived at photometric inference; redshift diagnostic only'
    if c in ['Right_Ascension','Declination','Right_Ascension_deg','Declination_deg']:
        return 'POSITION', 'Exclude predictor; use for duplicate grouping and spatial robustness'
    if c in RAW:
        return ('STRUCTURAL' if c in RADIUS+R50+AXIS else 'PHOTOMETRIC'), 'Keep after fixed validity checks'
    if c in [f'{b}_Magnitude' for b in BANDS]:
        return 'PHOTOMETRIC', 'Keep in five-band comparison'
    if c in ['ID','Object_ID','Spectroscopic_Object_ID','Imaging_Run_Number','Image_Rerun_Number',
             'Camera_Column','Field_Number','Spectroscopic_Plate_Number','Modified_Julian_Date','Fiber_ID']:
        return 'IDENTIFIER', 'Exclude: identifier or observation bookkeeping; audit identity precision'
    return 'OTHER', 'Exclude: no documented astronomical measurement meaning'


def audit(d, name, path, out):
    target = 'Object_Class' if 'Object_Class' in d else 'Class'
    records=[]
    for c in d:
        v=pd.to_numeric(d[c],errors='coerce')
        counts=d[c].value_counts(dropna=True)
        cat, reason=category(c)
        records.append(dict(dataset=name,column=c,category=cat,
            decision='target' if cat=='TARGET' else ('keep' if reason.startswith('Keep') else 'drop'),reason=reason,
            missing=int(d[c].isna().sum()),non_numeric=int((d[c].notna() & v.isna()).sum()),
            unique=int(d[c].nunique()),is_constant=bool(d[c].nunique()<=1),
            is_near_constant=bool(len(counts) and counts.iloc[0]/len(d)>=.95),
            dominant_fraction=float(counts.iloc[0]/len(d)) if len(counts) else 0,
            minimum=float(v.min()) if v.notna().any() else None,maximum=float(v.max()) if v.notna().any() else None,
            sentinel_minus9999=int((v==-9999).sum()),value_minus1=int((v==-1).sum()),value_99=int((v==99).sum())))
    ids=[c for c in ['ID','Object_ID','Spectroscopic_Object_ID'] if c in d]
    result=dict(dataset=name,rows=len(d),columns=len(d.columns),sha256=digest(path),
        memory_mib=float(d.memory_usage(deep=True).sum()/1024**2),
        class_counts=d[target].str.strip().str.upper().value_counts().to_dict(),
        full_row_duplicates=int(d.duplicated().sum()),
        identifier_duplicates={c:int(d[c].dropna().duplicated().sum()) for c in ids},
        identifier_examples={c:d[c].value_counts().head(5).to_dict() for c in ids},
        dtypes=d.dtypes.astype(str).to_dict(),head=d.head(3).fillna('<missing>').to_dict('records'))
    log(f'Phase 1 audit {name}: shape={d.shape}, memory={result["memory_mib"]:.1f} MiB, labels={result["class_counts"]}')
    save_json(out/f'audit_{name}.json',result)
    return result,records


def normalize(d, source):
    rename={'Class':'Object_Class','ID':'Object_ID','Right_Ascension':'Right_Ascension_deg','Declination':'Declination_deg'}
    rename.update({f'{b}_Magnitude':f'{b}_Band_Magnitude' for b in BANDS})
    d=d.rename(columns=rename).copy()
    d['Object_Class']=d['Object_Class'].str.strip().str.upper()
    d['source']=source
    d['source_row']=np.arange(len(d))
    return d


def identity_groups(a,b):
    """Conservative transitive linkage: 1 arcsec positions OR >=3 exact bands.

    Coarse scientific-notation IDs in Dataset29 cannot identify sources. No
    numeric conversion of those IDs can recover the lost significant digits.
    """
    all_rows=pd.concat([a,b],ignore_index=True)
    parent=np.arange(len(all_rows))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    def union(i,j):
        x,y=find(i),find(j)
        if x!=y: parent[max(x,y)]=min(x,y)
    coords=all_rows[['Right_Ascension_deg','Declination_deg']].apply(pd.to_numeric,errors='coerce')
    valid=coords.iloc[:,0].between(0,360)&coords.iloc[:,1].between(-90,90)
    idx=np.flatnonzero(valid)
    ra,dec=np.deg2rad(coords.loc[valid].to_numpy()).T
    xyz=np.column_stack([np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)])
    pairs=cKDTree(xyz).query_pairs(2*np.sin(np.deg2rad(1/3600)/2),output_type='ndarray')
    for i,j in pairs: union(idx[i],idx[j])
    phot=clean_numeric(all_rows[MAG]).round(5)
    links=0
    for cols in itertools.combinations(MAG,3):
        good=phot[list(cols)].notna().all(axis=1)
        frame=phot.loc[good,list(cols)]
        hashes=pd.util.hash_pandas_object(frame,index=False).to_numpy()
        seen={}
        for row,h in zip(frame.index,hashes):
            if h in seen:
                # Exact tuple check protects against hash collisions.
                previous=seen[h]
                if np.array_equal(frame.loc[row].to_numpy(),frame.loc[previous].to_numpy()):
                    union(row,previous); links+=1
            else: seen[h]=row
    # Dataset24 retains full ID strings. Group them within that file only.
    seen={}
    for i,v in all_rows['Object_ID'].items():
        if i<len(a) or pd.isna(v): continue
        if v in seen: union(i,seen[v])
        else: seen[v]=i
    groups=np.array([find(i) for i in range(len(all_rows))])
    identifiable=valid.to_numpy() | (phot.notna().sum(axis=1).to_numpy()>=3)
    all_rows['object_group']=groups
    all_rows['identifiable']=identifiable
    cross=set(groups[:len(a)])&set(groups[len(a):])
    return all_rows,dict(position_pairs=len(pairs),photometric_links=links,
        shared_object_groups=len(cross),dataset24_overlapping_rows=int(np.isin(groups[len(a):],list(cross)).sum()),
        unidentifiable_rows=int((~identifiable).sum()),total_groups=int(len(np.unique(groups))))


def make_pipeline(name='HistGB',mode='both',flags=False,redshift=False,k=None,params=None):
    features=AstronomyFeatures(mode,redshift)
    names=list(features.fit(pd.DataFrame(columns=RAW+['Redshift'])).get_feature_names_out())
    if flags: names+=['robust_extreme_flag','isolation_anomaly_flag']
    pre=PreprocessingPipelineBuilder.build_preprocessor(names,[],[],scale_numeric=(name=='Logistic'),robust_scaling=True)
    # Preserve all-empty inputs rather than shifting the feature schema.
    pre.transformers[0][1].named_steps['imputer'].set_params(keep_empty_features=True)
    if name=='Logistic': model=LogisticRegression(C=0.5,max_iter=1200,class_weight='balanced',random_state=SEED)
    elif name=='RandomForest': model=RandomForestClassifier(n_estimators=120,max_depth=20,min_samples_leaf=3,max_features='sqrt',class_weight='balanced_subsample',n_jobs=4,random_state=SEED)
    elif name=='Dummy': model=DummyClassifier(strategy='most_frequent')
    else: model=HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=31,learning_rate=0.08,l2_regularization=2,
            class_weight='balanced',early_stopping=False,random_state=SEED)
    if params: model.set_params(**params)
    steps=[('features',features),('anomalies',AnomalyFlags(flags)),('preprocessor',pre)]
    if k: steps.append(('selector',SelectKBest(f_classif,k=k)))
    return Pipeline(steps+[('model',model)])


def metrics(y,pred,prob):
    m=MetricsEvaluator.evaluate_classification(np.asarray(y),np.asarray(pred),np.asarray(prob),True)
    m['pr_auc']=float(average_precision_score(y,prob))
    m['brier']=float(brier_score_loss(y,prob))
    report=classification_report(y,pred,output_dict=True,zero_division=0)
    for c,label in [('0','GALAXY'),('1','STAR')]:
        for key in ['precision','recall','f1-score']:
            m[f'{label}_{key}']=float(report[c][key])
    return m


def predict_metrics(model,X,y):
    prob=model.predict_proba(X)[:,1]
    return metrics(y,(prob>=0.5).astype(int),prob)


def cv_run(pipeline,X,y,folds,name,out):
    rows=[]; oof=np.zeros(len(y)); first=None
    for fold,(tr,va) in enumerate(folds):
        t=time.monotonic()
        model=clone(pipeline)
        with threadpool_limits(limits=4):
            model.fit(X.iloc[tr],y.iloc[tr])
            prob=model.predict_proba(X.iloc[va])[:,1]
        oof[va]=prob
        m=metrics(y.iloc[va],(prob>=0.5).astype(int),prob)
        rows.append(dict(candidate=name,fold=fold+1,fit_seconds=time.monotonic()-t,**m))
        if fold==0: first=model
        log(f'Phase 5 {name} fold {fold+1}/5: macro-F1={m["f1_macro"]:.4f} ({time.monotonic()-t:.1f}s)')
    frame=pd.DataFrame(rows)
    frame.to_csv(out/f'cv_{name}.csv',index=False)
    summary={'candidate':name}
    for c in frame.select_dtypes('number'):
        if c=='fold': continue
        summary[c+'_mean']=float(frame[c].mean()); summary[c+'_std']=float(frame[c].std(ddof=1))
    return summary,oof,first


def plot_eda(X,y,out):
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,c in zip(axes.flat,MAG):
        for label,name,color in [(0,'GALAXY','#b45a27'),(1,'STAR','#246ea3')]:
            ax.hist(X.loc[y==label,c].dropna(),bins=45,density=True,alpha=.55,label=name,color=color)
        ax.set_title(c.replace('_Band_Magnitude',' band')); ax.set_xlabel('Magnitude'); ax.set_ylabel('Density')
    axes.flat[-1].bar(['GALAXY','STAR'],[(y==0).sum(),(y==1).sum()],color=['#b45a27','#246ea3'])
    axes.flat[-1].set_title('Development class counts'); axes.flat[0].legend()
    fig.suptitle('Development sample: brightness and class balance'); fig.tight_layout(); fig.savefig(out/'01_magnitudes.png',dpi=150); plt.close(fig)
    features=AstronomyFeatures().fit_transform(X)
    corr=features.corr()
    fig,ax=plt.subplots(figsize=(11,9)); im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
    def short_label(c):
        for long,short in [('Petrosian_50_Percent_Radius_','R50 '),('Petrosian_Radius_','Rp '),
            ('Petrosian_Flux_','Flux '),('Exponential_Axis_Ratio_','b/a '),
            ('PSF_Magnitude_','PSF '),('_Band_Magnitude',' mag'),
            ('radius_over_half_light_','Rp/R50 '),('psf_minus_band_','PSF-band '),('log1p_','log '),('asinh_','asinh ')]:
            c=c.replace(long,short)
        return c
    ticks=np.arange(0,len(corr),3); ticklabels=[short_label(c) for c in corr.columns[ticks]]
    ax.set_xticks(ticks,ticklabels,rotation=90,fontsize=8); ax.set_yticks(ticks,ticklabels,fontsize=8)
    fig.colorbar(im,ax=ax,label='Pearson correlation (pairwise complete)'); ax.set_title('Development feature correlation'); fig.tight_layout(); fig.savefig(out/'02_correlation.png',dpi=150); plt.close(fig)
    pairs=[dict(feature_a=corr.columns[i],feature_b=corr.columns[j],correlation=corr.iloc[i,j]) for i in range(len(corr)) for j in range(i) if abs(corr.iloc[i,j])>.95]
    pd.DataFrame(pairs).to_csv(out.parent/'high_correlations.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    sample=features.sample(min(7000,len(features)),random_state=42)
    for label,name,color in [(0,'GALAXY','#b45a27'),(1,'STAR','#246ea3')]:
        s=sample.loc[y.loc[sample.index]==label]
        axes[0].scatter(s['colour_G_R'],s['colour_R_I'],s=3,alpha=.3,label=name,color=color)
        axes[1].scatter(s['R_Band_Magnitude'],s['psf_minus_band_R'],s=3,alpha=.3,label=name,color=color)
    axes[0].set(xlabel='g - r',ylabel='r - i',xlim=(-2,4),ylim=(-2,3),title='Optical colour space')
    axes[1].set(xlabel='r magnitude',ylabel='PSF r - supplied r magnitude',ylim=(-3,8),title='Empirical brightness-profile contrast')
    axes[0].legend(); fig.tight_layout(); fig.savefig(out/'03_colour_morphology.png',dpi=150); plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data29',type=Path,default=ROOT/'selected_datasets'/'Dataset 29.csv')
    parser.add_argument('--data24',type=Path,default=ROOT/'selected_datasets'/'Dataset 24.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    figures=out/'figures'; figures.mkdir(exist_ok=True)
    log('Starting Deep Sky Divide; sequential jobs, four computation threads, seed 42')
    d29=pd.read_csv(args.data29,dtype=str); d24=pd.read_csv(args.data24,dtype=str)
    audits=[]; decisions=[]
    for d,name,path in [(d29,'29',args.data29),(d24,'24',args.data24)]:
        a,r=audit(d,name,path,out); audits.append(a); decisions+=r
    pd.DataFrame(decisions).to_csv(out/'02_feature_decisions.csv',index=False)
    a=normalize(d29,'29'); b=normalize(d24,'24')
    all_rows,linkage=identity_groups(a,b)
    save_json(out/'identity_linkage.json',linkage)
    log(f'Phase 1 identity linkage: {linkage}')
    binary=all_rows['Object_Class'].isin(['STAR','GALAXY'])
    conflicts=all_rows.loc[binary].groupby('object_group')['Object_Class'].nunique()
    conflict_groups=set(conflicts[conflicts>1].index)
    allowed=binary & all_rows['identifiable'] & ~all_rows['object_group'].isin(conflict_groups)
    primary=all_rows.loc[allowed & (all_rows.source=='29')].copy()
    clean=clean_numeric(primary[RAW])
    primary['_complete']=clean.notna().sum(axis=1)
    before=len(primary)
    primary=primary.sort_values(['_complete','source_row'],ascending=[False,True]).drop_duplicates('object_group').sort_values('source_row').reset_index(drop=True)
    removed_duplicates=before-len(primary)
    # Exclude every Dataset29 identity, including unlabeled/quarantined sources,
    # from Dataset24 evaluation. Never fill morphology across catalogues.
    groups29=set(all_rows.loc[all_rows.source=='29','object_group'])
    secondary=all_rows.loc[allowed & (all_rows.source=='24') & ~all_rows.object_group.isin(groups29)].drop_duplicates('object_group').reset_index(drop=True)
    assert not (set(primary.object_group)&set(secondary.object_group))
    labels=(primary.Object_Class=='STAR').astype(int)
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    dev_idx,test_idx=next(outer.split(primary,labels,primary.object_group))
    assert not (set(primary.iloc[dev_idx].object_group)&set(primary.iloc[test_idx].object_group))
    development=primary.iloc[dev_idx].reset_index(drop=True)
    X=development.reindex(columns=RAW+['Redshift']).copy(); y=(development.Object_Class=='STAR').astype(int)
    groups=development.object_group
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,y,groups))
    for tr,va in folds: assert not (set(groups.iloc[tr])&set(groups.iloc[va]))
    split=primary[['source','source_row','object_group','Object_Class']].copy(); split['partition']='development'; split.loc[test_idx,'partition']='test'
    split.to_csv(out/'split_assignments.csv',index=False)
    primary.to_csv(out/'primary_rows.csv',index=False)
    counts=dict(primary_rows=len(primary),development_rows=len(dev_idx),test_rows=len(test_idx),
        secondary_rows=len(secondary),conflicting_groups=len(conflict_groups),duplicate_primary_rows_removed=removed_duplicates)
    save_json(out/'partition_summary.json',counts)
    audit_text=f'''# Data audit\n\nTwo files verified by SHA-256 (see audit JSON). Dataset 29: {len(d29):,} rows / {len(d29.columns)} columns. Dataset 24: {len(d24):,} / {len(d24.columns)}.\n\nThese are partially overlapping survey samples with different feature coverage, not a ready train/test pair. Keep Dataset29 for the structural model; use nonoverlapping Dataset24 only for a separately trained five-band transfer check. No row concatenation or morphology join. Dataset29 IDs use rounded scientific notation, so cannot be reliable join keys.\n\nIdentity rule: link within 1 arcsecond, or exact agreement (5 decimal places) on any three valid ugriz bands, transitively. Full Dataset24 IDs also link records within that file. This is conservative and can merge close sources. Exclude rows with neither valid position nor three valid bands. Ambiguous mixed-label groups are quarantined. It prevents detected duplicates crossing partitions, but unrecognized identities remain a limitation.\n\nLinkage: {json.dumps(linkage)}\n\nRetained partitions: {json.dumps(counts)}\n\nLabels: {audits[0]['class_counts']} (29); {audits[1]['class_counts']} (24). QSO is excluded: the task is binary, not QSO detection.\n\nAll fields loaded as strings to protect IDs and expose contamination. Full per-column ranges, missingness, unique counts, dominant fractions, and sentinel occurrences are in 02_feature_decisions.csv. Original dtypes/head/memory are in audit JSON.\n\nFixed cleaning assumptions: magnitudes [-5,40], angular radii (0,1000], axis ratios (0,1], absolute flux <1e7; nonnumeric/nonfinite values become missing. These deliberately broad bounds reject corruption such as +/-1e12. Units follow SDSS-like names, without an organizer dictionary. Do not blanket-replace -1/99; their meaning is column-dependent. Negative flux is allowed. Bad cells are masked, not entire scientifically unusual rows.\n\nIDs/bookkeeping/position/notes are excluded from predictors. Redshift is diagnostic-only; spectral type and galaxy population have unverified derivation and are excluded. Median imputation and scaling are fitted within training folds. Final test is sealed until selection and development diagnostics finish.\n'''
    (out/'01_audit.md').write_text(audit_text,encoding='utf-8')
    log(f'Phase 1 complete: {counts}; sealed final test; exact object-group intersections=0')
    if args.audit_only: return
    run_experiments(X,y,groups,folds,development,primary,test_idx,secondary,counts,audits,linkage,out,figures)


def run_experiments(X,y,groups,folds,development,primary,test_idx,secondary,counts,audits,linkage,out,figures):
    # Fixed cleaning audit and single-feature diagnostics only on development.
    cleaned=clean_numeric(X)
    numeric=X.apply(pd.to_numeric,errors='coerce')
    invalid=(numeric.notna() & cleaned.isna())
    quality=pd.DataFrame({'missing_after_cleaning':cleaned.isna().sum(),
        'invalid_numeric_masked':invalid.sum(),'nonnumeric_tokens':(X.notna()&numeric.isna()).sum()})
    quality.to_csv(out/'cleaning_counts.csv')
    single=[]
    for c in cleaned:
        good=cleaned[c].notna()
        if good.sum()>50 and y[good].nunique()==2:
            auc=roc_auc_score(y[good],cleaned.loc[good,c]); strength=max(auc,1-auc)
            single.append(dict(feature=c,rows=int(good.sum()),auc=float(auc),direction_free_auc=float(strength),flag_gt099=bool(strength>.99)))
    pd.DataFrame(single).sort_values('direction_free_auc',ascending=False).to_csv(out/'single_feature_auc.csv',index=False)
    log(f'Phase 2 complete: {int(invalid.sum().sum())} invalid numeric cells masked in development; leakage candidates saved')
    plot_eda(cleaned,y,figures)
    log('Phase 3 complete: magnitude, colour/profile, correlation plots and engineered features saved')
    tr,va=folds[0]
    feat=AstronomyFeatures().fit_transform(X.iloc[tr])
    anomaly=AnomalyFlags(True).fit(feat)
    val_features=AstronomyFeatures().fit_transform(X.iloc[va])
    flagged=anomaly.transform(val_features)
    flag_summary=[]
    for flag in ['robust_extreme_flag','isolation_anomaly_flag']:
        for label,name in [(0,'GALAXY'),(1,'STAR')]:
            mask=y.iloc[va].to_numpy()==label
            flag_summary.append(dict(method=flag,label=name,rows=int(mask.sum()),flagged=int(flagged.loc[mask,flag].sum())))
    pd.DataFrame(flag_summary).to_csv(out/'outlier_counts.csv',index=False)
    examples=development.iloc[va].copy(); examples['robust_flag']=flagged['robust_extreme_flag'].to_numpy(); examples['isolation_flag']=flagged['isolation_anomaly_flag'].to_numpy()
    examples.loc[(examples.robust_flag==1)|(examples.isolation_flag==1)].head(100).to_csv(out/'outlier_examples.csv',index=False)
    log('Phase 4 complete: training-fitted robust-IQR and IsolationForest flags; keep unusual sources, mask impossible cells')
    configs={'Dummy':make_pipeline('Dummy'),'Logistic':make_pipeline('Logistic'),
        'RandomForest':make_pipeline('RandomForest'),'HistGB':make_pipeline(),
        'HistGB_flags':make_pipeline(flags=True),
        'HistGB_small':make_pipeline(params={'max_leaf_nodes':15,'l2_regularization':5}),
        'HistGB_deeper':make_pipeline(params={'max_leaf_nodes':47,'min_samples_leaf':35})}
    summaries=[]; candidate_objects=[]; first_models={}; oofs={}
    tuning_start=None
    for name,pipe in configs.items():
        if name=='HistGB_small': tuning_start=time.monotonic()
        if name=='HistGB_deeper' and time.monotonic()-tuning_start>240:
            log('Skipping second tuning variant to respect five-minute tuning budget'); continue
        summary,oof,first=cv_run(pipe,X,y,folds,name,out)
        summaries.append(summary); first_models[name]=first; oofs[name]=oof
        means={c[:-5]:v for c,v in summary.items() if c.endswith('_mean')}
        candidate_objects.append(ChampionCandidate(name,'sklearn',name,ProblemType.BINARY_CLASSIFICATION,
            means,'f1_macro',means['f1_macro'],class_labels=[0,1]))
        pd.DataFrame(summaries).to_csv(out/'cv_summary.csv',index=False)
    selection=ChampionSelector.select_champion(candidate_objects,ProblemType.BINARY_CLASSIFICATION,primary_metric='f1_macro')
    winner=selection.champion.model_name
    best_template=configs[winner]
    log(f'Phase 5 selection: {winner}, mean five-fold macro-F1={selection.champion.primary_val_score:.4f}')
    # Diagnostic experiments use the first development fold, never final test.
    diagnostics=[]
    def experiment(name,pipe,xtrain=None,xval=None,ytrain=None,yval=None):
        xx=X.iloc[tr] if xtrain is None else xtrain; xv=X.iloc[va] if xval is None else xval
        yy=y.iloc[tr] if ytrain is None else ytrain; yv=y.iloc[va] if yval is None else yval
        with threadpool_limits(limits=4):
            pipe.fit(xx,yy); result=predict_metrics(pipe,xv,yv)
        diagnostics.append(dict(experiment=name,evaluation='development_holdout',rows=len(yv),**result))
        log(f'Phase 6 {name}: development macro-F1={result["f1_macro"]:.4f}')
        return pipe
    diagnostics.append(dict(experiment='both_HistGB',evaluation='development_holdout',rows=len(va),**predict_metrics(first_models['HistGB'],X.iloc[va],y.iloc[va])))
    experiment('with_redshift_diagnostic_only',make_pipeline(redshift=True))
    experiment('photometric_only',make_pipeline(mode='photometric'))
    experiment('structural_only',make_pipeline(mode='structural'))
    experiment('top15_training_fold_ANOVA',make_pipeline(k=15))
    first=first_models[winner]
    noisy=X.iloc[va].copy(); rng=np.random.default_rng(SEED)
    for c in MAG+PSF:
        noisy[c]=pd.to_numeric(noisy[c],errors='coerce')+rng.normal(0,.02,len(noisy))
    diagnostics.append(dict(experiment='winner_0.02mag_noise',evaluation='development_holdout',rows=len(va),**predict_metrics(first,noisy,y.iloc[va])))
    alttr,altva=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=123).split(X,y,groups))
    experiment('winner_seed123',clone(best_template),X.iloc[alttr],X.iloc[altva],y.iloc[alttr],y.iloc[altva])
    ra=pd.to_numeric(development.Right_Ascension_deg,errors='coerce'); dec=pd.to_numeric(development.Declination_deg,errors='coerce')
    valid=ra.between(0,360)&dec.between(-90,90)
    positions=np.flatnonzero(valid)
    sky=(np.floor(ra[valid]/20)*10+np.floor((dec[valid]+90)/20)).astype(int)
    gtr,gva=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(X.iloc[positions],y.iloc[positions],sky))
    experiment('winner_disjoint_20degree_sky_cells',clone(best_template),X.iloc[positions[gtr]],X.iloc[positions[gva]],y.iloc[positions[gtr]],y.iloc[positions[gva]])
    pd.DataFrame(diagnostics).to_csv(out/'robustness.csv',index=False)
    # Training-only derived ANOVA selection above is deliberately independent of test importance.
    sample_va=np.random.default_rng(SEED).choice(va,size=min(2500,len(va)),replace=False)
    with threadpool_limits(limits=4):
        pi=permutation_importance(first,X.iloc[sample_va][RAW],y.iloc[sample_va],scoring='f1_macro',n_repeats=2,random_state=42,n_jobs=1)
    pd.DataFrame(dict(feature=RAW,importance=pi.importances_mean,std=pi.importances_std)).sort_values('importance',ascending=False).to_csv(out/'importance_development.csv',index=False)
    # A tree baseline supplies native feature importance even if the winner is HistGB.
    rf=first_models['RandomForest']; fn=rf.named_steps['preprocessor'].get_feature_names_out()
    pd.DataFrame(dict(feature=fn,importance=rf.named_steps['model'].feature_importances_)).sort_values('importance',ascending=False).to_csv(out/'importance_model_randomforest.csv',index=False)
    save_json(out/'selection.json',dict(winner=winner,criterion='mean development five-fold macro-F1',
        score=selection.champion.primary_val_score,threshold=.5,seed=42,test_used_for_decisions=False,
        final_input_features=RAW,final_engineered_features=list(first.named_steps['features'].get_feature_names_out()),
        strategy='balanced class weights; fixed 0.5 threshold; no calibration fit; no test tuning',
        diagnostics_completed=True,params=best_template.get_params(deep=False).__repr__()))
    log('All selection and robustness decisions frozen. Refitting champion on development; opening final test next.')
    with threadpool_limits(limits=4):
        final=clone(best_template).fit(X,y)
    # No further model/feature/threshold choices after this point.
    test=primary.iloc[test_idx].copy(); xt=test[RAW]; yt=(test.Object_Class=='STAR').astype(int)
    with threadpool_limits(limits=4): prob=final.predict_proba(xt)[:,1]
    pred=(prob>=.5).astype(int); test_metrics=metrics(yt,pred,prob)
    save_json(out/'test_metrics.json',test_metrics)
    cm=confusion_matrix(yt,pred,labels=[0,1]); save_json(out/'confusion_matrix.json',{'labels':['GALAXY','STAR'],'matrix':cm.tolist()})
    predictions=test[['source','source_row','object_group','Object_ID','Object_Class']].copy()
    predictions['predicted_class']=np.where(pred==1,'STAR','GALAXY'); predictions['probability_STAR']=prob; predictions['probability_GALAXY']=1-prob
    predictions.to_csv(out/'predictions.csv',index=False)
    predictions.loc[pred!=yt.to_numpy()].join(clean_numeric(xt)).to_csv(out/'misclassified_objects.csv',index=False)
    selection.champion.model_instance=final
    final_eval=FinalTestEvaluationResult(winner,'sklearn',ProblemType.BINARY_CLASSIFICATION,test_metrics,len(test),digest(out/'predictions.csv'))
    partition=dict(**counts,protocol='20% grouped holdout; stratified grouped 5-fold development CV',
        train_sha256=hashlib.sha256(development.object_group.to_numpy().tobytes()).hexdigest(),
        test_sha256=hashlib.sha256(test.object_group.to_numpy().tobytes()).hexdigest(),object_group_overlap=0)
    final_eval.test_sha256=partition['test_sha256']
    ChampionStore.save_champion(selection,final_eval,out,RAW,{c:'numeric' for c in RAW},'Object_Class',
        ProblemType.BINARY_CLASSIFICATION,partition_info=partition,random_state=42)
    loaded=UnifiedPredictor.load(out)
    np.testing.assert_allclose(loaded.predict_proba(xt.head(20))[:,1],prob[:20],rtol=1e-12)
    log(f'Final test: macro-F1={test_metrics["f1_macro"]:.4f}, ROC-AUC={test_metrics["roc_auc"]:.4f}; persisted inference verified')
    # Descriptive test permutation only; never used to select final features.
    positions=np.random.default_rng(SEED).choice(len(xt),min(2500,len(xt)),replace=False)
    with threadpool_limits(limits=4):
        pi=permutation_importance(final,xt.iloc[positions],yt.iloc[positions],scoring='f1_macro',n_repeats=2,random_state=42,n_jobs=1)
    importance=pd.DataFrame(dict(feature=RAW,importance=pi.importances_mean,std=pi.importances_std)).sort_values('importance',ascending=False)
    importance.to_csv(out/'importance_test_descriptive_only.csv',index=False)
    # Predeclared transfer diagnostic with shared ugriz features, not morphology.
    with threadpool_limits(limits=4):
        band_model=make_pipeline(mode='bands').fit(X,y)
        transfer=predict_metrics(band_model,secondary,(secondary.Object_Class=='STAR').astype(int))
        band_test=predict_metrics(band_model,xt,yt)
    save_json(out/'transfer_metrics.json',dict(dataset24_nonoverlap=transfer,dataset29_holdout_shared_bands=band_test,
        rows=len(secondary),interpretation='different five-band model; cannot evaluate structural champion on Dataset24'))
    joblib.dump(band_model,out/'five_band_comparison.joblib')
    # Postfreeze diagnostics grouped by data quality and confidence.
    quality_test=clean_numeric(xt).isna().any(axis=1).to_numpy(); subgroup=[]
    for name,mask in [('any_missing_or_invalid',quality_test),('complete_measurements',~quality_test)]:
        if mask.sum() and yt.iloc[np.flatnonzero(mask)].nunique()==2:
            subgroup.append(dict(group=name,rows=int(mask.sum()),**metrics(yt.iloc[np.flatnonzero(mask)],pred[mask],prob[mask])))
    pd.DataFrame(subgroup).to_csv(out/'test_quality_subgroups.csv',index=False)
    make_final_plots(yt,prob,cm,importance,figures)
    write_report(out,counts,linkage,summaries,winner,test_metrics,diagnostics,importance,transfer,band_test,quality,flag_summary)
    save_json(out/'run_summary.json',dict(elapsed_seconds=time.monotonic()-START,completed=True,winner=winner,
        final_test=test_metrics,reload_prediction_match=True,test_used_for_selection=False,input_hashes=[a['sha256'] for a in audits]))
    import subprocess,sys
    frozen=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (out/'requirements-lock.txt').write_text(frozen,encoding='utf-8')
    log(f'Phase 7 complete. Final inputs={len(RAW)}, engineered={len(final.named_steps["features"].get_feature_names_out())}; best={winner}; CV={selection.champion.primary_val_score:.4f}; test macro-F1={test_metrics["f1_macro"]:.4f}')
    log('Top five findings are recorded in REPORT.md. Model, predictions, figures and provenance saved.')


def make_final_plots(y,p,cm,importance,figures):
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2): axes[0].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    axes[0].set(xticks=[0,1],yticks=[0,1],xticklabels=['GALAXY','STAR'],yticklabels=['GALAXY','STAR'],xlabel='Predicted',ylabel='True',title='Frozen champion: final test')
    fpr,tpr,_=roc_curve(y,p); axes[1].plot(fpr,tpr,label=f'AUC={roc_auc_score(y,p):.4f}'); axes[1].plot([0,1],[0,1],'--',color='grey'); axes[1].set(xlabel='False-positive rate',ylabel='True-positive rate',title='STAR-positive ROC'); axes[1].legend()
    fig.tight_layout(); fig.savefig(figures/'04_test_confusion_roc.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5)); top=importance.head(12).iloc[::-1]; ax.barh(top.feature,top.importance,xerr=top['std'],color='#246ea3'); ax.set(xlabel='Decrease in macro-F1 after permutation',title='Test permutation importance (descriptive only; 2,500 sampled rows)'); fig.tight_layout(); fig.savefig(figures/'05_importance.png',dpi=150); plt.close(fig)
    frac,mean=calibration_curve(y,p,n_bins=10,strategy='quantile')
    fig,axes=plt.subplots(1,2,figsize=(10,4)); axes[0].plot(mean,frac,'o-'); axes[0].plot([0,1],[0,1],'--',color='grey'); axes[0].set(xlabel='Mean predicted P(STAR)',ylabel='Observed STAR fraction',title=f'Reliability; Brier={brier_score_loss(y,p):.4f}')
    axes[1].hist(p,bins=30,color='#246ea3'); axes[1].set(xlabel='Predicted P(STAR)',ylabel='Test objects',title='Probability distribution'); fig.tight_layout(); fig.savefig(figures/'06_calibration.png',dpi=150); plt.close(fig)


def write_report(out,counts,linkage,summaries,winner,test,diagnostics,importance,transfer,band_test,quality,flags):
    ranked=sorted(summaries,key=lambda x:x['f1_macro_mean'],reverse=True)
    table='\n'.join(f"| {r['candidate']} | {r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f} | {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f} |" for r in ranked)
    diag={r['experiment']:r for r in diagnostics}
    redgap=diag['with_redshift_diagnostic_only']['f1_macro']-diag['both_HistGB']['f1_macro']
    tops=', '.join(importance.head(5).feature)
    report=f'''# Deep Sky Divide — Astromanthan 2026

**Result:** {winner}, selected by mean five-fold development macro-F1. Final held-out macro-F1 **{test['f1_macro']:.4f}**, accuracy **{test['accuracy']:.4f}**, ROC-AUC **{test['roc_auc']:.4f}**, STAR-positive PR-AUC (average precision) **{test['pr_auc']:.4f}**. No test-based model, feature or threshold selection. **Cross-catalogue transfer is poor:** the separate five-band model scores only {transfer['f1_macro']:.4f} macro-F1 on Dataset24; the main score does not establish universal generalization.

## Data and evaluation
Dataset29 provides 100,050 rows and 46 columns; Dataset24 provides 99,999 and 12. Both include STAR/GALAXY/QSO; exclude QSO for this binary task. The schemas and populations differ. Use Dataset29 for morphology and nonoverlapping Dataset24 for a five-band transfer diagnostic. Rounded Dataset29 IDs cannot support reliable joins. Conservative linkage uses 1-arcsecond coordinates or any three exact valid bands, plus full IDs within Dataset24. Detected overlap: **{linkage['dataset24_overlapping_rows']:,} Dataset24 rows**. Quarantine {counts['conflicting_groups']:,} conflicting-label groups; collapse {counts['duplicate_primary_rows_removed']:,} repeated primary rows; exclude unidentifiable records. Retain **{counts['primary_rows']:,} primary sources**, {counts['development_rows']:,} development / {counts['test_rows']:,} test, and {counts['secondary_rows']:,} secondary sources. Group intersections are zero under these matching rules. Unknown residual duplicates remain possible.

20% stratified group holdout, seed 42; five stratified group folds on remaining 80%. Fit cleaning-derived feature transforms, median imputers, robust scaling, selectors and anomaly detectors inside each training fold. Fixed physical bounds do not learn from data. Class weights are balanced; probability threshold stays 0.5. All ablations and robustness choices precede final test. CV is selection evidence, not an unbiased final estimate.

## Cleaning, leakage and features
All columns are classified in `02_feature_decisions.csv`. Exclude IDs, observation bookkeeping, coordinates, arbitrary notes and spectroscopy/derived labels. Convert invalid tokens/nonfinite values to missing; use broad documented magnitude/radius/axis/flux bounds. Mask {int(quality.invalid_numeric_masked.sum()):,} impossible numeric cells in development and impute within folds. Do not blanket-delete -1, 99, negative flux or unusual sources. Bounds assume SDSS-like units because no dictionary was supplied.

Keep 30 measured photometric/structural inputs. Add five colours, signed asinh fluxes, log radii, radius/half-light ratios, and PSF-minus-supplied-band differences. Generic magnitudes are not documented as cmodel; differences are empirical morphology proxies. R90 is absent, so no conventional R90/R50 concentration is claimed. Correlations above |r|=0.95 are logged; correlated predictors can share importance. Redshift ablation macro-F1 gap: **{redgap:+.4f}** on the same development holdout. Redshift remains excluded regardless of score because final inputs must be photometric. Single-feature AUC>0.99 flags are saved, not automatically interpreted as leakage when structurally meaningful.

Two anomaly methods: robust IQR-scaled extremes and IsolationForest (2% training contamination). Keep/flag unusual observations; never delete by anomaly score or class. Compare HistGB with and without flags across the same five folds. Invalid measurements are masked in both. Outlier counts/examples and the impact are saved.

## Model comparison
| Candidate | CV macro-F1, mean ± SD | CV ROC-AUC, mean ± SD |
|---|---:|---:|
{table}

Efficient predefined settings plus at most two bounded boosting variants replace a larger randomized search. HistGradientBoosting is used because external boosters/AutoGluon are absent. The existing engine supplies preprocessing, metrics, champion selection, persistence and UnifiedPredictor; historical employee/house models are not astronomy transfer models.

## Interpretation and robustness
Test GALAXY precision/recall/F1: {test['GALAXY_precision']:.4f}/{test['GALAXY_recall']:.4f}/{test['GALAXY_f1-score']:.4f}; STAR: {test['STAR_precision']:.4f}/{test['STAR_recall']:.4f}/{test['STAR_f1-score']:.4f}. Brier **{test['brier']:.4f}**; reliability plot reports uncalibrated model probabilities. Confusion matrix, error rows and data-quality subgroups are provided.

Top raw-input permutation features: **{tops}**. Sizes/profile differences can separate extended galaxies from point-like stars; colours reflect spectral-energy distributions. These are predictive associations, not causal explanations. Native RF importance is labeled as a baseline's importance. Permutation importance is computed on development and, only after freezing, a 2,500-row test sample. Test importance never selects top-k; the top-15 ablation fits ANOVA on its training fold. SHAP is omitted for time.

Development macro-F1: photometric **{diag['photometric_only']['f1_macro']:.4f}**, structural **{diag['structural_only']['f1_macro']:.4f}**, both (fixed HistGB) **{diag['both_HistGB']['f1_macro']:.4f}**; top-15 **{diag['top15_training_fold_ANOVA']['f1_macro']:.4f}**. Winner with 0.02-mag Gaussian perturbations **{diag['winner_0.02mag_noise']['f1_macro']:.4f}**; seed123 split **{diag['winner_seed123']['f1_macro']:.4f}**; disjoint 20-degree sky cells **{diag['winner_disjoint_20degree_sky_cells']['f1_macro']:.4f}**. These are individual development stress tests, not five-fold estimates. Shared-band transfer macro-F1: Dataset29 test **{band_test['f1_macro']:.4f}**, nonoverlapping Dataset24 **{transfer['f1_macro']:.4f}**; it uses a separate five-band model, not the structural champion.

## Five findings and limitations
1. Structural and photometric coverage makes Dataset29 the primary sample; Dataset24 supports limited-feature transfer only.
2. Catalogues overlap and Dataset29 IDs have lost precision; object linkage is necessary before splitting.
3. Corrupted cells require validity masking; unusual valid sources should not be blindly removed.
4. CV selects {winner}; final holdout confirms the measured score above without post-test tuning.
5. Five-band transfer falls from {band_test['f1_macro']:.4f} to {transfer['f1_macro']:.4f} macro-F1: a major generalization limitation. No tuning follows this result.

Limitations: inferred units, label provenance unverified, imperfect crossmatching, no seeing/extinction/error model, no independent telescope survey, no QSO rejection, and balanced weights can affect probability calibration. More time: recover precise source IDs/data dictionary, validate spectroscopy-based ground truth, add uncertainty/seeing features, nested spatial CV, and calibration fitted on development folds. Full per-fold precision/recall/F1/ROC/PR metrics and assumptions are in the companion files.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')


if __name__=='__main__':
    main()
