"""Supplementary proxy audit: uses the development partition only."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT=Path(__file__).resolve().parents[1]


def main():
    raw=pd.read_csv(ROOT/'selected_datasets'/'Dataset 29.csv',dtype=str)
    split=pd.read_csv(ROOT/'outputs'/'split_assignments.csv')
    rows=split.loc[split.partition=='development','source_row'].to_numpy()
    d=raw.iloc[rows].reset_index(drop=True)
    y=(d.Object_Class=='STAR').astype(int)
    train,val=next(StratifiedKFold(5,shuffle=True,random_state=42).split(d,y))
    results=[]
    for col in d:
        if col=='Object_Class': continue
        n=pd.to_numeric(d[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
        if n.notna().mean()>.5 and n.nunique()>1:
            model=make_pipeline(SimpleImputer(strategy='median'),DecisionTreeClassifier(max_depth=3,min_samples_leaf=100,random_state=42))
            model.fit(n.iloc[train].to_frame(),y.iloc[train]); p=model.predict_proba(n.iloc[val].to_frame())[:,1]
            results.append(dict(column=col,method='depth3_tree_holdout',auc=roc_auc_score(y.iloc[val],p)))
        elif d[col].nunique(dropna=True)<=30:
            # A category's star fraction is learned from training only. Unknown
            # categories receive the training prior, not validation information.
            fit=pd.DataFrame({'category':d[col].iloc[train].fillna('<missing>'),'y':y.iloc[train]})
            means=fit.groupby('category').y.mean()
            p=d[col].iloc[val].fillna('<missing>').map(means).fillna(y.iloc[train].mean())
            results.append(dict(column=col,method='training_category_fraction_holdout',auc=roc_auc_score(y.iloc[val],p)))
        else:
            results.append(dict(column=col,method='constant_or_high_cardinality_excluded',auc=np.nan))
    result=pd.DataFrame(results); result['flag_auc_gt099']=result.auc>.99
    result.sort_values('auc',ascending=False).to_csv(ROOT/'outputs'/'all_column_proxy_audit.csv',index=False)
    print(result.head().to_string(index=False))


if __name__=='__main__': main()
