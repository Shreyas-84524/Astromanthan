import json, zipfile
import pandas as pd
import numpy as np

z = zipfile.ZipFile(r'C:\Users\Kryo\Downloads\drive-download-20261009T074827Z-1-001.zip')
frames = {}
for n in [24,29,33,41]:
    d = pd.read_csv(z.open(f'Dataset {n}.csv'), low_memory=False)
    frames[n] = d
    label = 'Object_Class' if 'Object_Class' in d else 'Class'
    num = d.select_dtypes(include='number')
    print(json.dumps({'dataset':n,'shape':list(d.shape),'labels':d[label].fillna('<missing>').value_counts().to_dict(),'missing':{c:int(v) for c,v in d.isna().sum().items() if v},'duplicates':int(d.duplicated().sum()),'sentinels':{c:int(v) for c,v in num.isin([-9999,-999,9999]).sum().items() if v},'infinite':int(np.isinf(num).sum().sum())}))
    for c in d.columns:
        if any(w in c for w in ['Magnitude','Radius','Flux','Ratio']):
            v = pd.to_numeric(d[c], errors='coerce')
            invalid=d[c].notna() & v.isna()
            print('FEATURE',n,c,'missing',int(v.isna().sum()),'invalid',d.loc[invalid,c].value_counts().head(6).to_dict(),'minmax',[v.min(),v.max()])
common = list(frames[33].columns)
a,b = frames[29][common],frames[33][common]
print('29_33_SHARED_COLUMNS',len(common),'ROWS',len(a),len(b),'EXACT_EQUAL',a.equals(b))
ha,hb = pd.util.hash_pandas_object(a,index=False),pd.util.hash_pandas_object(b,index=False)
print('29_33_EXACT_ROW_OVERLAP',int(hb.isin(ha).sum()))
for n in [24,41]:
    d = frames[n]
    print('CATEGORIES',n,{c:d[c].value_counts(dropna=False).head(12).to_dict() for c in ['Spectral_Type','Galaxy_Population']})
aa=a.copy(); bb=b.copy()
for c in common[:-1]:
    aa[c]=pd.to_numeric(aa[c],errors='coerce')
    bb[c]=pd.to_numeric(bb[c],errors='coerce')
print('29_33_NUMERIC_ALIGNED_EQUAL',aa.equals(bb),'DIFFERENT_CELLS',int(((aa!=bb)&~(aa.isna()&bb.isna())).sum().sum()))
bands=list('UGRIZ')
p=frames[29][[f'{x}_Band_Magnitude' for x in bands]].apply(pd.to_numeric,errors='coerce').round(5)
q=frames[24][[f'{x}_Magnitude' for x in bands]].round(5)
q.columns=p.columns
hp=pd.util.hash_pandas_object(p.dropna(),index=False)
hq=pd.util.hash_pandas_object(q.dropna(),index=False)
print('24_29_COMPLETE_FIVE_BAND_MATCHES',int(hq.isin(hp).sum()),'COMPLETE_ROWS_29',len(hp))
from pathlib import Path
out=Path('selected_datasets'); out.mkdir(exist_ok=True)
for n in [24,29]:
    (out/f'Dataset {n}.csv').write_bytes(z.read(f'Dataset {n}.csv'))
