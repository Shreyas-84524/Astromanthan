"""Post-hoc presentation comparison on the project's frozen Dataset29 split.

The selected champion and original outputs are not changed. Baseline is defined
here as unregularized logistic regression; L1 and L2 use C=0.5 fixed before this
comparison. This is a descriptive comparison, not model selection.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
from pathlib import Path
import hashlib
import json
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ML Model'))

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from threadpoolctl import threadpool_limits
from backend.engine.astronomy.workflow import make_pipeline
from backend.engine.astronomy.features import RAW

out = Path(__file__).resolve().parent
primary_path = ROOT / 'outputs/primary_rows.csv'
split_path = ROOT / 'outputs/split_assignments.csv'
primary = pd.read_csv(primary_path, low_memory=False)
split = pd.read_csv(split_path)
assert len(primary) == len(split)
for c in ['source', 'source_row', 'object_group', 'Object_Class']:
    assert primary[c].equals(split[c]), c
train = split.partition.eq('development')
test = split.partition.eq('test')
assert not set(split.loc[train, 'object_group']) & set(split.loc[test, 'object_group'])
X = primary.reindex(columns=RAW + ['Redshift'])
y = primary.Object_Class.eq('STAR').astype(int)
configs = [
    ('Baseline', None, 'lbfgs'),
    ('Lasso (L1)', 'l1', 'liblinear'),
    ('Ridge (L2)', 'l2', 'lbfgs'),
]
results = []
for name, penalty, solver in configs:
    start = time.monotonic()
    model = make_pipeline('Logistic')
    model.set_params(model=LogisticRegression(
        penalty=penalty, C=0.5, solver=solver, max_iter=2500,
        class_weight='balanced', random_state=42, tol=1e-4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with threadpool_limits(limits=4):
            model.fit(X.loc[train], y.loc[train])
            probability = model.predict_proba(X.loc[test])[:, 1]
    predicted = (probability >= 0.5).astype(int)
    row = dict(
        model=name, penalty=penalty, solver=solver,
        C=None if penalty is None else 0.5,
        accuracy=float(accuracy_score(y.loc[test], predicted)),
        f1_macro=float(f1_score(y.loc[test], predicted, average='macro')),
        f1_weighted=float(f1_score(y.loc[test], predicted, average='weighted')),
        f1_star=float(f1_score(y.loc[test], predicted)),
        roc_auc=float(roc_auc_score(y.loc[test], probability)),
        misclassified=int(np.sum(predicted != y.loc[test].to_numpy())),
        fit_and_predict_seconds=time.monotonic() - start,
        n_iter=model.named_steps['model'].n_iter_.tolist(),
        convergence_warnings=[str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)],
        nonzero_coefficients=int(np.count_nonzero(model.named_steps['model'].coef_)),
        total_coefficients=int(model.named_steps['model'].coef_.size),
    )
    results.append(row)
    pd.DataFrame(results).to_csv(out / 'regularization_test_metrics.csv', index=False)
    print(json.dumps(row), flush=True)

payload = dict(
    description='Post-hoc fixed-split logistic regularization comparison for presentation; original champion unchanged.',
    baseline_definition='Unregularized logistic regression',
    training_rows=int(train.sum()), test_rows=int(test.sum()),
    positive_class='STAR', threshold=0.5,
    preprocessing='Original AstronomyFeatures (both), fixed physical masks, development-fitted median imputation and RobustScaler; no anomaly flags.',
    selection='No hyperparameter or threshold selection; C=0.5 inherited from original Logistic specification.',
    limitations='Post-hoc descriptive test comparison. L1 uses liblinear; unregularized and L2 use lbfgs. Test was already opened for the original champion evaluation.',
    primary_sha256=hashlib.sha256(primary_path.read_bytes()).hexdigest(),
    split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
    results=results,
)
(out / 'regularization_test_metrics.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
