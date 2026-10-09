# Deep Sky Divide — Astromanthan 2026

**Result:** HistGB_deeper, selected by mean five-fold development macro-F1. Final held-out macro-F1 **0.9946**, accuracy **0.9948**, ROC-AUC **0.9978**, STAR-positive PR-AUC (average precision) **0.9934**. No test-based model, feature or threshold selection. **Cross-catalogue transfer is poor:** the separate five-band model scores only 0.4780 macro-F1 on Dataset24; the main score does not establish universal generalization.

## Data and evaluation
Dataset29 provides 100,050 rows and 46 columns; Dataset24 provides 99,999 and 12. Both include STAR/GALAXY/QSO; exclude QSO for this binary task. The schemas and populations differ. Use Dataset29 for morphology and nonoverlapping Dataset24 for a five-band transfer diagnostic. Rounded Dataset29 IDs cannot support reliable joins. Conservative linkage uses 1-arcsecond coordinates or any three exact valid bands, plus conservative repeated-ID grouping within Dataset24 (IDs are quantized). Detected overlap: **2,734 Dataset24 rows**. Quarantine 1,909 conflicting-label groups; collapse 197 repeated primary rows; exclude unidentifiable records. Retain **89,251 primary sources**, 71,401 development / 17,850 test, and 62,128 secondary sources. Group intersections are zero under these matching rules. Unknown residual duplicates remain possible.

20% stratified group holdout, seed 42; five stratified group folds on remaining 80%. Fit cleaning-derived feature transforms, median imputers, robust scaling, selectors and anomaly detectors inside each training fold. Fixed physical bounds do not learn from data. Class weights are balanced; probability threshold stays 0.5. All ablations and robustness choices precede final test. CV is selection evidence, not an unbiased final estimate.

## Cleaning, leakage and features
All columns are classified in `02_feature_decisions.csv`. Exclude IDs, observation bookkeeping, coordinates, arbitrary notes and spectroscopy/derived labels. Convert invalid tokens/nonfinite values to missing; use broad documented magnitude/radius/axis/flux bounds. Mask 407 impossible numeric cells in development and impute within folds. Do not blanket-delete -1, 99, negative flux or unusual sources. Bounds assume SDSS-like units because no dictionary was supplied.

Keep 30 measured photometric/structural inputs. Add five colours, signed asinh fluxes, log radii, radius/half-light ratios, and PSF-minus-supplied-band differences. Generic magnitudes are not documented as cmodel; differences are empirical morphology proxies. R90 is absent, so no conventional R90/R50 concentration is claimed. Correlations above |r|=0.95 are logged; correlated predictors can share importance. Redshift ablation macro-F1 gap: **+0.0026** on the same development holdout. Redshift remains excluded regardless of score because final inputs must be photometric. Single-feature AUC>0.99 flags are saved, not automatically interpreted as leakage when structurally meaningful.

Two anomaly methods: robust IQR-scaled extremes and IsolationForest (2% training contamination). Keep/flag unusual observations; never delete by anomaly score or class. Compare HistGB with and without flags across the same five folds. Invalid measurements are masked in both. Outlier counts/examples and the impact are saved.

## Model comparison
| Candidate | CV macro-F1, mean ± SD | CV ROC-AUC, mean ± SD |
|---|---:|---:|
| HistGB_deeper | 0.9946 ± 0.0011 | 0.9979 ± 0.0006 |
| HistGB | 0.9946 ± 0.0010 | 0.9979 ± 0.0007 |
| HistGB_flags | 0.9946 ± 0.0011 | 0.9979 ± 0.0007 |
| HistGB_small | 0.9946 ± 0.0012 | 0.9980 ± 0.0006 |
| RandomForest | 0.9945 ± 0.0011 | 0.9979 ± 0.0006 |
| Logistic | 0.9918 ± 0.0008 | 0.9956 ± 0.0009 |
| Dummy | 0.3691 ± 0.0000 | 0.5000 ± 0.0000 |

Boosting variants are nearly tied; their tiny CV differences do not establish statistical superiority. Efficient predefined settings plus at most two bounded boosting variants replace a larger randomized search. HistGradientBoosting is used because external boosters/AutoGluon are absent. The existing engine supplies preprocessing, metrics, champion selection, persistence and UnifiedPredictor; historical employee/house models are not astronomy transfer models.

## Interpretation and robustness
Test GALAXY precision/recall/F1: 0.9968/0.9943/0.9955; STAR: 0.9919/0.9955/0.9937. Brier **0.0048**; reliability plot reports uncalibrated model probabilities. Confusion matrix, error rows and data-quality subgroups are provided.

Top raw-input permutation features: **Z_Band_Magnitude, R_Band_Magnitude, PSF_Magnitude_Z, I_Band_Magnitude, PSF_Magnitude_G**. Sizes/profile differences can separate extended galaxies from point-like stars; colours reflect spectral-energy distributions. These are predictive associations, not causal explanations. Native RF importance is labeled as a baseline's importance. Permutation importance is computed on development and, only after freezing, a 2,500-row test sample. Test importance never selects top-k; the top-15 ablation fits ANOVA on its training fold. SHAP is omitted for time.

Development macro-F1: photometric **0.9941**, structural **0.9952**, both (fixed HistGB) **0.9952**; top-15 **0.9941**. Winner with 0.02-mag Gaussian perturbations **0.9954**; seed123 split **0.9947**; disjoint 20-degree sky cells **0.9937**. These are individual development stress tests, not five-fold estimates. Shared-band transfer macro-F1: Dataset29 test **0.9683**, nonoverlapping Dataset24 **0.4780**; it uses a separate five-band model, not the structural champion.

## Five findings and limitations
1. Structural and photometric coverage makes Dataset29 the primary sample; Dataset24 supports limited-feature transfer only.
2. Catalogues overlap and Dataset29 IDs have lost precision; object linkage is necessary before splitting.
3. Corrupted cells require validity masking; unusual valid sources should not be blindly removed.
4. CV selects HistGB_deeper; final holdout confirms the measured score above without post-test tuning.
5. Five-band transfer falls from 0.9683 to 0.4780 macro-F1: a major generalization limitation. Dataset24 is substantially fainter: median g=21.12 versus 17.42 in development (all binary Dataset24 rows; domain_shift.csv). No tuning follows this result.

Limitations: inferred units, label provenance unverified, imperfect crossmatching, no seeing/extinction/error model, no independent telescope survey, no QSO rejection, and balanced weights can affect probability calibration. More time: recover precise source IDs/data dictionary, validate spectroscopy-based ground truth, add uncertainty/seeing features, nested spatial CV, and calibration fitted on development folds. Full per-fold precision/recall/F1/ROC/PR metrics and assumptions are in the companion files.
