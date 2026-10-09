# Assumptions and decisions fixed before final test

1. The user-supplied files are authoritative. Their hashes match the previously
   extracted Dataset29 and Dataset24 copies. No public substitute is used.
2. Labels are uppercased and stripped. STAR/GALAXY are the binary target; QSO is
   excluded, not relabeled as a star. Ground-truth derivation is undocumented.
3. Column names suggest SDSS-like magnitudes, angular radii and fluxes; no data
   dictionary confirms units or the generic magnitude's model definition.
4. Two catalogues have different schemas and partial identity overlap. Dataset29
   supplies the primary structural task. Dataset24 is used only for a distinct
   five-band transfer model. No structural values are borrowed across files.
5. Dataset29 IDs are severely rounded and cannot identify objects. Matching uses
   1-arcsecond positions or at least three matching valid bands at five decimals.
   These links are transitive and ignore labels. Dataset24 ID strings also group
   repeated entries conservatively; repeated strings need not prove identical
   sources. This sacrifices coverage when identity is ambiguous. In Dataset24,
   all 99,999 IDs are divisible by 256 and only 78,052 ID strings are unique,
   suggesting earlier quantization. Repeated IDs are conservative groups, not
   verified physical identities; do not treat them as a reliable cross-file key.
6. Exclude sources lacking both valid coordinates and three usable bands; remove
   whole groups with conflicting binary labels. Pick the most complete primary
   row per remaining group, breaking ties by original row position. Report this
   eligibility restriction: performance applies to retained identifiable sources.
7. Fixed plausibility bounds: magnitude [-5,40], radius (0,1000], axis ratio (0,1],
   absolute flux <10,000,000, redshift [-0.1,10] for diagnostics. Broad bounds
   reject obvious corruptions, not rigorously validated quality cuts. No global
   -1/99 sentinel rule; negative magnitudes and negative measured flux may be valid.
8. Missing, nonnumeric and invalid cells become NaN. Fold-trained median imputation
   is the baseline; logistic regression uses fold-trained robust scaling. No
   global median, quantile clipping or fitted full-dataset preprocessing is used.
9. Pure IDs, observation metadata, sky position and undocumented notes are not
   predictors. Redshift and derived spectral/population columns are excluded.
   A high single-feature AUC flags a possible proxy, not automatic proof of leakage.
10. Five-fold StratifiedGroupKFold, seed42, supplies both the approximately 20%
    final holdout and development CV. Identities never cross a split under the
    matching rules. Residual unidentified repeats are possible.
11. Macro-F1 is the selection criterion; balanced class weights address the
    moderate imbalance. Threshold stays 0.5. The original engine's validation
    candidate contract stores mean CV scores, with partition metadata explicitly
    documenting this CV adaptation rather than a separate validation partition.
12. A small predefined grid replaces RandomizedSearchCV to bound resource use.
    Comparisons run sequentially with four threads. AutoGluon and external
    boosters are unavailable; sklearn HistGradientBoosting supplies boosting.
13. Robust-IQR extremes (max absolute scaled feature >4.45) and IsolationForest
    (2% training contamination) are flags, not deletion rules. Their full models
    fit only training folds. The no-flag/flag boosting variants share folds.
14. PSF-minus-band differences are empirical morphology proxies. Radius divided
    by half-light radius is not standard R90/R50 concentration. asinh transforms
    preserve negative flux; log1p applies only to positive valid radii.
15. Development robustness uses the first CV holdout unless otherwise labeled.
    Top-15 ANOVA is fitted inside its training fold. Alternative seed123 and a
    disjoint 20-degree RA/Dec grid are stress tests. Gaussian magnitude noise of
    0.02 mag is illustrative and not a measured survey error model.
16. Freeze selection.json before opening final test. Test feature importance is
    descriptive, sampled and never used for model selection. Native feature
    importance comes from the random-forest baseline, labeled accordingly.
17. Final test and the predeclared five-band transfer diagnostic do not trigger
    tuning. Probabilities are uncalibrated estimates; assess reliability/Brier
    without fitting a calibrator on the test set.
18. Reuse the existing engine's preprocessing, metrics, selection, persistence and
    inference. Employee-turnover/house-price weights are inappropriate for this
    task, so train astronomy-specific estimators within the existing platform.
19. No claim of QSO rejection, causal feature effects, rare-object discovery,
    independent-survey generalization, or production astronomy readiness is made.

Scientific reference: [SDSS morphology/classification](https://classic.sdss.org/dr6/algorithms/classify.php)
describes PSF-versus-cmodel separation. This catalogue's unspecified generic
magnitude is not assumed to be cmodel.
