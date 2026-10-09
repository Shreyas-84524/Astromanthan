"""Deterministic survey cleaning and fold-fitted anomaly handling."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

BANDS = list('UGRIZ')
MAG = [f'{b}_Band_Magnitude' for b in BANDS]
PSF = [f'PSF_Magnitude_{b}' for b in BANDS]
RADIUS = [f'Petrosian_Radius_{b}' for b in BANDS]
R50 = [f'Petrosian_50_Percent_Radius_{b}' for b in BANDS]
FLUX = [f'Petrosian_Flux_{b}' for b in BANDS]
AXIS = [f'Exponential_Axis_Ratio_{b}' for b in BANDS]
RAW = MAG + PSF + RADIUS + R50 + FLUX + AXIS


def clean_numeric(frame):
    """Fixed, deliberately broad rules, independent of fitted sample statistics.

    Negative flux is permitted (background-subtracted observations). Negative
    magnitudes are permitted. A bare -1 or 99 is NOT globally treated as missing.
    Units are inferred from SDSS-like names; no organizer dictionary was supplied.
    """
    out = frame.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    for c in out:
        v = out[c]
        if c in MAG + PSF:
            out[c] = v.where(v.between(-5, 40))
        elif c in RADIUS + R50:
            out[c] = v.where(v.between(0, 1000) & (v > 0))
        elif c in AXIS:
            out[c] = v.where(v.between(0, 1) & (v > 0))
        elif c in FLUX:
            out[c] = v.where(v.abs() < 1e7)
        elif c == 'Redshift':
            out[c] = v.where(v.between(-0.1, 10))
    return out


class AstronomyFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, mode='both', include_redshift=False):
        self.mode = mode
        self.include_redshift = include_redshift

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = np.asarray(self.transform(X.iloc[:1]).columns, dtype=object)
        return self

    def transform(self, X):
        cols = RAW + (['Redshift'] if self.include_redshift else [])
        d = clean_numeric(X.reindex(columns=cols))
        photo = d[MAG + PSF + FLUX].copy()
        for a, b in zip(BANDS[:-1], BANDS[1:]):
            photo[f'colour_{a}_{b}'] = d[f'{a}_Band_Magnitude'] - d[f'{b}_Band_Magnitude']
        photo['colour_U_Z'] = d[MAG[0]] - d[MAG[-1]]
        for c in FLUX:
            photo[f'asinh_{c}'] = np.arcsinh(d[c])
        structure = d[RADIUS + R50 + AXIS].copy()
        for b in BANDS:
            # The generic band magnitude has no documented model/cmodel definition.
            # This is an empirical PSF-minus-band difference, not a claimed cmodel cut.
            structure[f'psf_minus_band_{b}'] = d[f'PSF_Magnitude_{b}'] - d[f'{b}_Band_Magnitude']
            structure[f'radius_over_half_light_{b}'] = d[f'Petrosian_Radius_{b}'] / d[f'Petrosian_50_Percent_Radius_{b}']
        for c in RADIUS + R50:
            structure[f'log1p_{c}'] = np.log1p(d[c])
        if self.mode == 'photometric':
            result = photo
        elif self.mode == 'structural':
            result = structure
        elif self.mode == 'bands':
            result = photo[MAG + [c for c in photo if c.startswith('colour_')]]
        else:
            result = pd.concat([photo, structure], axis=1)
        if self.include_redshift:
            result['Redshift'] = d['Redshift']
        return result.replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class AnomalyFlags(TransformerMixin, BaseEstimator):
    """Keep observations; append robust-z and IsolationForest anomaly indicators.

    Every learned statistic and detector is fitted only inside the training fold.
    """
    def __init__(self, enabled=False, random_state=42):
        self.enabled = enabled
        self.random_state = random_state

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        if self.enabled:
            self.imputer_ = SimpleImputer(strategy='median', keep_empty_features=True)
            a = self.imputer_.fit_transform(X)
            self.scaler_ = RobustScaler().fit(a)
            a = self.scaler_.transform(a)
            self.detector_ = IsolationForest(n_estimators=80, max_samples=min(2048,len(a)),
                contamination=0.02, random_state=self.random_state, n_jobs=4).fit(a)
        return self

    def transform(self, X):
        if not self.enabled:
            return X.copy()
        a = self.scaler_.transform(self.imputer_.transform(X))
        out = X.copy()
        # RobustScaler uses IQR. 4.45 IQR is approximately 6 standard deviations
        # for Gaussian data; the threshold is fixed before evaluation.
        out['robust_extreme_flag'] = (np.abs(a).max(axis=1) > 4.45).astype(float)
        out['isolation_anomaly_flag'] = (self.detector_.predict(a) == -1).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        cols = list(self.feature_names_in_)
        return np.asarray(cols + (['robust_extreme_flag', 'isolation_anomaly_flag'] if self.enabled else []), dtype=object)
