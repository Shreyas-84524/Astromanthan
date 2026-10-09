"""Optional backend failures must preserve successful baseline candidates."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from backend.engine.backends.autogluon_backend import AutoGluonBackend
from backend.engine.training.m4_orchestrator import M4PipelineRunner


@pytest.mark.parametrize('failure_mode', ['missing_parquet', 'worker_failure'])
def test_baselines_survive_optional_backend_failure(monkeypatch, tmp_path, failure_mode):
    def fail(*args, **kwargs):
        if failure_mode == 'missing_parquet':
            raise ImportError('pyarrow is required for parquet support')
        return SimpleNamespace(status=SimpleNamespace(value='failed'), error_message='AutoGluon is not installed')
    monkeypatch.setattr(AutoGluonBackend, 'fit', fail)
    rng = np.random.default_rng(42)
    x = rng.normal(size=120)
    df = pd.DataFrame({'magnitude': x, 'target': np.where(x > 0, 'STAR', 'GALAXY')})
    result = M4PipelineRunner.run(df, 'target', tmp_path/'model', stage_dir=tmp_path/'stage')
    assert result['manifest'].backend_name == 'baseline'
    assert result['backend_warnings']
    assert 'baseline models only' in result['manifest'].selection_rationale
    assert len(result['predictor'].predict(df[['magnitude']].head(3))) == 3
    assert result['partition_info'].test_rows == 24


def test_no_successful_candidate_remains_an_error(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise ImportError('AutoGluon missing')
    monkeypatch.setattr(AutoGluonBackend, 'fit', fail)
    df = pd.DataFrame({'x': np.arange(100), 'target': ['STAR', 'GALAXY'] * 50})
    with pytest.raises(RuntimeError, match='No usable model candidates'):
        M4PipelineRunner.run(df, 'target', tmp_path/'model', include_baselines=False, stage_dir=tmp_path/'stage')
