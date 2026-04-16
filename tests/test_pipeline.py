import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.train_pipeline import run_training_pipeline


def test_pipeline_runs():
    result = run_training_pipeline()

    assert result is not None
    assert "isolation_forest" in result
    assert "zscore" in result
    assert "lstm" in result

    assert result["isolation_forest"]["accuracy"] > 0
    assert result["zscore"]["accuracy"] > 0
    assert result["lstm"]["accuracy"] > 0