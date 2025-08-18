"""
gnn_testing_framework.py
------------------------
Lightweight testing helpers for GNN experiments.

This module provides a small, dependency-light `GNNTester` class so
other scripts can import it during experiments and CI. It intentionally
keeps behaviour minimal — it's a thin wrapper around a predictor
object that exposes an `evaluate` method if available.

The goal is to remove import-time failures for `run_gnn_baselines.py`.
"""
from typing import Any, Dict, Iterable, List, Optional


class GNNTester:
    """Minimal testing helper for GNN predictors.

    Usage:
      tester = GNNTester(predictor)
      metrics = tester.evaluate(samples)

    The class does not enforce any particular predictor API, but will
    attempt to call common methods (`model`, `device`, `graph_data`,
    `predict`) if present. If the predictor does not implement a
    prediction method, evaluation will raise a RuntimeError.
    """

    def __init__(self, predictor: Optional[Any] = None):
        self.predictor = predictor

    def evaluate(self, samples: Iterable[Dict]) -> Dict[str, float]:
        """Evaluate predictor on an iterable of samples.

        samples should be an iterable of dicts that include at least the
        keys 'source_idx', 'target_idx' and 'flow' when using a
        predictor with a `predict` method. For predictors that expose
        a `model` and accept tensors, this helper tries to call
        predictor.predict(...) and falls back to raising a clear error.
        """

        if self.predictor is None:
            raise RuntimeError("No predictor attached to GNNTester")

        # Prefer a dedicated `predict` method if available
        if hasattr(self.predictor, 'predict'):
            return self.predictor.predict(list(samples))

        # If the predictor exposes `model` and `device`, attempt a
        # simple batching of samples and run inference. This is a best
        # effort path and may not work for every predictor implementation.
        if hasattr(self.predictor, 'model') and hasattr(self.predictor, 'device'):
            model = getattr(self.predictor, 'model')
            device = getattr(self.predictor, 'device')

            # Try using a provided helper `forward_samples` if present
            if hasattr(self.predictor, 'forward_samples'):
                return self.predictor.forward_samples(list(samples))

        raise RuntimeError('Predictor does not expose a compatible predict API')


__all__ = ['GNNTester']
