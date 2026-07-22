"""Evaluation: the execution-accuracy metric and (later) the run harness.

Strict pyright, no dspy import — the metric is a plain callable shaped like a
DSPy metric ``(example, pred, trace=None)`` so it can be unit-tested with
SimpleNamespace stand-ins and handed to optimizers unchanged.
"""

from dspyed.eval.metric import ExecutionAccuracy, MetricOutcome, gold_is_ordered

__all__ = ["ExecutionAccuracy", "MetricOutcome", "gold_is_ordered"]
