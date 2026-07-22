"""The measurement instruments: safe SQL execution + result-set comparison.

Everything in this package is held to strict pyright — a type hole here is a
wrong headline number. No DSPy imports allowed: the engine must be testable
offline with zero LLM involvement.
"""

from dspyed.engine.compare import ComparisonResult, compare_results
from dspyed.engine.executor import ExecResult, SafeExecutor

__all__ = ["ComparisonResult", "ExecResult", "SafeExecutor", "compare_results"]
