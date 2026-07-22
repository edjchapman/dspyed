"""DSPy programs: signatures, the TextToSQL module, and the baseline ladder.

The only package (besides optim/) allowed to import dspy. Everything here
returns predictions whose ``sql`` field is already cleaned — downstream code
(metric, API) never parses model output.
"""

from dspyed.pipeline.baselines import build_program
from dspyed.pipeline.modules import Attempt, TextToSQL, clean_sql

__all__ = ["Attempt", "TextToSQL", "build_program", "clean_sql"]
