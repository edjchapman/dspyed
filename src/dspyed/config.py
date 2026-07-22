"""Central configuration (pydantic-settings, env prefix ``DSPYED_``).

Model ids are LiteLLM strings for the two-tier pattern the experiment matrix
uses everywhere: SMALL is the cheap workhorse and optimization target, LARGE
is the comparison bar (and GEPA's reflection LM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DSPYED_", env_file="../../.env", extra="ignore")

    small_model: str = "anthropic/claude-haiku-4-5-20251001"
    large_model: str = "anthropic/claude-sonnet-5"

    data_root: Path = Path("data/spider")
    splits_dir: Path = Path("data/splits")
    artifacts_dir: Path = Path("artifacts")
    results_dir: Path = Path("experiments/results")

    max_tokens_generate: int = 600
    max_tokens_cot: int = 1_000
    exec_timeout_s: float = 5.0
    exec_max_rows: int = 10_000

    budget_cap_usd: float = 150.0  # the plan's hard cap; compile refuses past it


# USD per million tokens (prompt, completion) — verified against litellm's
# model_cost map at the C1 checkpoint (2026-07-22). Cost numbers in results
# JSONs cite this table, so a price change is a visible diff, not silent drift.
PRICES_PER_MTOK: Final[dict[str, tuple[float, float]]] = {
    "anthropic/claude-haiku-4-5-20251001": (1.0, 5.0),
    "anthropic/claude-sonnet-5": (2.0, 10.0),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_price, completion_price = PRICES_PER_MTOK.get(model, (0.0, 0.0))
    return (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1_000_000
