"""Strategy configuration for v9 hybrid solver.

Strategies control how the solver balances deterministic baseline
vs. LLM tree search iterations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Strategy:
    name: str
    description: str
    max_iterations: int  # LLM tree search iterations after baseline
    refine_steps: int    # persistent session refinement steps


STRATEGIES: dict[str, Strategy] = {
    "balanced": Strategy(
        name="balanced",
        description="Balanced: baseline + moderate tree search + refinement",
        max_iterations=10,
        refine_steps=5,
    ),
    "explore_heavy": Strategy(
        name="explore_heavy",
        description="Heavy exploration: baseline + many tree search iterations",
        max_iterations=15,
        refine_steps=3,
    ),
    "refine_heavy": Strategy(
        name="refine_heavy",
        description="Heavy refinement: baseline + few tree search + deep refinement",
        max_iterations=6,
        refine_steps=8,
    ),
}

DEFAULT_STRATEGY = "balanced"


def get_strategy(name: str) -> Strategy:
    return STRATEGIES.get(name, STRATEGIES[DEFAULT_STRATEGY])


def all_strategy_names() -> list[str]:
    return list(STRATEGIES.keys())
