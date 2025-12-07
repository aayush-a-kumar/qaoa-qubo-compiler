from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class QAOAResult:
    """
    Container for the result of a QAOA optimization run.
    """

    best_bitstring: str
    best_cost: float

    optimal_gammas: Sequence[float]
    optimal_betas: Sequence[float]
    optimal_value: float  # objective (expected cost) at optimal parameters

    history: List[float]  # objective values across iterations
