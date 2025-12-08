from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import networkx as nx

from .qubo import QUBOProblem


@dataclass
class MaxCutProblem:
    """
    Simple MaxCut problem defined by a dictionary of weighted edges.

    Edges are given as a dict mapping (i, j) -> weight, where i and j
    are integer node indices.
    """

    def __init__(self, edges: Dict[Tuple[int, int], float]) -> None:
        """
        Parameters
        ----------
        edges : dict
            Mapping (i, j) -> weight for each edge in the graph.
        """
        self.edges = edges

    def to_qubo(self) -> QUBOProblem:
        """
        Build a QUBOProblem corresponding to the MaxCut objective:

            Maximize  sum_{(i,j)} w_ij * (x_i XOR x_j)

        which is equivalent to minimizing a QUBO:

            C(x) = sum_i a_i x_i + sum_{i<j} b_ij x_i x_j + const
        """
        linear: Dict[int, float] = {}
        quadratic: Dict[Tuple[int, int], float] = {}
        constant = 0.0

        for (i, j), w in self.edges.items():
            # Normalize edge ordering
            if i == j:
                continue
            u, v = sorted((i, j))

            # Standard MaxCut QUBO encoding:
            # w * (x_i XOR x_j) = w * (x_i + x_j - 2 x_i x_j)
            linear[u] = linear.get(u, 0.0) + w
            linear[v] = linear.get(v, 0.0) + w
            key = (u, v)
            quadratic[key] = quadratic.get(key, 0.0) - 2.0 * w
            # constant term could be added, but it's irrelevant for argmin

        return QUBOProblem.from_dicts(linear, quadratic, constant=constant)
