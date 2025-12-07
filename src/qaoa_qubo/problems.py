from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import networkx as nx

from .qubo import QUBOProblem


@dataclass
class MaxCutProblem:
    """
    Simple MaxCut problem wrapper around a NetworkX graph.
    """

    graph: nx.Graph

    @classmethod
    def from_networkx(cls, graph: nx.Graph) -> "MaxCutProblem":
        return cls(graph=graph)

    def to_qubo(self) -> QUBOProblem:
        """
        Convert the MaxCut instance into a QUBOProblem.

        For each edge (i, j), we add:

            x_i + x_j - 2 x_i x_j

        to the objective.
        """
        linear: Dict[int, float] = {}
        quadratic: Dict[Tuple[int, int], float] = {}

        for (i, j) in self.graph.edges():
            # Linear contributions: x_i + x_j
            linear[i] = linear.get(i, 0.0) + 1.0
            linear[j] = linear.get(j, 0.0) + 1.0

            # Quadratic term: -2 x_i x_j
            key = (min(i, j), max(i, j))
            quadratic[key] = quadratic.get(key, 0.0) - 2.0

        num_variables = len(self.graph.nodes)

        return QUBOProblem.from_dicts(
            linear=linear,
            quadratic=quadratic,
            constant=0.0,
            num_variables=num_variables,
        )
