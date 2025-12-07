from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, Sequence, Optional

import numpy as np


@dataclass
class QUBOProblem:
    """
    Represents a QUBO of the form:

        C(x) = sum_i a_i x_i + sum_{i<j} b_ij x_i x_j + constant

    where x_i ∈ {0, 1}.

    Attributes
    ----------
    linear : Dict[int, float]
        Coefficients a_i for the linear terms x_i.
    quadratic : Dict[Tuple[int, int], float]
        Coefficients b_ij for the quadratic terms x_i x_j, with i < j.
    constant : float
        Constant offset term.
    num_variables : int
        Number of binary variables (x_0, ..., x_{n-1}).
    """

    linear: Dict[int, float]
    quadratic: Dict[Tuple[int, int], float]
    constant: float = 0.0
    num_variables: Optional[int] = None

    def __post_init__(self) -> None:
        """
        Normalize keys and infer num_variables if not provided.
        """
        # Normalize quadratic keys: ensure i < j and combine duplicates.
        normalized_quadratic: Dict[Tuple[int, int], float] = {}
        for (i, j), coeff in self.quadratic.items():
            if i == j:
                # x_i^2 = x_i for binary variables; fold into linear term
                self.linear[i] = self.linear.get(i, 0.0) + coeff
                continue

            if j < i:
                i, j = j, i  # enforce i < j

            key = (i, j)
            normalized_quadratic[key] = normalized_quadratic.get(key, 0.0) + coeff

        self.quadratic = normalized_quadratic

        # Infer num_variables if not provided
        if self.num_variables is None:
            indices = set(self.linear.keys())
            for i, j in self.quadratic.keys():
                indices.add(i)
                indices.add(j)

            self.num_variables = 0 if not indices else (max(indices) + 1)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dicts(
        cls,
        linear: Dict[int, float],
        quadratic: Dict[Tuple[int, int], float],
        constant: float = 0.0,
        num_variables: Optional[int] = None,
    ) -> "QUBOProblem":
        """
        Convenience constructor from linear and quadratic coefficient dicts.
        """
        # Make copies so we don't mutate caller's data
        return cls(
            linear=dict(linear),
            quadratic=dict(quadratic),
            constant=constant,
            num_variables=num_variables,
        )

    @classmethod
    def from_matrix(cls, Q: np.ndarray, constant: float = 0.0) -> "QUBOProblem":
        """
        Construct a QUBO from a full (n x n) Q matrix where:

            C(x) = x^T Q x + constant

        We interpret the diagonal as linear terms and the upper triangle
        (i < j) as quadratic terms.
        """
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be a square 2D matrix")

        n = Q.shape[0]
        linear: Dict[int, float] = {}
        quadratic: Dict[Tuple[int, int], float] = {}

        # Diagonal → linear
        for i in range(n):
            coeff = float(Q[i, i])
            if coeff != 0.0:
                linear[i] = coeff

        # Upper triangle → quadratic
        for i in range(n):
            for j in range(i + 1, n):
                coeff = float(Q[i, j] + Q[j, i])  # symmetrize just in case
                if coeff != 0.0:
                    quadratic[(i, j)] = coeff

        return cls(linear=linear, quadratic=quadratic, constant=constant, num_variables=n)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, bitstring: Sequence[int] | str) -> float:
        """
        Evaluate the QUBO cost C(x) for a given assignment x.

        Parameters
        ----------
        bitstring : Sequence[int] | str
            Either:
            - a string of '0'/'1' characters, e.g. "0101", or
            - a sequence of 0/1 integers, e.g. [0, 1, 0, 1].

        Returns
        -------
        float
            The cost C(x) for this bitstring.
        """
        x = self._bitstring_to_array(bitstring)

        if self.num_variables is None:
            raise ValueError("num_variables is not set")

        if len(x) != self.num_variables:
            raise ValueError(
                f"Expected bitstring of length {self.num_variables}, got {len(x)}"
            )

        value = self.constant

        # Linear terms
        for i, coeff in self.linear.items():
            value += coeff * x[i]

        # Quadratic terms
        for (i, j), coeff in self.quadratic.items():
            value += coeff * x[i] * x[j]

        return float(value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _bitstring_to_array(bitstring: Sequence[int] | str) -> np.ndarray:
        """
        Convert a bitstring (str or sequence) into a NumPy array of 0/1 integers.
        """
        if isinstance(bitstring, str):
            return np.array([int(b) for b in bitstring], dtype=int)
        else:
            return np.array(bitstring, dtype=int)
