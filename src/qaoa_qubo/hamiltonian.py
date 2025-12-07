from __future__ import annotations

from typing import Dict, Tuple

from .qubo import QUBOProblem


def qubo_to_ising(
    qubo: QUBOProblem,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    """
    Convert a QUBO of the form:

        C(x) = sum_i a_i x_i + sum_{i<j} b_ij x_i x_j + c

    with x_i ∈ {0, 1} into an Ising Hamiltonian of the form:

        H(z) = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + const

    with z_i ∈ {-1, +1}, using the mapping x_i = (1 - z_i) / 2.

    Parameters
    ----------
    qubo : QUBOProblem
        The QUBO instance to convert.

    Returns
    -------
    h : Dict[int, float]
        Coefficients h_i for single-spin terms z_i.
    J : Dict[Tuple[int, int], float]
        Coefficients J_ij for pairwise terms z_i z_j with i < j.
    constant_shift : float
        The additive constant term. This does not affect which configuration
        minimizes the energy, but is included for completeness.
    """
    h: Dict[int, float] = {}
    J: Dict[Tuple[int, int], float] = {}
    constant_shift = qubo.constant

    # Linear terms: a_i x_i
    # x_i = (1 - z_i)/2 → a_i x_i = a_i/2 - (a_i/2) z_i
    for i, a_i in qubo.linear.items():
        # z_i coefficient
        h[i] = h.get(i, 0.0) - 0.5 * a_i
        # constant term
        constant_shift += 0.5 * a_i

    # Quadratic terms: b_ij x_i x_j
    # x_i x_j = (1 - z_i - z_j + z_i z_j) / 4
    # → b_ij x_i x_j = b_ij/4
    #                  - (b_ij/4) z_i
    #                  - (b_ij/4) z_j
    #                  + (b_ij/4) z_i z_j
    for (i, j), b_ij in qubo.quadratic.items():
        # enforce i < j
        if j < i:
            i, j = j, i

        key = (i, j)

        # Pair term J_ij
        J[key] = J.get(key, 0.0) + 0.25 * b_ij

        # Single-spin contributions
        h[i] = h.get(i, 0.0) - 0.25 * b_ij
        h[j] = h.get(j, 0.0) - 0.25 * b_ij

        # Constant term
        constant_shift += 0.25 * b_ij

    return h, J, float(constant_shift)
