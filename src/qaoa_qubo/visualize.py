from __future__ import annotations

from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .result import QAOAResult
from .qubo import QUBOProblem
from .hamiltonian import qubo_to_ising
from .qaoa import QAOASolver


def plot_optimization_history(
    history: Sequence[float],
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot the objective value vs optimization iteration.

    Parameters
    ----------
    history : sequence of float
        Objective values recorded during optimization.
    ax : optional matplotlib Axes
        If provided, draw on this axes; otherwise create a new figure.
    title : optional str
        Plot title.

    Returns
    -------
    ax : matplotlib Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(range(len(history)), history, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    if title:
        ax.set_title(title)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    return ax


def plot_result_history(result: QAOAResult) -> None:
    """
    Convenience wrapper: plot the optimization history from a QAOAResult.
    """
    plot_optimization_history(result.history, title="QAOA Optimization History")
    plt.show()


def energy_landscape_p1(
    qubo: QUBOProblem,
    solver: QAOASolver,
    gamma_range: Tuple[float, float] = (0.0, np.pi),
    beta_range: Tuple[float, float] = (0.0, np.pi),
    num_points: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a coarse energy landscape E(gamma, beta) for p=1 QAOA.

    Uses the solver's internal circuit builder + statevector expectation.
    This is purely a visualization / debugging tool:

    - Assumes solver.p == 1
    - Uses a statevector-based expectation regardless of solver.mode
    - Does not touch hardware or Runtime samplers

    Returns
    -------
    G : np.ndarray
        2D array of gamma values (meshgrid).
    B : np.ndarray
        2D array of beta values (meshgrid).
    E : np.ndarray
        2D array of energy values at each (gamma, beta).
    """
    if solver.p != 1:
        raise ValueError("energy_landscape_p1 assumes p=1 QAOA (solver.p == 1).")

    h, J, const_shift = qubo_to_ising(qubo)
    n = qubo.num_variables
    if n is None:
        raise ValueError("QUBOProblem.num_variables is not set")

    gammas = np.linspace(gamma_range[0], gamma_range[1], num_points)
    betas = np.linspace(beta_range[0], beta_range[1], num_points)

    G, B = np.meshgrid(gammas, betas, indexing="ij")
    E = np.zeros_like(G, dtype=float)

    # For each (gamma, beta) pair, build a QAOA circuit and evaluate
    for i in range(num_points):
        for j in range(num_points):
            gamma = G[i, j]
            beta = B[i, j]

            # Reuse the solver's internal circuit builder
            qc = solver._build_qaoa_circuit(
                num_qubits=n,
                h=h,
                J=J,
                gammas=[gamma],
                betas=[beta],
            )

            # Always use statevector expectation for the landscape
            val = solver._expectation_statevector(qc, qubo) + const_shift
            E[i, j] = val

    return G, B, E


def plot_energy_landscape(G: np.ndarray, B: np.ndarray, E: np.ndarray) -> None:
    """
    Simple heatmap of the energy landscape for p=1.

    Parameters
    ----------
    G, B : np.ndarray
        Meshgrids of gamma and beta.
    E : np.ndarray
        Energy values at each (G[i,j], B[i,j]).
    """
    fig, ax = plt.subplots()
    c = ax.pcolormesh(G, B, E, shading="auto")
    fig.colorbar(c, ax=ax, label="Energy")
    ax.set_xlabel("gamma")
    ax.set_ylabel("beta")
    ax.set_title("QAOA Energy Landscape (p=1)")
    plt.show()
