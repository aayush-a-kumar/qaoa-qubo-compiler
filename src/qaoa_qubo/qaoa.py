from __future__ import annotations

from typing import Optional, Sequence, Tuple, Dict

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from .qubo import QUBOProblem
from .problems import MaxCutProblem
from .hamiltonian import qubo_to_ising
from .result import QAOAResult


class QAOASolver:
    """
    A simple QAOA solver for QUBO-based problems.

    Usage
    -----
    >>> solver = QAOASolver(p=1, maxiter=100, seed=42)
    >>> result = solver.solve(problem)
    >>> print(result.best_bitstring, result.best_cost)
    """

    def __init__(
        self,
        p: int = 1,
        maxiter: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        if p <= 0:
            raise ValueError("p (number of QAOA layers) must be positive")

        self.p = p
        self.maxiter = maxiter
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self, problem: QUBOProblem | MaxCutProblem) -> QAOAResult:
        """
        Run QAOA on either a QUBOProblem or a MaxCutProblem.

        Parameters
        ----------
        problem : QUBOProblem | MaxCutProblem
            The problem instance to solve.

        Returns
        -------
        QAOAResult
            The optimization result including best bitstring and parameters.
        """
        # Normalize to a QUBOProblem
        if isinstance(problem, MaxCutProblem):
            qubo = problem.to_qubo()
        elif isinstance(problem, QUBOProblem):
            qubo = problem
        else:
            raise TypeError(
                f"Unsupported problem type: {type(problem)}. "
                "Expected QUBOProblem or MaxCutProblem."
            )

        # Convert to Ising coefficients for circuit construction
        h, J, const_shift = qubo_to_ising(qubo)
        num_qubits = qubo.num_variables
        if num_qubits is None:
            raise ValueError("QUBOProblem.num_variables is not set")

        # Objective history for debugging/analysis
        history: list[float] = []

        def objective(theta: np.ndarray) -> float:
            gammas, betas = self._split_params(theta)
            qc = self._build_qaoa_circuit(
                num_qubits=num_qubits,
                h=h,
                J=J,
                gammas=gammas,
                betas=betas,
            )
            exp_val = self._expectation_from_statevector(qc, qubo) + const_shift
            history.append(float(exp_val))
            return float(exp_val)

        # Random initialization in [0, 2π)
        theta0 = self.rng.uniform(0.0, 2.0 * np.pi, size=2 * self.p)

        opt_result = minimize(
            objective,
            theta0,
            method="COBYLA",
            options={"maxiter": self.maxiter, "disp": False},
        )

        theta_opt = opt_result.x
        gammas_opt, betas_opt = self._split_params(theta_opt)

        # Build final circuit and get the most likely bitstring
        qc_opt = self._build_qaoa_circuit(
            num_qubits=num_qubits,
            h=h,
            J=J,
            gammas=gammas_opt,
            betas=betas_opt,
        )
        best_bitstring = self._most_probable_bitstring(qc_opt, num_qubits)
        best_cost = float(qubo.evaluate(best_bitstring))

        optimal_value = float(objective(theta_opt))  # last evaluation

        return QAOAResult(
            best_bitstring=best_bitstring,
            best_cost=best_cost,
            optimal_gammas=list(gammas_opt),
            optimal_betas=list(betas_opt),
            optimal_value=optimal_value,
            history=history,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _split_params(self, theta: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the parameter vector into gammas and betas of length p each.
        """
        theta = np.asarray(theta, dtype=float)
        if theta.size != 2 * self.p:
            raise ValueError(
                f"Expected parameter vector of size {2 * self.p}, got {theta.size}"
            )
        gammas = theta[: self.p]
        betas = theta[self.p :]
        return gammas, betas

    def _build_qaoa_circuit(
        self,
        num_qubits: int,
        h: Dict[int, float],
        J: Dict[tuple[int, int], float],
        gammas: Sequence[float],
        betas: Sequence[float],
    ) -> QuantumCircuit:
        """
        Construct the QAOA circuit for given parameters.

        - Initial state: |+>^n via Hadamards.
        - For each layer k:
            * Apply cost unitary U_C(gamma_k) from Ising Hamiltonian.
            * Apply mixer unitary U_B(beta_k) from X mixer.
        """
        qc = QuantumCircuit(num_qubits)

        # Initial state |+>^n
        for q in range(num_qubits):
            qc.h(q)

        for layer in range(self.p):
            gamma = float(gammas[layer])
            beta = float(betas[layer])

            # Cost Hamiltonian e^{-i gamma H_C}
            # Single Z terms (RZ rotations)
            for i, h_i in h.items():
                if h_i != 0.0:
                    angle = 2.0 * gamma * h_i  # RZ(θ) = e^{-i θ/2 Z}
                    qc.rz(angle, i)

            # ZZ terms (RZZ rotations)
            for (i, j), J_ij in J.items():
                if J_ij != 0.0:
                    angle = 2.0 * gamma * J_ij  # RZZ(θ) = e^{-i θ/2 Z⊗Z}
                    qc.rzz(angle, i, j)

            # Mixer Hamiltonian e^{-i beta ∑ X_i}
            for q in range(num_qubits):
                qc.rx(2.0 * beta, q)  # RX(θ) = e^{-i θ/2 X}

        return qc

    def _expectation_from_statevector(
        self,
        circuit: QuantumCircuit,
        qubo: QUBOProblem,
    ) -> float:
        """
        Compute E[C(x)] using a statevector simulation.

        We compute probabilities over all bitstrings and then:

            E[C] = sum_x p(x) C(x)
        """
        sv = Statevector.from_instruction(circuit)
        probs = np.abs(sv.data) ** 2

        if qubo.num_variables is None:
            raise ValueError("QUBOProblem.num_variables is not set")

        n = qubo.num_variables
        exp_val = 0.0
        for idx, p in enumerate(probs):
            if p == 0.0:
                continue
            bitstring = format(idx, f"0{n}b")
            cost = qubo.evaluate(bitstring)
            exp_val += float(p) * float(cost)

        return float(exp_val)

    def _most_probable_bitstring(
        self,
        circuit: QuantumCircuit,
        num_qubits: int,
    ) -> str:
        """
        Get the most probable bitstring from the statevector of the circuit.
        """
        sv = Statevector.from_instruction(circuit)
        probs = np.abs(sv.data) ** 2
        idx = int(np.argmax(probs))
        return format(idx, f"0{num_qubits}b")
