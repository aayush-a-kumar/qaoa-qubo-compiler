from __future__ import annotations

from typing import Optional, Sequence, Tuple, Dict, Literal

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler as Sampler
from scipy.optimize import minimize

from .qubo import QUBOProblem
from .problems import MaxCutProblem
from .hamiltonian import qubo_to_ising
from .result import QAOAResult
from qaoa_qubo.circuit_builder import QAOACircuitBuilder


BackendMode = Literal["statevector", "sampler"]


class QAOASolver:
    """
    A simple QAOA solver for QUBO-based problems.

    Supports two execution modes:

    - mode="statevector" (default): uses exact statevector simulation.
    - mode="sampler": uses a Qiskit Sampler primitive (can be local Aer or
      IBM Runtime Sampler, depending on what you pass in).

    This makes it easy to:
    - develop and debug locally with statevector, and
    - later switch to real hardware or managed backends by passing a sampler.
    """

    def __init__(
        self,
        p: int = 1,
        maxiter: int = 100,
        mode: str = "statevector",  # "statevector" or "sampler"
        sampler=None,
        shots: int = 1024,
        seed: Optional[int] = None,
        transpile_backend=None,
        transpile_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        p : int
            Number of QAOA layers.
        maxiter : int
            Maximum optimizer iterations.
        mode : {"statevector", "sampler"}
            Execution backend mode.
        sampler : Optional[Sampler]
            Qiskit Sampler primitive. If None and mode="sampler", a default
            Sampler() will be constructed. This can also be an IBM Runtime
            sampler object.
        shots : int
            Number of shots when using sampler mode.
        seed : Optional[int]
            Random seed for parameter initialization.
        """
        if p <= 0:
            raise ValueError("p (number of QAOA layers) must be positive")

        if mode not in ("statevector", "sampler"):
            raise ValueError(f"Unsupported mode: {mode}")

        self.p = p
        self.maxiter = maxiter
        self.mode = mode
        self.sampler = sampler
        self.shots = shots
        self.rng = np.random.default_rng(seed)

        # Optional backend used to transpile circuits before sampling
        self.transpile_backend = transpile_backend
        self.transpile_kwargs = transpile_kwargs or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self, problem: QUBOProblem | MaxCutProblem) -> QAOAResult:
        """
        Run QAOA on a QUBOProblem or MaxCutProblem.

        Returns a QAOAResult with:
        - best_bitstring
        - best_cost
        - optimal (gammas, betas)
        - optimization history
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

        # Convert to Ising form
        h, J, const_shift = qubo_to_ising(qubo)
        num_qubits = qubo.num_variables
        if num_qubits is None:
            raise ValueError("QUBOProblem.num_variables is not set")

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

            if self.mode == "statevector":
                exp_val = self._expectation_statevector(qc, qubo) + const_shift
            else:
                exp_val = self._expectation_sampler(qc, qubo) + const_shift

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

        # Build final circuit and get the most probable bitstring
        qc_opt = self._build_qaoa_circuit(
            num_qubits=num_qubits,
            h=h,
            J=J,
            gammas=gammas_opt,
            betas=betas_opt,
        )

        if self.mode == "statevector":
            best_bitstring = self._most_probable_bitstring_statevector(qc_opt, num_qubits)
        else:
            best_bitstring = self._most_probable_bitstring_sampler(qc_opt, num_qubits)

        best_cost = float(qubo.evaluate(best_bitstring))
        optimal_value = float(objective(theta_opt))

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
        Build the p-layer QAOA circuit:

        - Initial state |+>^n
        - For each layer k:
            * cost unitary from Ising Hamiltonian (Z, ZZ rotations)
            * mixer unitary from X mixer (RX rotations)
        """
        qc = QuantumCircuit(num_qubits)

        # Initial |+>^n
        for q in range(num_qubits):
            qc.h(q)

        for layer in range(self.p):
            gamma = float(gammas[layer])
            beta = float(betas[layer])

            # Cost Hamiltonian e^{-i gamma H_C}
            for i, h_i in h.items():
                if h_i != 0.0:
                    angle = 2.0 * gamma * h_i
                    qc.rz(angle, i)

            for (i, j), J_ij in J.items():
                if J_ij != 0.0:
                    angle = 2.0 * gamma * J_ij
                    qc.rzz(angle, i, j)

            # Mixer e^{-i beta ∑ X_i}
            for q in range(num_qubits):
                qc.rx(2.0 * beta, q)

        return qc

    # ----------------- expectation value helpers ----------------------
    def _expectation_statevector(
        self,
        circuit: QuantumCircuit,
        qubo: QUBOProblem,
    ) -> float:
        """
        Compute E[C(x)] = sum_x p(x) C(x) using Statevector simulation.
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

    def _expectation_sampler(
        self,
        circuit: QuantumCircuit,
        qubo: QUBOProblem,
    ) -> float:
        """
        Compute E[C(x)] using a sampler-like primitive.

        Supports:
        - qiskit.primitives.Sampler (local) via result.quasi_dists[0]
        - qiskit_ibm_runtime.SamplerV2 (Runtime) via result[0].data.meas.get_counts()
        """
        if qubo.num_variables is None:
            raise ValueError("QUBOProblem.num_variables is not set")
        n = qubo.num_variables

        # --- Ensure the circuit has measurements ---
        qc = circuit.copy()
        has_measure = any(op.operation.name == "measure" for op in qc.data)
        if not has_measure:
            qc.measure_all()

        # Optionally transpile for a hardware backend
        if self.transpile_backend is not None:
            circuit = transpile(
                circuit,
                backend=self.transpile_backend,
                **self.transpile_kwargs,
            )

        sampler = self.sampler or Sampler()

        job = sampler.run([circuit], shots=self.shots)
        result = job.result()

        # Case 1: local Sampler with quasi_dists
        if hasattr(result, "quasi_dists"):
            dist = result.quasi_dists[0]
            exp_val = 0.0
            for idx, p in dist.items():
                bitstring = format(idx, f"0{n}b")
                cost = qubo.evaluate(bitstring)
                exp_val += float(p) * float(cost)
            return float(exp_val)

        # Case 2: Runtime SamplerV2-style result: list-like of PubResult
        # with .data.meas.get_counts()
        pub0 = result[0]
        meas = getattr(pub0.data, "meas", None)
        if meas is None or not hasattr(meas, "get_counts"):
            raise RuntimeError(
                "Sampler result does not provide quasi_dists or meas.get_counts; "
                "unsupported sampler type."
            )

        counts = meas.get_counts()
        total_shots = sum(counts.values())
        if total_shots == 0:
            raise RuntimeError("Sampler returned zero shots in counts.")

        exp_val = 0.0
        for bitstring, count in counts.items():
            # Use last n bits in case of registers
            bitstring = str(bitstring)[-n:]
            prob = count / total_shots
            cost = qubo.evaluate(bitstring)
            exp_val += float(prob) * float(cost)

        return float(exp_val)

    # ----------------- bitstring extraction helpers -------------------
    def _most_probable_bitstring_statevector(
        self,
        circuit: QuantumCircuit,
        num_qubits: int,
    ) -> str:
        sv = Statevector.from_instruction(circuit)
        probs = np.abs(sv.data) ** 2
        idx = int(np.argmax(probs))
        return format(idx, f"0{num_qubits}b")

    def _most_probable_bitstring_sampler(
        self,
        circuit: QuantumCircuit,
        num_qubits: int,
    ) -> str:
        """
        Get most probable bitstring from a sampler-like primitive.

        Supports:
        - qiskit.primitives.Sampler
        - qiskit_ibm_runtime.SamplerV2
        """
        # --- Ensure measurements ---
        qc = circuit.copy()
        has_measure = any(op.operation.name == "measure" for op in qc.data)
        if not has_measure:
            qc.measure_all()

        # Optional transpilation for hardware
        if self.transpile_backend is not None:
            circuit = transpile(
                circuit,
                backend=self.transpile_backend,
                **self.transpile_kwargs,
            )

        sampler = self.sampler or Sampler()
        job = sampler.run([circuit], shots=self.shots)
        result = job.result()

        # Case 1: local Sampler with quasi_dists
        if hasattr(result, "quasi_dists"):
            dist = result.quasi_dists[0]
            idx = max(dist, key=dist.get)
            return format(idx, f"0{num_qubits}b")

        # Case 2: Runtime SamplerV2-style result with counts
        pub0 = result[0]
        meas = getattr(pub0.data, "meas", None)
        if meas is None or not hasattr(meas, "get_counts"):
            raise RuntimeError(
                "Sampler result does not provide quasi_dists or meas.get_counts; "
                "unsupported sampler type."
            )

        counts = meas.get_counts()
        # Pick bitstring with highest count
        best_bitstring = max(counts, key=counts.get)
        # Use last num_qubits bits in case of registers
        return str(best_bitstring)[-num_qubits:]

