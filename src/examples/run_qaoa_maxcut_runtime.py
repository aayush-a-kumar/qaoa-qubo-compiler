from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from qaoa_qubo.problems import MaxCutProblem
from qaoa_qubo.qaoa import QAOASolver

"""
Example: Running QAOA MaxCut using an IBM Quantum backend via Qiskit Runtime.

Note:
- This script assumes you have configured QiskitRuntimeService with a saved account.
- It will consume real IBM Quantum runtime minutes and shots.
"""

def build_maxcut_problem():
    # Simple 4-node cycle graph
    edges = {
        (0, 1): 1.0,
        (1, 2): 1.0,
        (2, 3): 1.0,
        (3, 0): 1.0,
    }
    return MaxCutProblem(edges)


def main():
    # Connect to IBM Runtime
    service = QiskitRuntimeService()  # assumes you've already saved your account
    backend = service.backend("ibm_fez")  # or any backend you have access to

    runtime_sampler = SamplerV2(backend=backend)

    problem = build_maxcut_problem()

    solver = QAOASolver(
        p=1,
        maxiter=20,
        mode="sampler",
        sampler=runtime_sampler,
        shots=2048,
        seed=0,
    )

    result = solver.solve(problem)

    print("Backend:", backend.name)
    print("Best bitstring:", result.best_bitstring)
    print("Best cost:", result.best_cost)
    print("Optimal gammas:", result.optimal_gammas)
    print("Optimal betas:", result.optimal_betas)
    print("Iterations:", len(result.history))


if __name__ == "__main__":
    main()
