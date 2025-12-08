from qaoa_qubo.problems import MaxCutProblem
from qaoa_qubo.qaoa import QAOASolver
from qaoa_qubo.visualize import plot_result_history

# Simple 4-node example graph
# Edge weights: fully connected graph
edges = {
    (0, 1): 1,
    (1, 2): 1,
    (2, 3): 1,
    (3, 0): 1,
}

problem = MaxCutProblem(edges)
solver = QAOASolver(p=1, maxiter=50, mode="sampler", shots=2048, seed=42)

result = solver.solve(problem)

print("Best bitstring:", result.best_bitstring)
print("MaxCut value:", result.best_cost)
print("Optimal params:")
print("  gammas =", result.optimal_gammas)
print("  betas  =", result.optimal_betas)

plot_result_history(result)


if __name__ == "__main__":
    main()
