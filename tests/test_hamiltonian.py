
from qaoa_qubo.qubo import QUBOProblem
from qaoa_qubo.hamiltonian import qubo_to_ising


def compute_ising_energy(bitstring: str, h, J, const: float) -> float:
    """
    Given a bitstring in {0,1}^n, map to spins z_i ∈ {-1, +1} via z_i = 1 - 2 x_i
    and compute E = const + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j.
    """
    x = [int(b) for b in bitstring]
    z = [1 - 2 * xi for xi in x]  # 0 -> +1, 1 -> -1

    energy = const

    for i, h_i in h.items():
        energy += h_i * z[i]

    for (i, j), J_ij in J.items():
        energy += J_ij * z[i] * z[j]

    return energy


def test_qubo_to_ising_consistency_two_vars():
    # Define a small 2-variable QUBO:
    # C(x) = a0 x0 + a1 x1 + b01 x0 x1 + c
    a0, a1, b01, c = 1.0, 2.0, 3.0, 0.5

    linear = {0: a0, 1: a1}
    quadratic = {(0, 1): b01}
    qubo = QUBOProblem.from_dicts(linear, quadratic, constant=c)

    h, J, const_shift = qubo_to_ising(qubo)

    # Check equality C(x) == IsingEnergy(z(x)) for all 4 assignments
    for bitstring in ["00", "01", "10", "11"]:
        qubo_val = qubo.evaluate(bitstring)
        ising_val = compute_ising_energy(bitstring, h, J, const_shift)
        assert pytest.approx(qubo_val, rel=1e-9) == ising_val
