from qiskit import QuantumCircuit
from qaoa_qubo.hamiltonian import qubo_to_ising

class QAOACircuitBuilder:
    """
    Build QAOA circuits given parameters gamma, beta.

    Designed to be independent of the solver so it can be used
    by visualization tools, runtime wrappers, or notebooks.
    """

    def __init__(self, p=1):
        self.p = p

    def build(self, gammas, betas, h=None, J=None, qubo=None):
        """
        Create a QAOA circuit for given gamma and beta parameters.

        Parameters
        ----------
        gammas : list[float]
        betas : list[float]
        h, J : dict (optional)
            Ising parameters. If omitted, compute from QUBO.
        qubo : QUBOProblem (optional)
            If h,J are not given, qubo must be provided.

        Returns
        -------
        qc : QuantumCircuit
        """
        if qubo is not None and (h is None or J is None):
            h, J, _ = qubo_to_ising(qubo)

        if h is None or J is None:
            raise ValueError("Must provide Ising parameters or a QUBOProblem.")

        num_qubits = max(max(h.keys(), default=0), max((max(edge) for edge in J), default=0)) + 1

        qc = QuantumCircuit(num_qubits)

        # Start in |+>^n
        for q in range(num_qubits):
            qc.h(q)

        # Layers
        for layer in range(self.p):
            gamma = gammas[layer]
            beta = betas[layer]

            # Cost unitary from h and J
            for i, coeff in h.items():
                qc.rz(2 * gamma * coeff, i)

            for (i, j), coeff in J.items():
                qc.cx(i, j)
                qc.rz(2 * gamma * coeff, j)
                qc.cx(i, j)

            # Mixer
            for q in range(num_qubits):
                qc.rx(2 * beta, q)

        return qc
