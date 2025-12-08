# QAOA-QUBO Compiler
*A lightweight Python library that converts QUBO problems into QAOA circuits and solves them using statevector or sampler backends.*

This project implements an **end-to-end QAOA pipeline**:

- Define optimization problems as **QUBOs**
- Convert to **Ising Hamiltonians**
- Build **parameterized QAOA circuits**
- Evaluate via **statevector simulation** or **Qiskit Sampler primitives**
- Use classical optimization to obtain the **best bitstring and cost**

---

## Pipeline Overview
                     ┌────────────────────────────┐
                     │       QUBO Problem         │
                     │  (linear, quadratic dicts) │
                     └──────────────┬─────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────┐
                     │   Ising Transformation      │
                     │     h, J, constant shift    │
                     └──────────────┬─────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────┐
                     │     QAOA Circuit Builder    │
                     │  (cost + mixer layers, p)   │
                     └──────────────┬─────────────┘
                                    │
                                    ▼
                 ┌───────────────────────────────┬───────────────────────────────┐
                 │                               │                               │
                 ▼                               ▼                               ▼
     ┌──────────────────────┐        ┌──────────────────────┐        ┌──────────────────────┐
     │   Statevector Eval   │        │   Sampler (Shots)    │        │  IBM Runtime (future)│
     └───────────┬──────────┘        └───────────┬──────────┘        └───────────┬──────────┘
                 │                               │                               │
                 └──────────────────────┬─────────┴───────────────┬──────────────┘
                                        ▼                         ▼
                          ┌────────────────────────────┐   ┌────────────────────────┐
                          │ Classical Optimization θ    │   │  Probability Distribution │
                          │ (COBYLA, SPSA, etc.)        │   │  from sampled bitstrings │
                          └──────────────┬─────────────┘   └─────────────┬──────────┘
                                         ▼                               ▼
                               ┌──────────────────────────┐    ┌──────────────────────────┐
                               │    Best Bitstring        │    │      Best Cost Value      │
                               └──────────────────────────┘    └──────────────────────────┘
## Installation

Inside the repo root:

```bash
pip install -e .
```

## Example: Solving MaxCut via QAOA

Run:

```bash
python examples/run_qaoa_maxcut.py
```

Example output:

```
Best bitstring: 0101
MaxCut value: 4.0
Optimal parameters:
  gammas = [...]
  betas  = [...]
```

## Project Structure

```
src/qaoa_qubo/
│
├── qubo.py          # QUBOProblem class: cost evaluation, helpers
├── hamiltonian.py   # QUBO → Ising transformation
├── qaoa.py          # QAOASolver (circuit builder + optimizer)
├── problems.py      # MaxCutProblem abstraction → QUBO
├── result.py        # QAOAResult dataclass (final outputs)
└── utils.py         # (future utilities)
```

## Roadmap

- ~~Add Qiskit Runtime support~~ Done!  
- Add analytic gradient-based QAOA (parameter shift)  
- Add visualization tools (energy landscape, convergence plots)  
- Add additional problem classes (k-SAT, graph coloring, clustering)

## License

MIT


