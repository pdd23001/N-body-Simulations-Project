# qae_utils.py

import math
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit_ibm_runtime import Sampler
from oracles_grover import boolean_oracle

def make_state_prep_A(n: int, marked_states) -> QuantumCircuit:
    """Prepare |+>^n on data, then write success to flag via boolean oracle."""
    qc = QuantumCircuit(n + 1, name="A")
    qc.h(range(n))
    qc.compose(boolean_oracle(marked_states, n), inplace=True)
    return qc

def make_problem(n: int, marked_states) -> EstimationProblem:
    A = make_state_prep_A(n, marked_states)
    return EstimationProblem(
        state_preparation=A,
        objective_qubits=[n],          # the flag qubit
        post_processing=lambda a: a     # return raw a in [0,1]
    )

def estimate_a_theta_mu(N: int, n: int, marked_states,
                        epsilon_target=1e-3, alpha=0.05,
                        sampler: Sampler | None = None):
    """
    Estimate a ≈ sin^2(theta) via IAE, then theta and μ (≈ # marked = a*N).
    Returns: (a_hat, theta_hat, mu_hat, result_obj)
    """
    problem = make_problem(n, marked_states)
    # sampler = sampler or Sampler(backend=AerSimulator())
    iae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon_target,  # abs error on a
        alpha=alpha,                    # confidence (1-alpha)
    )
    res = iae.estimate(problem)
    a_hat = float(res.estimation)
    a_hat = max(0.0, min(1.0, a_hat))  # clamp numerically
    theta_hat = math.asin(math.sqrt(a_hat))
    mu_hat = a_hat * N                 # μ = a * N (μ = #marked)
    return a_hat, theta_hat, mu_hat, res
