from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import SamplerV2 as Sampler
import numpy as np
from qiskit_aer import AerSimulator

def normalize(vec):
    vec = np.array(vec, dtype=float)
    return vec / np.linalg.norm(vec)

def wta_qiskit2x_with_registers(P, V, W):
    """
    Registered implementation of the Weapon–Target Assignment quantum algorithm.
    Updated to Qiskit 2.x syntax and SamplerV2 primitive.
    """
    m = len(P)
    n = len(P[0])
    
    # Calculate required qubits
    qw = int(np.ceil(np.log2(m)))
    qt = int(np.ceil(np.log2(n)))
    total_qubits = qw + qt
    
    # Define registers according to Qiskit 2.x API
    weapon_reg = QuantumRegister(qw, "weapon")
    target_reg = QuantumRegister(qt, "target")
    ancilla = AncillaRegister(1, "anc")     # optional auxiliary qubit for controlled ops
    classical_reg = ClassicalRegister(total_qubits, "c")
    
    # Initialize circuit with registers
    qc = QuantumCircuit(weapon_reg, target_reg, ancilla, classical_reg, name="WTA")
    
    # Pad and normalize P matrix
    m_pad, n_pad = 2**qw, 2**qt
    P_pad = np.zeros((m_pad, n_pad))
    P_pad[:m, :n] = P
    P_vec = P_pad.flatten()
    P_norm = normalize(P_vec)
    
    # Amplitude encoding via initialize (IBM docs conform)
    qc.initialize(P_norm, weapon_reg[:] + target_reg[:])
    # --- Random pre-rotations before W=0 and W=1 blocks ---
    rng = np.random.default_rng(42)       # optional seed for reproducibility
    theta_w0 = rng.uniform(-0.1*np.pi, 0.1*np.pi) # random angle for W=0 branch
    theta_w1 = rng.uniform(-0.1*np.pi, 0.1*np.pi) # random angle for W=1 branch

    # ----- control on W = 0 -----
    qc.x(weapon_reg[0])                  # X-sandwich to make control=0 behave as control=1
    qc.cry(theta_w0, weapon_reg[0], target_reg[0])
    qc.x(weapon_reg[0])                  # undo X

    # ----- control on W = 1 -----
    qc.cry(theta_w1, weapon_reg[0], target_reg[0])
    # Encode target importance as controlled phase rotations
    V_pad = np.zeros(n_pad)
    V_pad[:n] = normalize(V)
    for j in range(qt):
        theta = 2 * np.pi * V_pad[j]
        qc.cry(theta, target_reg[j], ancilla[0])
    
    # Measurement section
    qc.measure(weapon_reg[:] + target_reg[:], classical_reg[:total_qubits])
    
    backend=AerSimulator()
    # Execute using Qiskit Runtime SamplerV2
    sampler = Sampler(mode=backend)   # defaults to local
    job = sampler.run([qc], shots=4096)
    result = job.result()
    
    # Retrieve counts from SamplerV2 API
    counts = result[0].join_data().get_counts()
    return qc, counts

# Define probabilities matrix for 2 weapons and 2 targets
P = [
    [0.5, 0.8],  # Weapon 1 probabilities for target 1 and 2
    [0.7, 0.2],  # Weapon 2
]

# Define importance values for 2 targets
V = [0.45, 0.75]

# Define weights for 2 weapons (may represent number of each weapon)
W = [1, 1]

qc, counts = wta_qiskit2x_with_registers(P, V, W)

print(counts)

def decode_counts_to_assignment(counts, num_weapons, num_targets):
    m = 2 ** num_weapons
    n = 2 ** num_targets
    assignment = np.zeros((m, n))

    for bitstring, cnt in counts.items():
        # Assuming left bits weapon, right bits target
        weapon_bits = bitstring[0]  # first bit
        target_bits = bitstring[1]  # second bit
        weapon_idx = int(weapon_bits, 2)
        target_idx = int(target_bits, 2)
        assignment[weapon_idx, target_idx] += cnt

    assignment = assignment / assignment.sum()  # normalize
    return assignment

# Example
weapon_qubit = 1
target_qubit = 1
assignment_matrix = decode_counts_to_assignment(counts, weapon_qubit, target_qubit)
print(assignment_matrix)