from qiskit import QuantumCircuit
from qiskit.circuit.library import grover_operator, MCMTGate, ZGate, XGate

def marked_oracle(marked_states, num_qubits):
    """Build a Grover oracle for multiple marked states. This basically gives a oracle that can mark multiple states according to whatever bitstrings we do give it. 
       Not apt implementation of oracle for n-body simulations. Need an oracle that calculates distances between points in a pair on the fly and
       flips that indexes/btstrings phase. This calculation happens in superposition (i.e. where the quantum advantage comes in). Right now using an oracle that 
       takes marked states as an input and then marks them for testing the close_neighbors algorithms. Will soon integrate an on-the-fly oracle with grover's instead
    """
    if not isinstance(marked_states, set):
        marked_states = {marked_states}
 
    qc = QuantumCircuit(num_qubits)
    # Mark each target state in the input list
    for target in marked_states:
        # Flip target bit-string to match Qiskit bit-ordering
        rev_target = target[::-1]
        # Find the indices of all the '0' elements in bit-string
        zero_inds = [
            ind
            for ind in range(num_qubits)
            if rev_target.startswith("0", ind)
        ]
        # Add a multi-controlled Z-gate with pre- and post-applied X-gates (open-controls)
        # where the target bit-string has a '0' entry
        if zero_inds:
            qc.x(zero_inds)
        qc.compose(MCMTGate(ZGate(), num_qubits - 1, 1), inplace=True)
        if zero_inds:
            qc.x(zero_inds)
    return qc


def boolean_oracle(marked_states, n: int) -> QuantumCircuit:
    """
    n data qubits [0..n-1] + 1 flag qubit at index n.
    Flips the flag to |1> iff the data register equals ANY of the marked bitstrings.
    Bitstrings assumed little-endian in the rest of your code; we reverse accordingly.
    """
    if not isinstance(marked_states, set):
        marked_states = {marked_states}
    qc = QuantumCircuit(n + 1, name="BoolOracle")
    for s in marked_states:
        s = s[::-1]                 # match your little-endian convention
        zeros = [i for i, b in enumerate(s) if b == '0']
        if zeros:
            qc.x(zeros)             # map 0-controls to 1-controls

        controls = list(range(n))
        target = n                  # flag qubit
        qc.append(MCMTGate(XGate(), num_ctrl_qubits=n, num_target_qubits=1),
                  controls + [target])

        if zeros:
            qc.x(zeros)             # uncompute
    return qc


from qiskit.circuit import AncillaRegister
def marked_oracle_v2(marked_states, num_qubits, mcx_mode="v-chain"):
    if isinstance(marked_states, (str, int)):
        marked_states = {marked_states}

    qc = QuantumCircuit(num_qubits, name="oracle")
    target = num_qubits - 1
    ctrls = list(range(num_qubits - 1))

    # Add ancillas if the chosen mode needs them (v-chain uses k-2)
    num_controls = len(ctrls)
    need_anc = max(0, num_controls - 2) if mcx_mode == "v-chain" else 0
    anc = None
    if need_anc > 0:
        anc = AncillaRegister(need_anc, "anc")
        qc.add_register(anc)

    for bitstr in marked_states:
        if isinstance(bitstr, int):
            bitstr = format(bitstr, f"0{num_qubits}b")
        rev = bitstr[::-1]

        zero_idx = [i for i, b in enumerate(rev) if b == "0"]
        if zero_idx:
            qc.x(zero_idx)

        qc.h(target)
        if num_controls == 0:
            qc.z(target)
        elif num_controls == 1:
            qc.cz(ctrls[0], target)
        else:
            qc.mcx(ctrls, target, ancilla_qubits=anc, mode=mcx_mode)
        qc.h(target)

        if zero_idx:
            qc.x(zero_idx)

    return qc
