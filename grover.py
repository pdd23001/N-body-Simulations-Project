from qiskit.visualization import plot_distribution
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit
from qiskit.circuit.library import grover_operator, MCMTGate, ZGate
from qiskit.primitives import StatevectorSampler

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

from oracles_grover import marked_oracle
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2


class Grover_Algorithm:

    def __init__(self, oracle, backend=None, optimization_level=0):
        self.grover_op=grover_operator(oracle)
        self.backend=backend
        self.pm=None
        if self.backend:
            self.pm = generate_preset_pass_manager(target=self.backend.target, optimization_level=optimization_level)
    
    def create_grover_circuit(self, iterations):
        qc = QuantumCircuit(self.grover_op.num_qubits)
        qc.h(range(self.grover_op.num_qubits)) #Hadamard Transform
        qc.compose(self.grover_op.power(iterations), inplace=True) #grover iteration
        #Measure all qubits
        qc.measure_all()
        if self.backend:
            qc=self.pm.run(qc)
        return qc
    
    def run(self, qc):
        if not self.backend:
            result = StatevectorSampler.run([qc], shots=1).result()[0]
            bitstring = result.join_data().get_bitstrings()[0]
        else:
            job = self.backend.run(qc, shots=1)
            result=job.result()
            bitstring = next(iter(result.get_counts()))

        return bitstring
    
    def update_oracle(self, new_marked, num_qubits):
        oracle = marked_oracle(new_marked, num_qubits)
        self.grover_op = grover_operator(oracle)
    


        
    




    






