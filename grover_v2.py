# grover.py — Qiskit 2.x, with circuit caching + optional batching

from __future__ import annotations
from typing import Dict, List, Optional
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit_aer import AerSimulator

# ---------- reusable diffuser ----------
def _diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc

class Grover_Algorithm:
    """
    Grover with:
      - oracle/diffuser packed into Gates (no repeated compose)
      - per-j transpiled circuit cache
      - optional batching helper for repeated runs at same j
    """

    def __init__(self,
                 oracle: QuantumCircuit,
                 backend=None,
                 optimization_level: int = 1,
                 measure: bool = True):
        """
        oracle: unitary QuantumCircuit on n qubits (no classical bits).
        backend: Qiskit backend (defaults to AerSimulator()).
        optimization_level: transpiler opt level (0–3).
        measure: measure all qubits in created circuits if True.
        """
        self.backend = backend or AerSimulator()
        self.opt_level = optimization_level
        self.measure = measure
        self.n = oracle.num_qubits

        # Build reusable Gates once
        self._set_oracle(oracle)

        # Per-j transpiled cache
        self._tcache: Dict[int, QuantumCircuit] = {}

    # --------- public API (compatible with your notebook) ---------

    def create_grover_circuit(self, iterations: int) -> QuantumCircuit:
        """
        Return an UNTRANSPILED circuit for given j=iterations.
        We tag j in metadata so run() can pick the cached compiled version.
        """
        qc = self._build_untranspiled(iterations)
        qc.metadata = dict(qc.metadata or {}, j=iterations)
        return qc

    def run(self, circuit: QuantumCircuit, shots: int = 1) -> str:
        """
        Execute circuit. If 'j' is present in metadata, reuse transpiled cache.
        Returns most frequent bitstring.
        """
        j = (circuit.metadata or {}).get("j", None)
        if j is not None:
            tqc = self._get_transpiled_j(j)
        else:
            tqc = transpile(circuit, self.backend, optimization_level=self.opt_level)

        job = self.backend.run(tqc, shots=shots)
        res = job.result()
        counts = res.get_counts()
        return max(counts, key=counts.get)
    

    def run_j(self, j: int, shots: int = 1) -> str:
        """
        Execute circuit by j
        """
        tqc = self._get_transpiled_j(j)
        job = self.backend.run(tqc, shots=shots)
        res = job.result()
        counts = res.get_counts()
        return max(counts, key=counts.get)
    
    # --------- batching helper ---------

    def run_batch_by_j(self, j: int, num_repeats: int, shots: int = 1) -> List[str]:
        """
        Submit the same-j circuit 'num_repeats' times in one backend call.
        Returns list of majority bitstrings (length == num_repeats).
        """
        tqc = self._get_transpiled_j(j)
        circuits = [tqc] * num_repeats
        job = self.backend.run(circuits, shots=shots)
        result = job.result()
        out: List[str] = []
        for i in range(len(circuits)):
            counts = result.get_counts(i)
            out.append(max(counts, key=counts.get))
        return out
    
    def run_hits_for_j(self, j:int, shots:int, marked:set[str], seen)->int:
        tqc = self._get_transpiled_j(j)
        res = self.backend.run(tqc, shots=shots).result()
        counts = res.get_counts()
        hits=sum(counts.get(s, 0) for s in marked)
        new = [s for s in counts.keys() if s in marked and s not in seen]
        return hits, new
        
    
  # --------- oracle updater ---------

    def update_oracle(self, new_oracle: QuantumCircuit):
        """
        Rebuild oracle/diffuser-based step gate and invalidate per-j cache.
        Call this whenever the marked set changes.
        """
        self._set_oracle(new_oracle)
        self._tcache.clear()

    # ------------------ internals ------------------

    def _set_oracle(self, oracle: QuantumCircuit):
        if oracle.num_qubits != self.n:
            raise ValueError("Oracle qubit count mismatch.")
        # Pack unitary subcircuits as Gates (faster reuse than compose)
        self.oracle_gate: Gate = oracle.to_gate(label="Oracle")
        self.diffuser_gate: Gate = _diffuser(self.n).to_gate(label="Diffuser")

        # One Grover step gate: G = Diffuser ∘ Oracle (append order: Oracle then Diffuser)
        step = QuantumCircuit(self.n, name="GroverStep")
        step.append(self.oracle_gate, range(self.n))
        step.append(self.diffuser_gate, range(self.n))
        self.grover_step_gate: Gate = step.to_gate(label="G")

    def _build_untranspiled(self, j: int) -> QuantumCircuit:
        qc = QuantumCircuit(self.n, self.n, name=f"Grover_{j}")
        qc.h(range(self.n))
        for _ in range(j):
            qc.append(self.grover_step_gate, range(self.n))
        if self.measure:
            qc.measure(range(self.n), range(self.n))
        return qc

    def _get_transpiled_j(self, j: int) -> QuantumCircuit:
        """Fetch transpiled circuit for iteration-count j from cache (build once if missing)."""
        if j not in self._tcache:
            qc = self._build_untranspiled(j)
            self._tcache[j] = transpile(qc, self.backend, optimization_level=self.opt_level)
        return self._tcache[j]
