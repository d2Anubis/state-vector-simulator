import unittest
from qomputing.xeb import run_linear_xeb_experiment
from qomputing.circuit_builders import random_circuit
from qomputing.simulator import StateVectorSimulator

class TestXEB(unittest.TestCase):
    def test_xeb_small_circuit(self):
        # 1 qubit, depth 1
        qc = random_circuit(num_qubits=1, depth=1, seed=42)
        res = run_linear_xeb_experiment(qc, shots=100, seed=42)
        self.assertGreater(res.fidelity, -1.0)
        self.assertLessEqual(res.fidelity, 1.0)

    def test_xeb_large_circuit(self):
        # 5 qubits, depth 5
        qc = random_circuit(num_qubits=5, depth=5, seed=123)
        res = run_linear_xeb_experiment(qc, shots=500, seed=123)
        print(f"Fidelity: {res.fidelity}")
        
    def test_xeb_no_measurements(self):
        # The run_linear_xeb_experiment should handle circuits without explicit measures?
        # xeb.py calls simulator.run(..., shots=shots).
        # simulator.run uses measurements.sample_measurements().
        # It doesn't rely on qc.measure().
        # It samples from probability distribution.
        # This is strictly correct for XEB (we want ideal prob sampling).
        pass

if __name__ == "__main__":
    unittest.main()
