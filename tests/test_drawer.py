import unittest
from qomputing.circuit import QuantumCircuit
from qomputing.visualisation import draw_circuit

class TestDrawer(unittest.TestCase):
    def test_empty_circuit(self):
        qc = QuantumCircuit(1)
        # It should basically print empty wires or "Empty Circuit" if check is NumGates==0
        # The visualizer returns "Empty Circuit" if num_qubits == 0.
        # But if num_qubits > 0 and no gates?
        # The current visualizer code:
        # if circuit.num_qubits == 0: return "Empty Circuit"
        # Since circuit(0) raises error, we can't test "Empty Circuit" output with standard class unless we bypass check.
        # However, for 1 qubit with no gates, it draws wires.
        # Let's adjust expectation.
        output = draw_circuit(qc)
        self.assertIn("q0:", output)

    def test_simple_circuit(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        output = draw_circuit(qc)
        # Check for key characters
        self.assertIn("H", output)
        self.assertIn("■", output)
        self.assertIn("│", output)
        
    def test_measurements(self):
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        output = draw_circuit(qc)
        self.assertIn("M", output)
        self.assertIn("c: 1/", output)

if __name__ == '__main__':
    unittest.main()
