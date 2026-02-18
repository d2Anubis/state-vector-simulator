import unittest
import numpy as np
import math
from qomputing.circuit import QuantumCircuit
from qomputing.backend import QomputingSimulator

class TestBugs(unittest.TestCase):
    def setUp(self):
        self.sim = QomputingSimulator.get_backend()

    def test_generic_controlled_h(self):
        """Test a controlled-Hadamard gate which was previously unsupported."""
        qc = QuantumCircuit(2)
        # Initialize |10> state: q0=0, q1=1 
        # Actually our simulator starts at |0...0>. 
        # Let's target |10> -> |1>|H|0> vs |00> -> |00>
        
        # Case 1: Control is 0 (Off)
        # q0 (control) is 0. q1 (target) is 0.
        # C-H(0->1) should do NOTHING.
        qc.h(1, controls=[0]) 
        res = self.sim.run(qc)
        state = res.get_statevector() # Should be |00> = index 0 = 1.0
        self.assertAlmostEqual(abs(state[0]), 1.0)

        # Case 2: Control is 1 (On)
        qc = QuantumCircuit(2)
        qc.x(0) # Set q0 to 1
        # Now C-H(0->1) should apply H to q1.
        # q1 starts 0 -> H -> |+>
        # q0 remain 1.
        # State: |1>|0> -> |1>|+> = 1/sqrt(2) (|10> + |11>)
        # Little endian mapping: q0 is low? q1 high? 
        # Simulator tensor usage: 
        # axis_for_qubit(0) = -1 (last axis).
        # axis_for_qubit(1) = -2.
        # State vector layout is usually big-endian in array print?
        # |q_{n-1} ... q_0>
        # |00> = index 0. |01> (q0=1) = index 1.
        # |10> (q1=1) = index 2. |11> = index 3.
        
        # So |1>_q1 |+>_q0 ? No wait. 
        # Control q0, target q1.
        # q0=1. q1 starts 0.
        # H on q1: |0> -> |+>.
        # Final state: |+>_q1 |1>_q0  (if q1 is MSB)
        # = (|0> + |1>) * |1> = |01> + |11>
        # Indices: 1 and 3.
        
        qc.h(1, controls=[0])
        res = self.sim.run(qc)
        state = res.get_statevector()
        
        # Check amplitudes
        self.assertAlmostEqual(abs(state[1])**2, 0.5) # |01>
        self.assertAlmostEqual(abs(state[3])**2, 0.5) # |11>
        self.assertAlmostEqual(abs(state[0]), 0.0)
        self.assertAlmostEqual(abs(state[2]), 0.0)

    def test_multi_controlled_rotation(self):
        """Test a rotation with multiple controls."""
        # 3 qubits. C-C-RX(pi).
        # Controls q0, q1. Target q2.
        # Should be equivalent to Toffoli but with rotation.
        # If theta=pi, RX(pi) ~ -iX. 
        # So |110> -> -i|111>.
        
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.x(1)
        # Now apply C-C-RX(pi) on q2 controlled by q0, q1
        qc.rx(2, math.pi, controls=[0, 1])
        
        res = self.sim.run(qc)
        state = res.get_statevector()
        
        # Expected: |111> (index 7 if q0 is LSB)
        # q0 is index 0 (LSB). q1 index 1. q2 index 2 (MSB).
        # |q2 q1 q0> = |1 1 1> = 7.
        
        # Let's check index 3 (|011>)? No.
        # Inputs: q0=1, q1=1, q2=0. -> |011> = 3.
        # Op: q0=1,q1=1 => Apply RX(pi) on q2.
        # q2: |0> -> -i|1>.
        # Final: -i |111> = index 7.
        
        # Index 3 should be 0. Index 7 should be 1.0 magnitude.
        self.assertAlmostEqual(abs(state[7]), 1.0)
        self.assertAlmostEqual(abs(state[3]), 0.0)

if __name__ == '__main__':
    unittest.main()
