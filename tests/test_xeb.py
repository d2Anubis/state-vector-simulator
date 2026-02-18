import unittest

import pytest

from qomputing.circuit_builders import random_circuit
from qomputing.xeb import compute_linear_xeb_fidelity, run_linear_xeb_experiment


def test_compute_linear_xeb_fidelity_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        compute_linear_xeb_fidelity([1.0], [])


def test_compute_linear_xeb_fidelity_rejects_mismatched_sample_width() -> None:
    with pytest.raises(ValueError, match="same bit-width"):
        compute_linear_xeb_fidelity([0.25, 0.25, 0.25, 0.25], ["00", "0"])


def test_compute_linear_xeb_fidelity_rejects_non_binary_samples() -> None:
    with pytest.raises(ValueError, match="binary strings"):
        compute_linear_xeb_fidelity([0.25, 0.25, 0.25, 0.25], ["00", "2a"])


def test_compute_linear_xeb_fidelity_rejects_probability_vector_size_mismatch() -> None:
    with pytest.raises(ValueError, match="vector size"):
        compute_linear_xeb_fidelity([0.5, 0.5], ["00", "01"])


def test_compute_linear_xeb_fidelity_valid_input() -> None:
    fidelity = compute_linear_xeb_fidelity([0.1, 0.2, 0.3, 0.4], ["00", "01", "10", "11"])
    assert fidelity == pytest.approx(0.0)


class TestXEB(unittest.TestCase):
    """Integration tests for XEB with random circuits."""

    def test_xeb_small_circuit(self) -> None:
        qc = random_circuit(num_qubits=1, depth=1, seed=42)
        res = run_linear_xeb_experiment(qc, shots=100, seed=42)
        self.assertGreater(res.fidelity, -1.0)
        self.assertLessEqual(res.fidelity, 1.0)

    def test_xeb_larger_circuit(self) -> None:
        qc = random_circuit(num_qubits=5, depth=5, seed=123)
        res = run_linear_xeb_experiment(qc, shots=500, seed=123)
        self.assertGreater(res.fidelity, -1.0)
        # Linear XEB fidelity can exceed 1.0 for some circuits; just check it's finite
        self.assertLess(res.fidelity, 100.0)
