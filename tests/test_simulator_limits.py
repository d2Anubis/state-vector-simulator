import numpy as np
import pytest

from qomputing import QuantumCircuit, StateVectorSimulator
from qomputing.engine import statevector as statevector_module


def test_run_rejects_num_qubits_over_default_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    simulator = StateVectorSimulator()
    circuit = QuantumCircuit(31)

    def _fail_initial_state(*_args, **_kwargs):
        raise AssertionError("initial_state should not be called when qubit limit is exceeded")

    monkeypatch.setattr(statevector_module.linalg, "initial_state", _fail_initial_state)
    with pytest.raises(ValueError, match=r"exceeding the simulator limit"):
        simulator.run(circuit)


def test_run_rejects_shots_over_default_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    simulator = StateVectorSimulator()
    circuit = QuantumCircuit(1)

    def _fail_sample_measurements(*_args, **_kwargs):
        raise AssertionError("sample_measurements should not be called when shot limit is exceeded")

    monkeypatch.setattr(statevector_module.measurements, "sample_measurements", _fail_sample_measurements)
    with pytest.raises(ValueError, match=r"shots 1000001 exceeds max_shots 1000000"):
        simulator.run(circuit, shots=1_000_001, seed=7)


def test_run_accepts_custom_limits() -> None:
    simulator = StateVectorSimulator(max_qubits=2, max_shots=3)
    circuit = QuantumCircuit(2)
    result = simulator.run(circuit, shots=3, seed=7)
    assert result.shots == 3
    assert result.samples is not None
    assert len(result.samples) == 3


def test_run_allows_disabling_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    simulator = StateVectorSimulator(max_qubits=None, max_shots=None)
    circuit = QuantumCircuit(64)

    def _tiny_state(num_qubits: int, *, dtype: np.dtype) -> np.ndarray:
        assert num_qubits == 64
        return np.array([1.0 + 0j], dtype=dtype)

    monkeypatch.setattr(statevector_module.linalg, "initial_state", _tiny_state)
    result = simulator.run(circuit, shots=0)
    assert result.shots is None


def test_constructor_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_qubits"):
        StateVectorSimulator(max_qubits=0)
    with pytest.raises(ValueError, match="max_shots"):
        StateVectorSimulator(max_shots=0)
