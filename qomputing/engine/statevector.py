"""NumPy-based state vector simulator."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .. import linalg
from ..circuit import QuantumCircuit
from . import measurements, registry
from .result import SimulationResult


class StateVectorSimulator:
    """State vector simulator delegating gate logic to handler modules."""

    DEFAULT_MAX_QUBITS = 30
    DEFAULT_MAX_SHOTS = 1_000_000

    def __init__(
        self,
        *,
        dtype: np.dtype = np.complex128,
        max_qubits: int | None = DEFAULT_MAX_QUBITS,
        max_shots: int | None = DEFAULT_MAX_SHOTS,
    ) -> None:
        if max_qubits is not None and max_qubits < 1:
            raise ValueError("max_qubits must be >= 1 or None")
        if max_shots is not None and max_shots < 1:
            raise ValueError("max_shots must be >= 1 or None")
        self.dtype = dtype
        self.max_qubits = max_qubits
        self.max_shots = max_shots

    def run(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> SimulationResult:
        if shots is not None and shots < 0:
            raise ValueError("shots must be >= 0")
        if self.max_qubits is not None and circuit.num_qubits > self.max_qubits:
            raise ValueError(f"num_qubits {circuit.num_qubits} exceeds max_qubits {self.max_qubits}")
        if shots is not None and self.max_shots is not None and shots > self.max_shots:
            raise ValueError(f"shots {shots} exceeds max_shots {self.max_shots}")

        state = linalg.initial_state(circuit.num_qubits, dtype=self.dtype)
        for gate in circuit.gates:
            state = registry.apply_gate(state, gate, circuit.num_qubits, self.dtype)

        probabilities = np.abs(state) ** 2
        result_shots = shots if shots and shots > 0 else None
        samples: Optional[List[str]] = None
        counts: Optional[Dict[str, int]] = None
        if result_shots:
            rng = np.random.default_rng(seed)
            samples = measurements.sample_measurements(probabilities, circuit.num_qubits, result_shots, rng)
            if circuit.num_clbits and circuit._measurements:
                counts = measurements.samples_to_classical_counts(
                    samples, circuit.num_qubits, circuit._measurements, circuit.num_clbits
                )
            else:
                counts = measurements.counts_from_samples(samples)

        return SimulationResult(
            circuit=circuit,
            final_state=state,
            probabilities=probabilities,
            shots=result_shots,
            samples=samples,
            counts=counts,
        )
