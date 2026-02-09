"""Gate registry mapping names to handler implementations."""

from __future__ import annotations

import numpy as np

from ..linalg import apply_controlled_unitary
from ..circuit import Gate
from ..gates import (
    MULTI_QUBIT_HANDLERS,
    SINGLE_QUBIT_HANDLERS,
    TWO_QUBIT_HANDLERS,
)
from ..gates.single_qubit import get_matrix


def apply_gate(
    state: np.ndarray,
    gate: Gate,
    num_qubits: int,
    dtype: np.dtype,
) -> np.ndarray:
    name = gate.name
    if name in SINGLE_QUBIT_HANDLERS and not gate.controls:
        handler = SINGLE_QUBIT_HANDLERS[name]
        return handler(state, gate, num_qubits, dtype)
    if name in TWO_QUBIT_HANDLERS:
        handler = TWO_QUBIT_HANDLERS[name]
        return handler(state, gate, num_qubits, dtype)
    if name in MULTI_QUBIT_HANDLERS:
        handler = MULTI_QUBIT_HANDLERS[name]
        return handler(state, gate, num_qubits, dtype)
    
    # Fallback: Check if it is a controlled version of a single-qubit gate
    if gate.controls:
        try:
            # Try to get the underlying matrix for the base gate
            matrix = get_matrix(gate, dtype)
            return apply_controlled_unitary(state, gate.controls, gate.targets, matrix, num_qubits)
        except ValueError:
            # Not a known single qubit gate, or other issue
            pass
            
    raise ValueError(f"Unsupported gate: {gate.name}")

