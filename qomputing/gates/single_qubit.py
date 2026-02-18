"""Single-qubit gate implementations and registry."""

from __future__ import annotations

import cmath
import math
from typing import Callable, Dict, List

import numpy as np

from .. import linalg
from ..circuit import Gate

SingleQubitHandler = Callable[[np.ndarray, Gate, int, np.dtype], np.ndarray]


def _apply_matrix(state: np.ndarray, gate: Gate, num_qubits: int, matrix: np.ndarray) -> np.ndarray:
    if len(gate.targets) != 1:
        raise ValueError(f"{gate.name} gate expects one target qubit")
    target = gate.targets[0]
    tensor = linalg.reshape_state(state, num_qubits)
    axis = linalg.axis_for_qubit(target)
    updated = np.tensordot(matrix, tensor, axes=([1], [axis]))
    updated = np.moveaxis(updated, 0, axis)
    return updated.reshape(state.shape)


def apply_identity(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    return _apply_matrix(state, gate, num_qubits, np.eye(2, dtype=dtype))


def apply_x(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    return _apply_matrix(state, gate, num_qubits, np.array([[0, 1], [1, 0]], dtype=dtype))


def apply_y(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    return _apply_matrix(state, gate, num_qubits, np.array([[0, -1j], [1j, 0]], dtype=dtype))


def apply_z(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    return _apply_matrix(state, gate, num_qubits, np.array([[1, 0], [0, -1]], dtype=dtype))


def apply_h(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    return _apply_matrix(
        state,
        gate,
        num_qubits,
        (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=dtype),
    )


def apply_s(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    return _apply_matrix(state, gate, num_qubits, np.array([[1, 0], [0, 1j]], dtype=dtype))


def apply_sdg(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    return _apply_matrix(state, gate, num_qubits, np.array([[1, 0], [0, -1j]], dtype=dtype))


def apply_t(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    return _apply_matrix(
        state,
        gate,
        num_qubits,
        np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=dtype),
    )


def apply_tdg(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    return _apply_matrix(
        state,
        gate,
        num_qubits,
        np.array([[1, 0], [0, cmath.exp(-1j * math.pi / 4)]], dtype=dtype),
    )


def apply_sx(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    matrix = 0.5 * np.array(
        [
            [1 + 1j, 1 - 1j],
            [1 - 1j, 1 + 1j],
        ],
        dtype=dtype,
    )
    return _apply_matrix(state, gate, num_qubits, matrix)


def apply_sxdg(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    matrix = 0.5 * np.array(
        [
            [1 - 1j, 1 + 1j],
            [1 + 1j, 1 - 1j],
        ],
        dtype=dtype,
    )
    return _apply_matrix(state, gate, num_qubits, matrix)


def apply_rx(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    theta = float(gate.params.get("theta", 0.0))
    half = theta / 2.0
    matrix = np.array(
        [
            [math.cos(half), -1j * math.sin(half)],
            [-1j * math.sin(half), math.cos(half)],
        ],
        dtype=dtype,
    )
    return _apply_matrix(state, gate, num_qubits, matrix)


def apply_ry(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    theta = float(gate.params.get("theta", 0.0))
    half = theta / 2.0
    matrix = np.array(
        [
            [math.cos(half), -math.sin(half)],
            [math.sin(half), math.cos(half)],
        ],
        dtype=dtype,
    )
    return _apply_matrix(state, gate, num_qubits, matrix)


def apply_rz(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    theta = float(gate.params.get("theta", 0.0))
    half = theta / 2.0
    matrix = np.array(
        [
            [cmath.exp(-1j * half), 0],
            [0, cmath.exp(1j * half)],
        ],
        dtype=dtype,
    )
    return _apply_matrix(state, gate, num_qubits, matrix)


def apply_u1(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    lam = float(gate.params.get("lambda", 0.0))
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, cmath.exp(1j * lam)],
        ],
        dtype=dtype,
    )
    return _apply_matrix(state, gate, num_qubits, matrix)


def apply_u2(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    phi = float(gate.params.get("phi", 0.0))
    lam = float(gate.params.get("lambda", 0.0))
    matrix = (1 / math.sqrt(2)) * np.array(
        [
            [1.0, -cmath.exp(1j * lam)],
            [cmath.exp(1j * phi), cmath.exp(1j * (phi + lam))],
        ],
        dtype=dtype,
    )
    return _apply_matrix(state, gate, num_qubits, matrix)


def apply_u3(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    theta = float(gate.params.get("theta", 0.0))
    phi = float(gate.params.get("phi", 0.0))
    lam = float(gate.params.get("lambda", 0.0))
    cos = math.cos(theta / 2.0)
    sin = math.sin(theta / 2.0)
    matrix = np.array(
        [
            [cos, -cmath.exp(1j * lam) * sin],
            [cmath.exp(1j * phi) * sin, cmath.exp(1j * (phi + lam)) * cos],
        ],
        dtype=dtype,
    )
    return _apply_matrix(state, gate, num_qubits, matrix)


def get_matrix(gate: Gate, dtype: np.dtype = np.complex128) -> np.ndarray:
    """Return the 2x2 unitary matrix for a single-qubit gate."""
    name = gate.name
    if name == "id":
        return np.eye(2, dtype=dtype)
    if name == "x":
        return np.array([[0, 1], [1, 0]], dtype=dtype)
    if name == "y":
        return np.array([[0, -1j], [1j, 0]], dtype=dtype)
    if name == "z":
        return np.array([[1, 0], [0, -1]], dtype=dtype)
    if name == "h":
        return (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=dtype)
    if name == "s":
        return np.array([[1, 0], [0, 1j]], dtype=dtype)
    if name == "sdg":
        return np.array([[1, 0], [0, -1j]], dtype=dtype)
    if name == "t":
        return np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=dtype)
    if name == "tdg":
        return np.array([[1, 0], [0, cmath.exp(-1j * math.pi / 4)]], dtype=dtype)
    if name == "sx":
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=dtype)
    if name == "sxdg":
        return 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=dtype)
    
    # Parameterized gates
    if name == "rx":
        theta = float(gate.params.get("theta", 0.0))
        half = theta / 2.0
        return np.array([[math.cos(half), -1j * math.sin(half)], [-1j * math.sin(half), math.cos(half)]], dtype=dtype)
    if name == "ry":
        theta = float(gate.params.get("theta", 0.0))
        half = theta / 2.0
        return np.array([[math.cos(half), -math.sin(half)], [math.sin(half), math.cos(half)]], dtype=dtype)
    if name == "rz":
        theta = float(gate.params.get("theta", 0.0))
        half = theta / 2.0
        return np.array([[cmath.exp(-1j * half), 0], [0, cmath.exp(1j * half)]], dtype=dtype)
    if name == "u1":
        lam = float(gate.params.get("lambda", 0.0))
        return np.array([[1.0, 0.0], [0.0, cmath.exp(1j * lam)]], dtype=dtype)
    if name == "u2":
        phi = float(gate.params.get("phi", 0.0))
        lam = float(gate.params.get("lambda", 0.0))
        return (1 / math.sqrt(2)) * np.array(
            [[1.0, -cmath.exp(1j * lam)], [cmath.exp(1j * phi), cmath.exp(1j * (phi + lam))]], dtype=dtype
        )
    if name == "u3":
        theta = float(gate.params.get("theta", 0.0))
        phi = float(gate.params.get("phi", 0.0))
        lam = float(gate.params.get("lambda", 0.0))
        cos = math.cos(theta / 2.0)
        sin = math.sin(theta / 2.0)
        return np.array(
            [[cos, -cmath.exp(1j * lam) * sin], [cmath.exp(1j * phi) * sin, cmath.exp(1j * (phi + lam)) * cos]],
            dtype=dtype,
        )
            
    raise ValueError(f"Unknown single qubit gate matrix: {name}")


HANDLERS: Dict[str, SingleQubitHandler] = {
    "id": apply_identity,
    "x": apply_x,
    "y": apply_y,
    "z": apply_z,
    "h": apply_h,
    "s": apply_s,
    "sdg": apply_sdg,
    "t": apply_t,
    "tdg": apply_tdg,
    "sx": apply_sx,
    "sxdg": apply_sxdg,
    "rx": apply_rx,
    "ry": apply_ry,
    "rz": apply_rz,
    "u1": apply_u1,
    "u2": apply_u2,
    "u3": apply_u3,
}

GATE_NAMES: List[str] = list(HANDLERS.keys())

__all__ = ["HANDLERS", "GATE_NAMES", "SingleQubitHandler", "get_matrix"]

