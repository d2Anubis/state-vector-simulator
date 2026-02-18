"""Two-qubit gate implementations and registry."""

from __future__ import annotations

import cmath
import math
from typing import Callable, Dict, List

import numpy as np

from .. import linalg
from ..circuit import Gate

TwoQubitHandler = Callable[[np.ndarray, Gate, int, np.dtype], np.ndarray]


def _apply_unitary(state: np.ndarray, qubits: List[int], matrix: np.ndarray, num_qubits: int) -> np.ndarray:
    return linalg.apply_unitary(state, qubits, matrix, num_qubits)


def apply_cnot(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    if len(gate.controls) != 1 or len(gate.targets) != 1:
        raise ValueError("cx gate requires one control and one target")
    control = gate.controls[0]
    target = gate.targets[0]
    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=dtype,
    )
    return _apply_unitary(state, [control, target], matrix, num_qubits)


def apply_cy(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    if len(gate.controls) != 1 or len(gate.targets) != 1:
        raise ValueError("cy gate requires one control and one target")
    control = gate.controls[0]
    target = gate.targets[0]
    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0],
        ],
        dtype=dtype,
    )
    return _apply_unitary(state, [control, target], matrix, num_qubits)


def apply_cz(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    if len(gate.controls) != 1 or len(gate.targets) != 1:
        raise ValueError("cz gate requires one control and one target")
    control = gate.controls[0]
    target = gate.targets[0]
    matrix = np.diag([1, 1, 1, -1]).astype(dtype)
    return _apply_unitary(state, [control, target], matrix, num_qubits)


def apply_cp(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    if len(gate.controls) != 1 or len(gate.targets) != 1:
        raise ValueError("cp gate requires one control and one target")
    phi = float(gate.params.get("phi", 0.0))
    matrix = np.diag([1, 1, 1, cmath.exp(1j * phi)]).astype(dtype)
    return _apply_unitary(state, [gate.controls[0], gate.targets[0]], matrix, num_qubits)


def apply_swap(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    if len(gate.targets) != 2:
        raise ValueError("swap gate requires two target qubits")
    q1, q2 = gate.targets
    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=dtype,
    )
    return _apply_unitary(state, [q1, q2], matrix, num_qubits)


def apply_iswap(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    if len(gate.targets) != 2:
        raise ValueError("iswap gate requires two target qubits")
    q1, q2 = gate.targets
    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=dtype,
    )
    return _apply_unitary(state, [q1, q2], matrix, num_qubits)


def apply_sqrtiswap(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    if len(gate.targets) != 2:
        raise ValueError("sqrtiswap gate requires two target qubits")
    q1, q2 = gate.targets
    s = 1.0 / math.sqrt(2)
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, s, 1j * s, 0.0],
            [0.0, 1j * s, s, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )
    return _apply_unitary(state, [q1, q2], matrix, num_qubits)


def _rotation_gate(pauli: np.ndarray, theta: float, dtype: np.dtype) -> np.ndarray:
    half = theta / 2.0
    return math.cos(half) * np.eye(4, dtype=dtype) - 1j * math.sin(half) * pauli


def apply_rxx(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    if len(gate.targets) != 2:
        raise ValueError("rxx gate requires two target qubits")
    theta = float(gate.params.get("theta", 0.0))
    pauli = np.kron(np.array([[0, 1], [1, 0]], dtype=dtype), np.array([[0, 1], [1, 0]], dtype=dtype))
    matrix = _rotation_gate(pauli, theta, dtype)
    return _apply_unitary(state, gate.targets, matrix, num_qubits)


def apply_ryy(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    if len(gate.targets) != 2:
        raise ValueError("ryy gate requires two target qubits")
    theta = float(gate.params.get("theta", 0.0))
    pauli = np.kron(
        np.array([[0, -1j], [1j, 0]], dtype=dtype),
        np.array([[0, -1j], [1j, 0]], dtype=dtype),
    )
    matrix = _rotation_gate(pauli, theta, dtype)
    return _apply_unitary(state, gate.targets, matrix, num_qubits)


def apply_rzz(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    if len(gate.targets) != 2:
        raise ValueError("rzz gate requires two target qubits")
    theta = float(gate.params.get("theta", 0.0))
    pauli = np.kron(
        np.array([[1, 0], [0, -1]], dtype=dtype),
        np.array([[1, 0], [0, -1]], dtype=dtype),
    )
    matrix = _rotation_gate(pauli, theta, dtype)
    return _apply_unitary(state, gate.targets, matrix, num_qubits)


def apply_csx(state: np.ndarray, gate: Gate, num_qubits: int, dtype: np.dtype) -> np.ndarray:
    if len(gate.controls) != 1 or len(gate.targets) != 1:
        raise ValueError("csx gate requires one control and one target")
    control = gate.controls[0]
    target = gate.targets[0]
    sx = 0.5 * np.array(
        [
            [1 + 1j, 1 - 1j],
            [1 - 1j, 1 + 1j],
        ],
        dtype=dtype,
    )
    matrix = np.block(
        [
            [np.eye(2, dtype=dtype), np.zeros((2, 2), dtype=dtype)],
            [np.zeros((2, 2), dtype=dtype), sx],
        ]
    )
    return _apply_unitary(state, [control, target], matrix, num_qubits)


HANDLERS: Dict[str, TwoQubitHandler] = {
    "cx": apply_cnot,
    "cy": apply_cy,
    "cz": apply_cz,
    "cp": apply_cp,
    "swap": apply_swap,
    "iswap": apply_iswap,
    "sqrtiswap": apply_sqrtiswap,
    "rxx": apply_rxx,
    "ryy": apply_ryy,
    "rzz": apply_rzz,
    "csx": apply_csx,
}

GATE_NAMES: List[str] = list(HANDLERS.keys())

__all__ = ["HANDLERS", "GATE_NAMES", "TwoQubitHandler"]

