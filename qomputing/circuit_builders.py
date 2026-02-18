"""Circuit builders for programmatic use and random-circuit XEB."""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, Sequence, Tuple

from .circuit import QuantumCircuit
from .gates import (
    DEFAULT_MULTI_QUBIT_GATES,
    DEFAULT_SINGLE_QUBIT_GATES,
    DEFAULT_TWO_QUBIT_GATES,
)


def random_circuit(
    *,
    num_qubits: int,
    depth: int,
    single_qubit_gates: Iterable[str] | None = None,
    two_qubit_gates: Iterable[str] | None = None,
    multi_qubit_gates: Iterable[str] | None = None,
    seed: int | None = None,
) -> QuantumCircuit:
    """Build a random circuit with one layer of single-qubit gates and optional two/multi-qubit gates per depth.

    Args:
        num_qubits: Number of qubits.
        depth: Number of layers (each layer: single-qubit gates on all qubits, then one two-qubit, optionally one multi-qubit).
        single_qubit_gates: Gate names to sample from; default is DEFAULT_SINGLE_QUBIT_GATES.
        two_qubit_gates: Two-qubit gate names; default is DEFAULT_TWO_QUBIT_GATES.
        multi_qubit_gates: Multi-qubit gate names (e.g. ccx, ccz, cswap); default is DEFAULT_MULTI_QUBIT_GATES.
        seed: Random seed for reproducibility.

    Returns:
        A QuantumCircuit ready for simulation or XEB.
    """
    if num_qubits <= 0:
        raise ValueError("Number of qubits must be positive")
    if depth <= 0:
        raise ValueError("Depth must be positive")

    single_qubit_gates = list(single_qubit_gates or DEFAULT_SINGLE_QUBIT_GATES)
    if not single_qubit_gates:
        raise ValueError("Single-qubit gate set must not be empty")

    two_qubit_gates = list(two_qubit_gates or DEFAULT_TWO_QUBIT_GATES)
    multi_qubit_gates = list(multi_qubit_gates or DEFAULT_MULTI_QUBIT_GATES)
    if num_qubits > 1 and not two_qubit_gates:
        raise ValueError("Two-qubit gate set must not be empty when num_qubits > 1")

    multi_specs: Dict[str, Tuple[int, int]] = {
        "ccx": (2, 1),
        "ccz": (2, 1),
        "cswap": (1, 2),
    }

    rng = random.Random(seed)
    circuit = QuantumCircuit(num_qubits)

    for _ in range(depth):
        for qubit in range(num_qubits):
            gate_name = rng.choice(single_qubit_gates)
            _apply_single_qubit_gate(circuit, gate_name, qubit, rng)

        if num_qubits > 1 and two_qubit_gates:
            gate_name = rng.choice(two_qubit_gates)
            q1, q2 = rng.sample(range(num_qubits), 2)
            _apply_two_qubit_gate(circuit, gate_name, q1, q2, rng)

        if num_qubits > 2 and multi_qubit_gates:
            gate_name = rng.choice(multi_qubit_gates)
            if gate_name not in multi_specs:
                raise ValueError(f"Unsupported multi-qubit gate: {gate_name}")
            controls_required, targets_required = multi_specs[gate_name]
            total_required = controls_required + targets_required
            if num_qubits >= total_required:
                selected = rng.sample(range(num_qubits), total_required)
                controls = selected[:controls_required]
                targets = selected[controls_required:]
                _apply_multi_qubit_gate(circuit, gate_name, controls, targets, rng)

    return circuit


def _validate_angle(gate_name: str, angle: float | None) -> float:
    """Ensure rotation gates receive a valid numerical angle."""
    if angle is None or not isinstance(angle, (float, int)):
        raise ValueError(f"Rotation gate {gate_name} requires a valid float angle.")
    return float(angle)


def _apply_single_qubit_gate(
    circuit: QuantumCircuit,
    gate_name: str,
    qubit: int,
    rng: random.Random,
) -> None:
    two_pi = 2.0 * math.pi
    if gate_name == "id":
        circuit.i(qubit)
    elif gate_name == "x":
        circuit.x(qubit)
    elif gate_name == "y":
        circuit.y(qubit)
    elif gate_name == "z":
        circuit.z(qubit)
    elif gate_name == "h":
        circuit.h(qubit)
    elif gate_name == "s":
        circuit.s(qubit)
    elif gate_name == "sdg":
        circuit.sdg(qubit)
    elif gate_name == "t":
        circuit.t(qubit)
    elif gate_name == "tdg":
        circuit.tdg(qubit)
    elif gate_name == "sx":
        circuit.sx(qubit)
    elif gate_name == "sxdg":
        circuit.sxdg(qubit)
    elif gate_name in {"rx", "ry", "rz", "u1", "u2", "u3"}:
        if gate_name == "rx":
            angle = _validate_angle(gate_name, rng.uniform(0.0, two_pi))
            circuit.rx(qubit, angle)
        elif gate_name == "ry":
            angle = _validate_angle(gate_name, rng.uniform(0.0, two_pi))
            circuit.ry(qubit, angle)
        elif gate_name == "rz":
            angle = _validate_angle(gate_name, rng.uniform(0.0, two_pi))
            circuit.rz(qubit, angle)
        elif gate_name == "u1":
            circuit.u1(qubit, _validate_angle("u1", rng.uniform(0.0, two_pi)))
        elif gate_name == "u2":
            circuit.u2(
                qubit,
                _validate_angle("u2", rng.uniform(0.0, two_pi)),
                _validate_angle("u2", rng.uniform(0.0, two_pi)),
            )
        elif gate_name == "u3":
            circuit.u3(
                qubit,
                _validate_angle("u3", rng.uniform(0.0, two_pi)),
                _validate_angle("u3", rng.uniform(0.0, two_pi)),
                _validate_angle("u3", rng.uniform(0.0, two_pi)),
            )
    else:
        raise ValueError(f"Unsupported single-qubit gate: {gate_name}")


def _apply_two_qubit_gate(
    circuit: QuantumCircuit,
    gate_name: str,
    q1: int,
    q2: int,
    rng: random.Random,
) -> None:
    two_pi = 2.0 * math.pi
    if gate_name == "cx":
        circuit.cx(q1, q2)
    elif gate_name == "cy":
        circuit.cy(q1, q2)
    elif gate_name == "cz":
        circuit.cz(q1, q2)
    elif gate_name == "cp":
        angle = _validate_angle("cp", rng.uniform(0.0, two_pi))
        circuit.cp(q1, q2, angle)
    elif gate_name == "swap":
        circuit.swap(q1, q2)
    elif gate_name == "iswap":
        circuit.iswap(q1, q2)
    elif gate_name == "sqrtiswap":
        circuit.sqrt_iswap(q1, q2)
    elif gate_name == "rxx":
        circuit.rxx(q1, q2, rng.uniform(0.0, two_pi))
    elif gate_name == "ryy":
        circuit.ryy(q1, q2, rng.uniform(0.0, two_pi))
    elif gate_name == "rzz":
        circuit.rzz(q1, q2, rng.uniform(0.0, two_pi))
    elif gate_name == "csx":
        circuit.csx(q1, q2)
    else:
        raise ValueError(f"Unsupported two-qubit gate: {gate_name}")


def _apply_multi_qubit_gate(
    circuit: QuantumCircuit,
    gate_name: str,
    controls: Sequence[int],
    targets: Sequence[int],
    rng: random.Random,
) -> None:
    if gate_name == "ccx":
        circuit.ccx(controls[0], controls[1], targets[0])
    elif gate_name == "ccz":
        circuit.ccz(controls[0], controls[1], targets[0])
    elif gate_name == "cswap":
        circuit.cswap(controls[0], targets[0], targets[1])
    else:
        raise ValueError(f"Unsupported multi-qubit gate: {gate_name}")
