"""Utility for comparing the state vector simulator against Cirq.

This script generates random quantum circuits, executes them with both
the in-tree state vector simulator and Cirq's reference simulator, and
reports discrepancies in amplitudes, probabilities, and XEB fidelity.

Usage (after installing Cirq):

    python tools/cirq_comparison.py --max-qubits 3 --depths 3 5
"""

from __future__ import annotations

import argparse
import cmath
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
from qomputing.circuit_builders import random_circuit

try:
    import cirq
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "Cirq is required for this comparison script. Install with `pip install cirq`."
    ) from exc

import numpy as np

from qomputing import (
    QuantumCircuit,
    StateVectorSimulator,
    compute_linear_xeb_fidelity,
)
from qomputing import cli as _cli
from qomputing.gates import (
    DEFAULT_MULTI_QUBIT_GATES,
    DEFAULT_SINGLE_QUBIT_GATES,
    DEFAULT_TWO_QUBIT_GATES,
)
GLOBAL_PHASE_TOLERANCE = 1e-12


@dataclass
class ComparisonMetrics:
    num_qubits: int
    depth: int
    circuit_index: int
    amplitude_error: float
    probability_error: float
    xeb_error: float | None


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    simulator = StateVectorSimulator()

    metrics: List[ComparisonMetrics] = []

    for num_qubits in range(args.min_qubits, args.max_qubits + 1):
        for depth in args.depths:
            for circuit_index in range(args.circuits_per_config):
                circuit_seed = rng.randint(0, 2**31 - 1)
                circuit = random_circuit(
                num_qubits=num_qubits,
                depth=depth,
                single_qubit_gates=args.single_qubit_gates,
                two_qubit_gates=args.two_qubit_gates,
                multi_qubit_gates=args.multi_qubit_gates,
                seed=circuit_seed,
                )
                metric = _compare_with_cirq(
                    circuit,
                    simulator=simulator,
                    shots=args.shots,
                    seed=circuit_seed,
                    depth=depth,
                    circuit_index=circuit_index,
                )
                metrics.append(metric)
                _print_metric(metric)

    summary = _compute_summary(metrics)
    _print_summary(summary)

    max_amp_err = summary["max_amplitude_error"]
    max_prob_err = summary["max_probability_error"]
    amp_ok = max_amp_err <= args.amplitude_tolerance
    prob_ok = max_prob_err <= args.probability_tolerance
    xeb_ok = True
    if args.shots > 0:
        max_xeb_err = summary["max_xeb_error"]
        xeb_ok = max_xeb_err <= args.xeb_tolerance

    return 0 if amp_ok and prob_ok and xeb_ok else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare simulator outputs with Cirq")
    parser.add_argument("--min-qubits", type=int, default=1, help="Minimum number of qubits to test")
    parser.add_argument("--max-qubits", type=int, default=3, help="Maximum number of qubits to test")
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[3],
        help="List of circuit depths to evaluate",
    )
    parser.add_argument(
        "--circuits-per-config",
        type=int,
        default=3,
        help="Number of random circuits per (qubits, depth) combo",
    )
    parser.add_argument(
        "--single-qubit-gates",
        type=str,
        nargs="*",
        default=DEFAULT_SINGLE_QUBIT_GATES,
        help="Pool of single-qubit gates to sample from",
    )
    parser.add_argument(
        "--two-qubit-gates",
        type=str,
        nargs="*",
        default=DEFAULT_TWO_QUBIT_GATES,
        help="Pool of two-qubit gates to sample from",
    )
    parser.add_argument(
        "--multi-qubit-gates",
        type=str,
        nargs="*",
        default=DEFAULT_MULTI_QUBIT_GATES,
        help="Pool of multi-qubit gates to sample from",
    )
    parser.add_argument("--shots", type=int, default=512, help="Number of samples for XEB comparison")
    parser.add_argument("--seed", type=int, default=1234, help="Master random seed")
    parser.add_argument(
        "--amplitude-tolerance",
        type=float,
        default=1e-6,
        help="Maximum tolerated amplitude sup-norm difference",
    )
    parser.add_argument(
        "--probability-tolerance",
        type=float,
        default=1e-6,
        help="Maximum tolerated probability sup-norm difference",
    )
    parser.add_argument(
        "--xeb-tolerance",
        type=float,
        default=1e-3,
        help="Maximum tolerated linear XEB fidelity difference",
    )
    return parser


def _compare_with_cirq(
    circuit: QuantumCircuit,
    *,
    simulator: StateVectorSimulator,
    shots: int,
    seed: int,
    depth: int,
    circuit_index: int,
) -> ComparisonMetrics:
    ours = simulator.run(circuit, shots=shots, seed=seed)
    cirq_circuit, qubits = _circuit_to_cirq(circuit)
    cirq_sim = cirq.Simulator(dtype=np.complex128)
    cirq_result = cirq_sim.simulate(cirq_circuit)
    cirq_state = _to_little_endian_state(cirq_result.final_state_vector, circuit.num_qubits)

    aligned_cirq_state = _align_global_phase(ours.final_state, cirq_state)
    amplitude_error = _max_difference(ours.final_state, aligned_cirq_state)

    cirq_probabilities = [abs(amplitude) ** 2 for amplitude in aligned_cirq_state]
    probability_error = _max_difference(ours.probabilities, cirq_probabilities)

    xeb_error = None
    if shots > 0:
        sample_rng = random.Random(seed)
        samples = _sample_bitstrings(sample_rng, cirq_probabilities, circuit.num_qubits, shots)
        xeb_ours = compute_linear_xeb_fidelity(ours.probabilities, samples)
        xeb_cirq = compute_linear_xeb_fidelity(cirq_probabilities, samples)
        xeb_error = abs(xeb_ours - xeb_cirq)

    return ComparisonMetrics(
        num_qubits=circuit.num_qubits,
        depth=depth,
        circuit_index=circuit_index,
        amplitude_error=amplitude_error,
        probability_error=probability_error,
        xeb_error=xeb_error,
    )


def _circuit_to_cirq(circuit: QuantumCircuit) -> Tuple[cirq.Circuit, Tuple[cirq.Qid, ...]]:
    qubits = tuple(cirq.LineQubit.range(circuit.num_qubits))
    operations: List[cirq.Operation] = []
    for gate in circuit.gates:
        if len(gate.targets) == 1 and not gate.controls:
            target = qubits[gate.targets[0]]
            operations.append(_single_qubit_operation(gate.name, target, gate.params))
        elif gate.name == "cx" and len(gate.controls) == 1 and len(gate.targets) == 1:
            control = qubits[gate.controls[0]]
            target = qubits[gate.targets[0]]
            operations.append(cirq.CNOT(control, target))
        elif gate.name == "cy" and len(gate.controls) == 1 and len(gate.targets) == 1:
            control = qubits[gate.controls[0]]
            target = qubits[gate.targets[0]]
            operations.append(cirq.ControlledGate(cirq.Y)(control, target))
        elif gate.name == "cz" and len(gate.controls) == 1 and len(gate.targets) == 1:
            control = qubits[gate.controls[0]]
            target = qubits[gate.targets[0]]
            operations.append(cirq.CZ(control, target))
        elif gate.name == "cp" and len(gate.controls) == 1 and len(gate.targets) == 1:
            control = qubits[gate.controls[0]]
            target = qubits[gate.targets[0]]
            phi = float(gate.params.get("phi", 0.0))
            operations.append(cirq.CZPowGate(exponent=phi / np.pi)(control, target))
        elif gate.name == "swap" and len(gate.targets) == 2:
            q1 = qubits[gate.targets[0]]
            q2 = qubits[gate.targets[1]]
            operations.append(cirq.SWAP(q1, q2))
        elif gate.name == "iswap" and len(gate.targets) == 2:
            q1 = qubits[gate.targets[0]]
            q2 = qubits[gate.targets[1]]
            operations.append(cirq.ISWAP(q1, q2))
        elif gate.name == "sqrtiswap" and len(gate.targets) == 2:
            q1 = qubits[gate.targets[0]]
            q2 = qubits[gate.targets[1]]
            operations.append((cirq.ISWAP ** 0.5)(q1, q2))
        elif gate.name in {"rxx", "ryy", "rzz"} and len(gate.targets) == 2:
            q1 = qubits[gate.targets[0]]
            q2 = qubits[gate.targets[1]]
            theta = float(gate.params.get("theta", 0.0))
            exponent = theta / np.pi
            if gate.name == "rxx":
                operations.append(cirq.XXPowGate(exponent=exponent)(q1, q2))
            elif gate.name == "ryy":
                operations.append(cirq.YYPowGate(exponent=exponent)(q1, q2))
            else:
                operations.append(cirq.ZZPowGate(exponent=exponent)(q1, q2))
        elif gate.name == "csx" and len(gate.controls) == 1 and len(gate.targets) == 1:
            control = qubits[gate.controls[0]]
            target = qubits[gate.targets[0]]
            operations.append(cirq.ControlledGate(cirq.X ** 0.5)(control, target))
        elif gate.name == "ccx" and len(gate.controls) == 2 and len(gate.targets) == 1:
            c1 = qubits[gate.controls[0]]
            c2 = qubits[gate.controls[1]]
            target = qubits[gate.targets[0]]
            operations.append(cirq.CCX(c1, c2, target))
        elif gate.name == "ccz" and len(gate.controls) == 2 and len(gate.targets) == 1:
            c1 = qubits[gate.controls[0]]
            c2 = qubits[gate.controls[1]]
            target = qubits[gate.targets[0]]
            operations.append(cirq.CCZ(c1, c2, target))
        elif gate.name == "cswap" and len(gate.controls) == 1 and len(gate.targets) == 2:
            control = qubits[gate.controls[0]]
            q1 = qubits[gate.targets[0]]
            q2 = qubits[gate.targets[1]]
            operations.append(cirq.CSWAP(control, q1, q2))
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported gate for Cirq conversion: {gate}")
    return cirq.Circuit(operations), qubits


def _single_qubit_operation(name: str, qubit: cirq.Qid, params: Dict[str, float]) -> cirq.Operation:
    two_pi = 2.0 * math.pi
    if name == "id":
        return cirq.I(qubit)
    if name == "x":
        return cirq.X(qubit)
    if name == "y":
        return cirq.Y(qubit)
    if name == "z":
        return cirq.Z(qubit)
    if name == "h":
        return cirq.H(qubit)
    if name == "s":
        return cirq.S(qubit)
    if name == "sdg":
        return (cirq.S ** -1)(qubit)
    if name == "t":
        return cirq.T(qubit)
    if name == "tdg":
        return (cirq.T ** -1)(qubit)
    if name == "sx":
        return (cirq.X ** 0.5)(qubit)
    if name == "sxdg":
        return (cirq.X ** -0.5)(qubit)
    if name in {"rx", "ry", "rz"}:
        theta = float(params.get("theta", 0.0))
        if name == "rx":
            return cirq.rx(theta)(qubit)
        if name == "ry":
            return cirq.ry(theta)(qubit)
        return cirq.rz(theta)(qubit)
    if name == "u1":
        lam = float(params.get("lambda", 0.0))
        matrix = np.array([[1.0, 0.0], [0.0, cmath.exp(1j * lam)]], dtype=complex)
        return cirq.MatrixGate(matrix)(qubit)
    if name == "u2":
        phi = float(params.get("phi", 0.0))
        lam = float(params.get("lambda", 0.0))
        matrix = (1 / math.sqrt(2)) * np.array(
            [
                [1.0, -cmath.exp(1j * lam)],
                [cmath.exp(1j * phi), cmath.exp(1j * (phi + lam))],
            ],
            dtype=complex,
        )
        return cirq.MatrixGate(matrix)(qubit)
    if name == "u3":
        theta = float(params.get("theta", 0.0))
        phi = float(params.get("phi", 0.0))
        lam = float(params.get("lambda", 0.0))
        cos = math.cos(theta / 2.0)
        sin = math.sin(theta / 2.0)
        matrix = np.array(
            [
                [cos, -cmath.exp(1j * lam) * sin],
                [cmath.exp(1j * phi) * sin, cmath.exp(1j * (phi + lam)) * cos],
            ],
            dtype=complex,
        )
        return cirq.MatrixGate(matrix)(qubit)
    raise ValueError(f"Unsupported single-qubit gate: {name}")


def _align_global_phase(
    reference: Sequence[complex],
    target: Sequence[complex],
) -> List[complex]:
    phase = 1 + 0j
    for ref, tgt in zip(reference, target):
        if abs(tgt) > GLOBAL_PHASE_TOLERANCE:
            phase = ref / tgt
            break
    return [tgt * phase for tgt in target]


def _max_difference(
    reference: Sequence[float | complex],
    candidate: Sequence[float | complex],
) -> float:
    return max(abs(r - c) for r, c in zip(reference, candidate))


def _to_little_endian_state(state: Sequence[complex], num_qubits: int) -> List[complex]:
    """Reorder Cirq's big-endian state vector into little-endian ordering."""
    size = 1 << num_qubits
    if len(state) != size:
        raise ValueError("State vector size does not match qubit count")
    reordered = [0j] * size
    for index, amplitude in enumerate(state):
        little_index = _reverse_bits(index, num_qubits)
        reordered[little_index] = amplitude
    return reordered


def _reverse_bits(value: int, width: int) -> int:
    reversed_value = 0
    for _ in range(width):
        reversed_value = (reversed_value << 1) | (value & 1)
        value >>= 1
    return reversed_value


def _sample_bitstrings(
    rng: random.Random,
    probabilities: Sequence[float],
    num_qubits: int,
    shots: int,
) -> List[str]:
    outcomes = [format(index, f"0{num_qubits}b") for index in range(len(probabilities))]
    return rng.choices(outcomes, weights=probabilities, k=shots)


def _print_metric(metric: ComparisonMetrics) -> None:
    summary = (
        f"[qubits={metric.num_qubits} depth={metric.depth} circuit={metric.circuit_index}] "
        f"amp_err={metric.amplitude_error:.3e} "
        f"prob_err={metric.probability_error:.3e}"
    )
    if metric.xeb_error is not None:
        summary += f" xeb_err={metric.xeb_error:.3e}"
    print(summary)


def _compute_summary(metrics: List[ComparisonMetrics]) -> Dict[str, float]:
    max_amp = max(metric.amplitude_error for metric in metrics) if metrics else 0.0
    max_prob = max(metric.probability_error for metric in metrics) if metrics else 0.0
    max_xeb = max(
        (metric.xeb_error for metric in metrics if metric.xeb_error is not None),
        default=0.0,
    )
    return {
        "max_amplitude_error": max_amp,
        "max_probability_error": max_prob,
        "max_xeb_error": max_xeb,
    }


def _print_summary(summary: Dict[str, float]) -> None:
    print("=== Comparison Summary ===")
    print(f"Max amplitude error   : {summary['max_amplitude_error']:.6e}")
    print(f"Max probability error : {summary['max_probability_error']:.6e}")
    print(f"Max XEB error         : {summary['max_xeb_error']:.6e}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

