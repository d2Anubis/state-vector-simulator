"""Command line interface for the state vector simulator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .circuit import QuantumCircuit
from .circuit_builders import random_circuit
from .gates import (
    DEFAULT_MULTI_QUBIT_GATES,
    DEFAULT_SINGLE_QUBIT_GATES,
    DEFAULT_TWO_QUBIT_GATES,
)
from .simulator import StateVectorSimulator
from .xeb import run_linear_xeb_experiment


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the simulator CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "simulate":
        return _cmd_simulate(args)
    if args.command == "random-circuit":
        return _cmd_random_circuit(args)

    parser.print_help()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(description="State vector simulator CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand: simulate
    simulate_parser = subparsers.add_parser("simulate", help="Simulate a circuit from JSON")
    simulate_parser.add_argument("--circuit", type=Path, required=True, help="Path to circuit JSON file")
    simulate_parser.add_argument("--shots", type=int, default=0, help="Number of measurement shots to sample")
    simulate_parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    simulate_parser.add_argument(
        "--max-qubits",
        type=int,
        default=StateVectorSimulator.DEFAULT_MAX_QUBITS,
        help=f"Maximum allowed qubits (default: {StateVectorSimulator.DEFAULT_MAX_QUBITS})",
    )
    simulate_parser.add_argument(
        "--max-shots",
        type=int,
        default=StateVectorSimulator.DEFAULT_MAX_SHOTS,
        help=f"Maximum allowed shots (default: {StateVectorSimulator.DEFAULT_MAX_SHOTS})",
    )

    # Subcommand: random-circuit
    rand_parser = subparsers.add_parser("random-circuit", help="Generate a random circuit and run XEB")
    rand_parser.add_argument("--qubits", type=int, required=True, help="Number of qubits")
    rand_parser.add_argument("--depth", type=int, required=True, help="Circuit depth (layers)")
    rand_parser.add_argument("--shots", type=int, required=True, help="Number of measurement shots")
    rand_parser.add_argument("--seed", type=int, default=None, help="Random seed for circuit generation and sampling")
    
    rand_parser.add_argument(
        "--max-qubits",
        type=int,
        default=StateVectorSimulator.DEFAULT_MAX_QUBITS,
        help=f"Maximum allowed qubits (default: {StateVectorSimulator.DEFAULT_MAX_QUBITS})",
    )
    rand_parser.add_argument(
        "--max-shots",
        type=int,
        default=StateVectorSimulator.DEFAULT_MAX_SHOTS,
        help=f"Maximum allowed shots (default: {StateVectorSimulator.DEFAULT_MAX_SHOTS})",
    )
    rand_parser.add_argument(
        "--single-qubit-gates",
        type=str,
        nargs="*",
        default=DEFAULT_SINGLE_QUBIT_GATES,
        help="Set of single-qubit gates to sample from",
    )
    rand_parser.add_argument(
        "--two-qubit-gates",
        type=str,
        nargs="*",
        default=DEFAULT_TWO_QUBIT_GATES,
        help="Set of two-qubit gates to sample from",
    )
    rand_parser.add_argument(
        "--multi-qubit-gates",
        type=str,
        nargs="*",
        default=DEFAULT_MULTI_QUBIT_GATES,
        help="Set of multi-qubit gates (three or more qubits) to sample from",
    )

    return parser


def _cmd_simulate(args: argparse.Namespace) -> int:
    try:
        circuit = _load_circuit(args.circuit)
    except Exception as e:
        print(f"Error loading circuit: {e}", file=sys.stderr)
        return 1
    simulator = StateVectorSimulator(max_qubits=args.max_qubits, max_shots=args.max_shots)
    result = simulator.run(circuit, shots=args.shots, seed=args.seed)
    
    payload = {
        "num_qubits": circuit.num_qubits,
        "shots": result.shots,
        "final_state_real": result.final_state.real.tolist(),
        "final_state_imag": result.final_state.imag.tolist(),
        "probabilities": result.probabilities.tolist(),
        "counts": result.counts,
        "samples": result.samples,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_random_circuit(args: argparse.Namespace) -> int:
    """Execute the random-circuit command."""
    # Build the random circuit based on CLI arguments
    circuit = random_circuit(
        num_qubits=args.qubits,
        depth=args.depth,
        single_qubit_gates=args.single_qubit_gates,
        two_qubit_gates=args.two_qubit_gates,
        multi_qubit_gates=args.multi_qubit_gates,
        seed=args.seed,
    )
    simulator = StateVectorSimulator(max_qubits=args.max_qubits, max_shots=args.max_shots)
    xeb_result = run_linear_xeb_experiment(
        circuit,
        simulator=simulator,
        shots=args.shots,
        seed=args.seed,
    )

    payload = {
        "num_qubits": circuit.num_qubits,
        "depth": args.depth,
        "shots": args.shots,
        "fidelity": xeb_result.fidelity,
        "sample_probabilities": xeb_result.sample_probabilities,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _load_circuit(path: Path) -> QuantumCircuit:
    """Helper to load a circuit from a file."""
    # Delayed import to avoid circular dependencies if .run imports .cli
    from .run import load_circuit
    return load_circuit(path)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
