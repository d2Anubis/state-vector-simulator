"""Circuit primitives for the state vector simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def _ensure_tuple(values: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in values)


@dataclass(frozen=True)
class Gate:
    """Immutable gate description."""

    name: str
    targets: Tuple[int, ...]
    controls: Tuple[int, ...] = field(default_factory=tuple)
    params: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.lower()
        object.__setattr__(self, "name", name)

        if not self.targets:
            raise ValueError("Gate must have at least one target qubit")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError(f"Duplicate target qubits in gate {self.name}")
        if any(t < 0 for t in self.targets):
            raise ValueError("Target qubit indices must be non-negative")

        if len(set(self.controls)) != len(self.controls):
            raise ValueError(f"Duplicate control qubits in gate {self.name}")
        if any(c < 0 for c in self.controls):
            raise ValueError("Control qubit indices must be non-negative")

        overlap = set(self.targets) & set(self.controls)
        if overlap:
            raise ValueError(f"Control and target sets overlap for gate {self.name}: {overlap}")


class QuantumCircuit:
    """Simple representation of a quantum circuit."""

    def __init__(self, num_qubits: int, num_clbits: int = 0) -> None:
        if num_qubits <= 0:
            raise ValueError("Circuit must have at least one qubit")
        if num_clbits < 0:
            raise ValueError("num_clbits must be non-negative")
        self.num_qubits = int(num_qubits)
        self.num_clbits = int(num_clbits)
        self._gates: List[Gate] = []
        self._measurements: List[Tuple[int, int]] = []  # (qubit, clbit) pairs

    @property
    def gates(self) -> Tuple[Gate, ...]:
        return tuple(self._gates)

    @property
    def measurements(self) -> Tuple[Tuple[int, int], ...]:
        """Return the (qubit, clbit) measurement map. [Fix: Added public property for encapsulation]"""
        return tuple(self._measurements)

    def add_gate(
        self,
        name: str,
        targets: Sequence[int],
        *,
        controls: Sequence[int] | None = None,
        params: Mapping[str, float] | None = None,
    ) -> "QuantumCircuit":
        gate = Gate(
            name=name,
            targets=_ensure_tuple(targets),
            controls=_ensure_tuple(controls or ()),
            params=dict(params or {}),
        )
        self._validate_gate_qubits(gate)
        self._gates.append(gate)
        return self

    # Convenience builders -------------------------------------------------
    def i(self, target: int) -> "QuantumCircuit":
        return self.add_gate("id", [target])

    def x(self, target: int) -> "QuantumCircuit":
        return self.add_gate("x", [target])

    def y(self, target: int) -> "QuantumCircuit":
        return self.add_gate("y", [target])

    def z(self, target: int) -> "QuantumCircuit":
        return self.add_gate("z", [target])

    def h(self, target: int) -> "QuantumCircuit":
        return self.add_gate("h", [target])

    def s(self, target: int) -> "QuantumCircuit":
        return self.add_gate("s", [target])

    def sdg(self, target: int) -> "QuantumCircuit":
        return self.add_gate("sdg", [target])

    def t(self, target: int) -> "QuantumCircuit":
        return self.add_gate("t", [target])

    def tdg(self, target: int) -> "QuantumCircuit":
        return self.add_gate("tdg", [target])

    def sx(self, target: int) -> "QuantumCircuit":
        return self.add_gate("sx", [target])

    def sxdg(self, target: int) -> "QuantumCircuit":
        return self.add_gate("sxdg", [target])

    def rx(self, target: int, theta: float) -> "QuantumCircuit":
        return self.add_gate("rx", [target], params={"theta": float(theta)})

    def ry(self, target: int, theta: float) -> "QuantumCircuit":
        return self.add_gate("ry", [target], params={"theta": float(theta)})

    def rz(self, target: int, theta: float) -> "QuantumCircuit":
        return self.add_gate("rz", [target], params={"theta": float(theta)})

    def u1(self, target: int, lam: float) -> "QuantumCircuit":
        return self.add_gate("u1", [target], params={"lambda": float(lam)})

    def u2(self, target: int, phi: float, lam: float) -> "QuantumCircuit":
        return self.add_gate("u2", [target], params={"phi": float(phi), "lambda": float(lam)})

    def u3(self, target: int, theta: float, phi: float, lam: float) -> "QuantumCircuit":
        return self.add_gate(
            "u3",
            [target],
            params={"theta": float(theta), "phi": float(phi), "lambda": float(lam)},
        )

    def cx(self, control: int, target: int) -> "QuantumCircuit":
        return self.add_gate("cx", [target], controls=[control])

    def cy(self, control: int, target: int) -> "QuantumCircuit":
        return self.add_gate("cy", [target], controls=[control])

    def cz(self, control: int, target: int) -> "QuantumCircuit":
        return self.add_gate("cz", [target], controls=[control])

    def cp(self, control: int, target: int, phi: float) -> "QuantumCircuit":
        return self.add_gate("cp", [target], controls=[control], params={"phi": float(phi)})

    def swap(self, q1: int, q2: int) -> "QuantumCircuit":
        if q1 == q2:
            raise ValueError("Swap operands must refer to distinct qubits")
        return self.add_gate("swap", [q1, q2])

    def iswap(self, q1: int, q2: int) -> "QuantumCircuit":
        if q1 == q2:
            raise ValueError("iswap operands must refer to distinct qubits")
        return self.add_gate("iswap", [q1, q2])

    def sqrt_iswap(self, q1: int, q2: int) -> "QuantumCircuit":
        if q1 == q2:
            raise ValueError("sqrtiswap operands must refer to distinct qubits")
        return self.add_gate("sqrtiswap", [q1, q2])

    def rxx(self, q1: int, q2: int, theta: float) -> "QuantumCircuit":
        if q1 == q2:
            raise ValueError("rxx operands must refer to distinct qubits")
        return self.add_gate("rxx", [q1, q2], params={"theta": float(theta)})

    def ryy(self, q1: int, q2: int, theta: float) -> "QuantumCircuit":
        if q1 == q2:
            raise ValueError("ryy operands must refer to distinct qubits")
        return self.add_gate("ryy", [q1, q2], params={"theta": float(theta)})

    def rzz(self, q1: int, q2: int, theta: float) -> "QuantumCircuit":
        if q1 == q2:
            raise ValueError("rzz operands must refer to distinct qubits")
        return self.add_gate("rzz", [q1, q2], params={"theta": float(theta)})

    def csx(self, control: int, target: int) -> "QuantumCircuit":
        return self.add_gate("csx", [target], controls=[control])

    def ccx(self, control1: int, control2: int, target: int) -> "QuantumCircuit":
        return self.add_gate("ccx", [target], controls=[control1, control2])

    def ccz(self, control1: int, control2: int, target: int) -> "QuantumCircuit":
        return self.add_gate("ccz", [target], controls=[control1, control2])

    def cswap(self, control: int, q1: int, q2: int) -> "QuantumCircuit":
        if q1 == q2:
            raise ValueError("cswap target qubits must be distinct")
        return self.add_gate("cswap", [q1, q2], controls=[control])

    def measure(
        self,
        qubits: Sequence[int],
        clbits: Sequence[int] | None = None,
    ) -> "QuantumCircuit":
        """Measure qubits into classical bits. If clbits is None, use same indices as qubits."""
        qubits = _ensure_tuple(qubits)
        if clbits is None:
            clbits = qubits
        else:
            clbits = _ensure_tuple(clbits)
        if len(qubits) != len(clbits):
            raise ValueError("qubits and clbits must have the same length")
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                raise ValueError(f"Qubit index {q} out of range for {self.num_qubits} qubits")
        for c in clbits:
            if c < 0 or c >= self.num_clbits:
                raise ValueError(f"Classical bit index {c} out of range for {self.num_clbits} classical bits")
        
        # [Bug Fix: Check for classical bit collisions to prevent accidental overwriting]
        existing_clbits = {pair[1] for pair in self._measurements}
        for c in clbits:
            if c in existing_clbits:
                raise ValueError(f"Classical bit index {c} is already being used for measurement")
            existing_clbits.add(c)

        self._measurements.extend(zip(qubits, clbits))
        return self

    def measure_all(self) -> "QuantumCircuit":
        """Measure all qubits into corresponding classical bits. [Feature: New convenience method]
        
        Requires num_clbits to be at least num_qubits.
        """
        if self.num_clbits < self.num_qubits:
            raise ValueError(
                f"Cannot measure_all: circuit has {self.num_qubits} qubits "
                f"but only {self.num_clbits} classical bits"
            )
        return self.measure(range(self.num_qubits), range(self.num_qubits))

    # Serialisation --------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "num_qubits": self.num_qubits,
            "gates": [
                {
                    "name": gate.name,
                    "targets": list(gate.targets),
                    **(
                        {"controls": list(gate.controls)}
                        if gate.controls
                        else {}
                    ),
                    **({"params": dict(gate.params)} if gate.params else {}),
                }
                for gate in self._gates
            ],
        }
        if self.num_clbits:
            payload["num_clbits"] = self.num_clbits
        if self._measurements:
            payload["measurements"] = [list(pair) for pair in self._measurements]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "QuantumCircuit":
        num_qubits = int(payload["num_qubits"])
        num_clbits = int(payload.get("num_clbits", 0))
        circuit = cls(num_qubits, num_clbits)
        for gate_payload in payload.get("gates", []):
            if not isinstance(gate_payload, Mapping):
                raise TypeError("Gate payload must be a mapping")
            circuit.add_gate(
                str(gate_payload["name"]),
                gate_payload.get("targets", ()),
                controls=gate_payload.get("controls"),
                params=gate_payload.get("params"),
            )
        for pair in payload.get("measurements", []):
            q, c = int(pair[0]), int(pair[1])
            circuit._measurements.append((q, c))
        return circuit

    # Internal helpers -----------------------------------------------------
    def _validate_gate_qubits(self, gate: Gate) -> None:
        for index in gate.targets + gate.controls:
            if index < 0 or index >= self.num_qubits:
                raise ValueError(
                    f"Qubit index {index} for gate {gate.name} "
                    f"is out of range for circuit with {self.num_qubits} qubits",
                )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"QuantumCircuit(num_qubits={self.num_qubits}, num_clbits={self.num_clbits}, gates={self._gates!r})"

