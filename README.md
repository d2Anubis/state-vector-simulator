# Qomputing Simulator

A lightweight, pure Python state-vector simulator with linear cross-entropy benchmarking (XEB). For contributors and users who install from source.

---

## Install

From the project root:

```bash
git clone <this-repo-url>
cd simulator   # or your clone directory
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

- `[dev]` adds `pytest`. For a normal (non-editable) install: `pip install .`

Without cloning (if the package is published):

```bash
pip install qomputing
# Optional: pip install qomputing[dev]
```

---

## Usage

### CLI

```bash
# Random circuit + XEB
qomputing-sim random-circuit --qubits 3 --depth 5 --shots 1000

# Simulate from JSON
qomputing-sim simulate --circuit path/to/circuit.json --shots 512 --seed 42
```

Optional: `--max-qubits`, `--max-shots` to cap resources; `--single-qubit-gates`, `--two-qubit-gates`, `--multi-qubit-gates` to override default gate sets.

### Library

**Backend API:**

```python
from qomputing import QomputingSimulator, QuantumCircuit

backend = QomputingSimulator.get_backend("state_vector")
qc = QuantumCircuit(4, 4)
qc.h(0).cx(0, 1).measure(range(3), range(3))
result = backend.run(qc, shots=1024)
print(result.get_counts())
```

**Direct run:**

```python
from qomputing import run, QuantumCircuit, run_xeb, random_circuit

qc = QuantumCircuit(2)
qc.h(0).cx(0, 1)
result = run(qc, shots=1000, seed=42)
print(result.final_state, result.probabilities, result.counts)

qc = random_circuit(num_qubits=3, depth=5, seed=7)
xeb = run_xeb(qc, shots=1000, seed=7)
print(xeb.fidelity, xeb.sample_probabilities)
```

Rotation gates use **qubit first, then angle**: e.g. `qc.rz(qubit, theta)`, `qc.ry(qubit, theta)`.  
Measure takes **sequences**: e.g. `qc.measure([0], [0])` for one qubit; `qc.measure_all()` if `num_clbits >= num_qubits`.

---

## Repository layout

```
qomputing/
├── circuit.py           # QuantumCircuit, gates, measure, JSON (de)serialization
├── circuit_builders.py   # random_circuit for XEB
├── gates/                # Single-, two-, multi-qubit gate implementations
├── engine/               # State vector core, measurements, registry
├── linalg.py             # Tensor / linear algebra helpers
├── cli.py                # qomputing-sim entry point
├── xeb.py                # Linear XEB fidelity
├── backend.py            # QomputingSimulator, Result
├── run.py                # run(), load_circuit(), run_xeb()
├── visualisation*.py     # Circuit drawer and Bloch (optional)
tools/
├── cirq_comparison.py    # Parity tests vs Cirq
tests/                    # Pytest: parity, limits, XEB, drawer, bugs
```

---

## Circuit JSON format

```json
{
  "num_qubits": 2,
  "gates": [
    {"name": "h", "targets": [0]},
    {"name": "cx", "controls": [0], "targets": [1]},
    {"name": "rz", "targets": [0], "params": {"theta": 1.5708}}
  ]
}
```

Supported gates: single-qubit `id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg`, `rx`, `ry`, `rz`, `u1`, `u2`, `u3`; two-qubit `cx`, `cy`, `cz`, `cp`, `csx`, `swap`, `iswap`, `sqrtiswap`, `rxx`, `ryy`, `rzz`; multi-qubit `ccx`, `ccz`, `cswap`.

---

## Testing

```bash
pytest
```

Parity vs Cirq (optional):

```bash
export PYTHONPATH="$PWD"
python tools/cirq_comparison.py --min-qubits 1 --max-qubits 5 --depths 3 5 --circuits-per-config 3 --shots 256 --seed 7
```

---

## Development

- Run `pytest` before committing.
- Use `CONTRIBUTING.md` for contribution guidelines (if present).
- Build package: `python -m build` (creates `dist/`). For publishing to PyPI, use `./scripts/publish-to-pypi.sh` (maintainer only).

---

## License

MIT
