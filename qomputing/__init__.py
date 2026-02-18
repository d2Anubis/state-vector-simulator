"""Qomputing Simulator package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Gate",
    "QuantumCircuit",
    "StateVectorSimulator",
    "SimulationResult",
    "LinearXEBResult",
    "compute_linear_xeb_fidelity",
    "run_linear_xeb_experiment",
    # Backend API (Qomputing style)
    "QomputingSimulator",
    "Backend",
    "Result",
    # Library API
    "load_circuit",
    "run",
    "run_xeb",
    "random_circuit",
]

_EXPORT_MODULES = {
    "Gate": ".circuit",
    "QuantumCircuit": ".circuit",
    "StateVectorSimulator": ".engine",
    "SimulationResult": ".engine",
    "LinearXEBResult": ".xeb",
    "compute_linear_xeb_fidelity": ".xeb",
    "run_linear_xeb_experiment": ".xeb",
    "QomputingSimulator": ".backend",
    "Backend": ".backend",
    "Result": ".backend",
    "load_circuit": ".run",
    "run": ".run",
    "run_xeb": ".run",
    "random_circuit": ".run",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
