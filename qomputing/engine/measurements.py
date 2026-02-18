"""Measurement sampling utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def sample_measurements(
    probabilities: np.ndarray,
    num_qubits: int,
    shots: int,
    rng: np.random.Generator,
) -> List[str]:
    prob_sum = probabilities.sum()
    if prob_sum < 1e-10:
        raise ValueError("Probability sum too small; state vector may be invalid.")
    probabilities = probabilities / prob_sum
    basis_indices = rng.choice(len(probabilities), size=shots, p=probabilities)
    return [format(index, f"0{num_qubits}b") for index in basis_indices]


def counts_from_samples(samples: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for measurement in samples:
        counts[measurement] = counts.get(measurement, 0) + 1
    return counts


def samples_to_classical_counts(
    samples: List[str],
    num_qubits: int,
    measure_map: List[tuple],
    num_clbits: int,
) -> Dict[str, int]:
    """Map full-state samples to classical-bit counts using (qubit, clbit) measure map.
    Key is MSB-first (clbit 0 = LSB on the right), e.g. '0011' for Bell |11⟩ on (q0,q1).
    """
    counts: Dict[str, int] = {}
    for sample in samples:
        cl_bits = ["0"] * num_clbits
        for q, c in measure_map:
            cl_bits[c] = sample[num_qubits - 1 - q]  # sample is big-endian (q high on left)
        # MSB-first string: left = highest clbit, right = clbit 0 (LSB)
        key = "".join(reversed(cl_bits))
        counts[key] = counts.get(key, 0) + 1
    return counts

