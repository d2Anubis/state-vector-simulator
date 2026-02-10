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
    """Sample bitstrings from a probability distribution.
    
    Args:
        probabilities: 1D array of probabilities for each computational basis state.
        num_qubits: Number of qubits in the full state.
        shots: Number of samples to draw.
        rng: NumPy random generator instance.
    """
    prob_sum = probabilities.sum()
    if prob_sum < 1e-10:
        raise ValueError("Probability sum too small; state vector may be invalid.")
    
    # Normalize probabilities to ensure they sum strictly to 1.0
    probabilities = probabilities / prob_sum

    basis_indices = rng.choice(len(probabilities), size=shots, p=probabilities)
    return [format(index, f"0{num_qubits}b") for index in basis_indices]


def counts_from_samples(samples: Iterable[str]) -> Dict[str, int]:
    """Convert a list of samples into a dictionary of counts."""
    counts: Dict[str, int] = {}
    for measurement in samples:
        counts[measurement] = counts.get(measurement, 0) + 1
    return counts


def samples_to_classical_counts(
    samples: List[str],
    num_qubits: int,
    measure_map: List[tuple[int, int]],
    num_clbits: int,
) -> Dict[str, int]:
    """Map full-state samples to classical-bit counts using (qubit, clbit) measure map.
    
    Key is MSB-first (clbit 0 = LSB on the right), e.g. '0011' for Bell |11⟩ on (q0,q1).
    
    Args:
        samples: List of bitstrings from a full state measurement.
        num_qubits: Total number of qubits in the sample strings.
        measure_map: List of mappings in the form (qubit_index, classical_bit_index).
        num_clbits: The size of the target classical register.
    """
    counts: Dict[str, int] = {}
    for sample in samples:
        # Initialize classical register with zeros
        cl_bits = ["0"] * num_clbits
        
        for q, c in measure_map:
            # sample is big-endian (q high on left). 
            # num_qubits - 1 - q converts qubit index to string position.
            cl_bits[c] = sample[num_qubits - 1 - q]  
        
        # MSB-first string: left = highest clbit, right = clbit 0 (LSB)
        key = "".join(reversed(cl_bits))
        counts[key] = counts.get(key, 0) + 1
        
    return counts


if __name__ == "__main__":
    # Example usage: Sampling a Bell State (|00> + |11>) / sqrt(2)
    # Probabilities: |00>=0.5, |01>=0, |10>=0, |11>=0.5
    probs = np.array([0.5, 0.0, 0.0, 0.5])
    gen = np.random.default_rng(42)
    
    raw_samples = sample_measurements(probs, num_qubits=2, shots=10, rng=gen)
    print(f"Raw Samples (q1q0): {raw_samples}")
    
    # Map q0 to clbit 0 and q1 to clbit 1
    mapping = [(0, 0), (1, 1)]
    cl_counts = samples_to_classical_counts(raw_samples, 2, mapping, 2)
    print(f"Classical Counts (c1c0): {cl_counts}")