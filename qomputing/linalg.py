"""Shared tensor and linear algebra helpers."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def initial_state(num_qubits: int, *, dtype: np.dtype) -> np.ndarray:
    size = 1 << num_qubits
    state = np.zeros(size, dtype=dtype)
    state[0] = 1.0 + 0j
    return state


def axis_for_qubit(qubit: int) -> int:
    """Return tensor axis index for the given little-endian qubit."""
    return -(qubit + 1)


def reshape_state(state: np.ndarray, num_qubits: int) -> np.ndarray:
    return state.reshape((2,) * num_qubits)


def swap_axes(tensor: np.ndarray, axis_a: int, axis_b: int) -> np.ndarray:
    return np.swapaxes(tensor, axis_a, axis_b)


def select_indices(num_qubits: int, indices: Dict[int, int]) -> Tuple[slice, ...]:
    slices = [slice(None)] * num_qubits
    for axis, value in indices.items():
        slices[axis] = value
    return tuple(slices)


def _apply_tensor_op(tensor: np.ndarray, axes: List[int], matrix: np.ndarray) -> np.ndarray:
    """Apply unitary matrix to specific axes of a tensor."""
    dimension = 1 << len(axes)
    dest_axes = list(range(-len(axes), 0))
    
    # Move target axes to the end
    permutated = np.moveaxis(tensor, axes, dest_axes)
    
    # Reshape for matrix multiplication
    leading_shape = permutated.shape[:-len(axes)]
    # (..., 2^k)
    flattened = permutated.reshape(-1, dimension).T 
    
    # Apply matrix
    transformed = matrix @ flattened
    
    # Reshape back
    restored = transformed.T.reshape(*leading_shape, *([2] * len(axes)))
    
    # Move axes back to original positions
    final = np.moveaxis(restored, dest_axes, axes)
    return final


def apply_unitary(
    state: np.ndarray,
    qubits: List[int],
    matrix: np.ndarray,
    num_qubits: int,
) -> np.ndarray:
    qubits = list(qubits)
    if not qubits:
        return state
    matrix = np.asarray(matrix, dtype=state.dtype)
    dimension = 1 << len(qubits)
    if matrix.shape != (dimension, dimension):
        raise ValueError("Unitary size does not match qubit count")

    tensor = reshape_state(state, num_qubits)
    axes = [axis_for_qubit(q) for q in qubits]
    
    result_tensor = _apply_tensor_op(tensor, axes, matrix)
    return result_tensor.reshape(state.shape)


def apply_controlled_unitary(
    state: np.ndarray,
    controls: List[int],
    targets: List[int],
    matrix: np.ndarray,
    num_qubits: int,
) -> np.ndarray:
    """Apply a unitary controlled by one or more qubits."""
    if not controls:
        return apply_unitary(state, targets, matrix, num_qubits)

    tensor = reshape_state(state, num_qubits)
    
    # Construct slice to select subspace where all controls are |1>
    idx = [slice(None)] * num_qubits
    for c in controls:
        axis = axis_for_qubit(c)
        # axis is negative loop index: -1, -2...
        # Standard positive index = num_qubits + axis
        pos_axis = axis + num_qubits if axis < 0 else axis
        idx[pos_axis] = 1
        
    # Extract view of subspace
    # Note: simple indexing with integers reduces dimension.
    # The axes corresponding to controls are removed.
    # We need to map target qubits to strict axes of this sub-tensor.
    sub_tensor = tensor[tuple(idx)]
    
    # Mapping target qubits to sub-tensor axes
    # The sub-tensor has (num_qubits - len(controls)) dimensions.
    # Original axes were 0..N-1.
    # Controls were removed.
    # New axis for a target T is:
    #   original_axis(T) - (number of controls < original_axis(T))
    
    control_axes = sorted([axis_for_qubit(c) + num_qubits for c in controls])
    target_tensor_axes = []
    
    for t in targets:
        t_axis = axis_for_qubit(t) + num_qubits
        # Adjust for removed dimensions
        shift = sum(1 for c_ax in control_axes if c_ax < t_axis)
        new_axis = t_axis - shift
        # Convert back to negative indexing for consistency (optional but usually safer with dynamic dims)
        # But _apply_tensor_op handles negative logic fine if we pass negative.
        # Let's keep it positive for simplicity here, logic holds.
        target_tensor_axes.append(new_axis - sub_tensor.ndim) # convert to -X style
        
    # Apply unitary to sub-tensor
    # This creates a new array for the slice
    transformed_sub = _apply_tensor_op(sub_tensor, target_tensor_axes, matrix)
    
    # Assign back to original tensor
    tensor[tuple(idx)] = transformed_sub
    
    return tensor.reshape(state.shape)

