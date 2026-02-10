import pytest

from qomputing.xeb import compute_linear_xeb_fidelity


def test_compute_linear_xeb_fidelity_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        compute_linear_xeb_fidelity([1.0], [])


def test_compute_linear_xeb_fidelity_rejects_mismatched_sample_width() -> None:
    with pytest.raises(ValueError, match="same bit-width"):
        compute_linear_xeb_fidelity([0.25, 0.25, 0.25, 0.25], ["00", "0"])


def test_compute_linear_xeb_fidelity_rejects_non_binary_samples() -> None:
    with pytest.raises(ValueError, match="binary strings"):
        compute_linear_xeb_fidelity([0.25, 0.25, 0.25, 0.25], ["00", "2a"])


def test_compute_linear_xeb_fidelity_rejects_probability_vector_size_mismatch() -> None:
    with pytest.raises(ValueError, match="vector size"):
        compute_linear_xeb_fidelity([0.5, 0.5], ["00", "01"])


def test_compute_linear_xeb_fidelity_valid_input() -> None:
    fidelity = compute_linear_xeb_fidelity([0.1, 0.2, 0.3, 0.4], ["00", "01", "10", "11"])
    assert fidelity == pytest.approx(0.0)
