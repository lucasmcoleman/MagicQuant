"""Unit tests for magicquant.imatrix._reduce_per_expert_imatrix.

Pure numpy in/out (no GGUF I/O, no llama-imatrix binary, no `gguf` pip
package) so these run in any environment; the GGUF-round-trip side of
load_imatrix's 2D-shape detection is covered by test_imatrix.py (needs the
`gguf` package to author fixtures) and was additionally verified directly
against a real llama-imatrix capture over gpt-oss-20b during development.
"""
import numpy as np

from magicquant.imatrix import _reduce_per_expert_imatrix


def test_all_experts_visited_divides_elementwise():
    sum2 = np.array([[4.0, 8.0], [30.0, 60.0], [9.0, 18.0]], dtype=np.float32)
    counts = np.array([[2.0], [10.0], [3.0]], dtype=np.float32)
    result = _reduce_per_expert_imatrix("w", sum2, counts)
    expected = np.array([2.0, 4.0, 3.0, 6.0, 3.0, 6.0], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_unvisited_expert_filled_with_mean_of_visited():
    # Expert 1 has count=0 (never routed to during the calibration corpus).
    sum2 = np.array([[10.0, 20.0], [999.0, 999.0], [30.0, 40.0]], dtype=np.float32)
    counts = np.array([[2.0], [0.0], [2.0]], dtype=np.float32)
    result = _reduce_per_expert_imatrix("w", sum2, counts).reshape(3, 2)
    # Visited experts: [5,10] and [15,20] -> mean [10, 15].
    np.testing.assert_allclose(result[0], [5.0, 10.0])
    np.testing.assert_allclose(result[1], [10.0, 15.0])
    np.testing.assert_allclose(result[2], [15.0, 20.0])


def test_no_experts_visited_returns_none():
    sum2 = np.zeros((2, 4), dtype=np.float32)
    counts = np.zeros((2, 1), dtype=np.float32)
    assert _reduce_per_expert_imatrix("w", sum2, counts) is None


def test_returns_flat_expert_major_vector():
    sum2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    counts = np.array([[1.0], [1.0]], dtype=np.float32)
    result = _reduce_per_expert_imatrix("w", sum2, counts)
    assert result.shape == (6,)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
