from typing import Sequence

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import wavepacket as wp


def build_grid(sizes: Sequence[int]) -> wp.grid.Grid:
    dofs = [wp.grid.PlaneWaveDof(1, 2, n) for n in sizes]
    return wp.grid.Grid(dofs)


def test_require_at_least_one_dof():
    with pytest.raises(wp.InvalidValueError):
        # noinspection PyTypeChecker
        wp.grid.Grid(None)


def test_set_and_access_dofs():
    dof1 = wp.grid.PlaneWaveDof(10, 20, 10)
    dof2 = wp.grid.PlaneWaveDof(0, 10, 5)

    grid = wp.grid.Grid([dof1, dof2])

    assert grid.dofs == [dof1, dof2]


def test_size_of_grid():
    grid = wp.grid.Grid([wp.grid.PlaneWaveDof(1, 2, 3),
                         wp.grid.PlaneWaveDof(1, 2, 10),
                         wp.grid.PlaneWaveDof(1, 2, 4)])

    assert grid.size == 3 * 10 * 4


def test_shapes():
    shape = (7, 5, 3)
    grid = build_grid(shape)

    assert grid.shape == shape
    assert grid.operator_shape == (7, 5, 3, 7, 5, 3)


def test_indices():
    grid = build_grid([1, 2, 3, 1, 2])

    assert grid.normalize_index(4) == 4
    assert grid.normalize_index(-1) == 4

    with pytest.raises(IndexError):
        grid.normalize_index(5)

    with pytest.raises(IndexError):
        grid.normalize_index(-6)


def test_broadcast():
    shape = (7, 5, 3)
    data = [i + np.arange(shape[i]) for i in range(3)]
    grid = build_grid(shape)

    expected = [np.reshape(data[0], (7, 1, 1)),
                np.reshape(data[1], (1, 5, 1)),
                np.reshape(data[2], (1, 1, 3))]

    assert_array_equal(grid.broadcast(data[0], -3), expected[0])
    assert_array_equal(grid.broadcast(data[1], 1), expected[1])
    assert_array_equal(grid.broadcast(data[2], 2), expected[2])

    with pytest.raises(IndexError):
        grid.broadcast(data[0], 3)


def test_operator_broadcast():
    shape = (6, 4, 2)
    data = np.arange(4)
    grid = build_grid(shape)

    result = grid.operator_broadcast(data, 1)
    assert result.shape == (1, 4, 1, 1, 1, 1)
    assert_array_equal(np.ravel(result), data)

    result2 = grid.operator_broadcast(data, -2)
    assert_array_equal(result, result2)

    bra_result = grid.operator_broadcast(data, 1, is_ket=False)
    assert bra_result.shape == (1, 1, 1, 1, 4, 1)
    assert_array_equal(np.ravel(result), data)

    bra_result2 = grid.operator_broadcast(data, -2, is_ket=False)
    assert_array_equal(bra_result, bra_result2)


def test_bad_broadcast():
    shape = (4, 3, 2)
    data = np.arange(5)
    grid = build_grid(shape)

    with pytest.raises(ValueError):
        grid.broadcast(data, 0)

    with pytest.raises(ValueError):
        grid.operator_broadcast(data, 0)
