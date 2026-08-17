import numpy as np
import pytest
from numpy.testing import assert_array_equal

import wavepacket as wp


def test_reject_nonpositive_number_of_channels():
    with pytest.raises(wp.InvalidValueError):
        wp.grid.ChannelDof(-3)

    with pytest.raises(wp.InvalidValueError):
        wp.grid.ChannelDof(0)


def test_correctly_sized_grids():
    dof = wp.grid.ChannelDof(5)

    assert dof.size == 5
    assert_array_equal(dof.dvr_points, np.arange(5))
    assert_array_equal(dof.fbr_points, np.arange(5))


def test_transformations_are_no_ops():
    rng = np.random.default_rng(42)
    data = rng.random(6)

    dof = wp.grid.ChannelDof(6)

    assert_array_equal(dof.to_dvr(data, 0), data)
    assert_array_equal(dof.to_fbr(data, 0), data)
    assert_array_equal(dof.from_dvr(data, 0), data)
    assert_array_equal(dof.from_fbr(data, 0), data)


def test_initialize_grid_with_names():
    dof = wp.grid.ChannelDof(["ground", "first", "second"])

    assert dof.size == 3
    assert_array_equal(dof.dvr_points, np.arange(3))
    assert_array_equal(dof.fbr_points, np.arange(3))


def test_reject_empty_or_missing_names():
    with pytest.raises(wp.InvalidValueError):
        wp.grid.ChannelDof([])

    with pytest.raises(wp.InvalidValueError):
        wp.grid.ChannelDof(["a channel", ""])


def test_reject_duplicate_names():
    with pytest.raises(wp.InvalidValueError):
        wp.grid.ChannelDof(["a", "b", "a"])


def test_retrieve_names():
    names = ["a", "b", "c"]
    dof = wp.grid.ChannelDof(names)

    assert dof.names == names

    names.append("more")
    assert len(dof.names) == 3

    numeric_dof = wp.grid.ChannelDof(4)
    assert numeric_dof.names == ["0", "1", "2", "3"]


def test_get_channel_index():
    dof = wp.grid.ChannelDof(["a", "b", "c"])

    assert dof.get_index(0) == 0
    assert dof.get_index(-3) == -3
    assert dof.get_index("b") == 1

    with pytest.raises(wp.InvalidValueError):
        dof.get_index(-4)
    with pytest.raises(wp.InvalidValueError):
        dof.get_index(3)
    with pytest.raises(wp.InvalidValueError):
        dof.get_index("d")

    noname_dof = wp.grid.ChannelDof(3)
    assert noname_dof.get_index(-1) == -1
    with pytest.raises(wp.InvalidValueError):
        noname_dof.get_index("a")
