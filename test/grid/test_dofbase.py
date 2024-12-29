import numpy as np
import pytest
from numpy.testing import assert_array_equal

import wavepacket as wp
from wavepacket.testing import DummyDof


def test_reject_multidimensional_arrays():
    good_array = np.ones(5)
    bad_array = np.ones([2, 3])

    with pytest.raises(wp.InvalidValueError):
        DummyDof(bad_array, good_array)

    with pytest.raises(wp.InvalidValueError):
        DummyDof(good_array, bad_array)

    # no exception
    DummyDof(good_array, good_array)


def test_reject_empty_arrays():
    empty = np.ones(0)
    good_array = np.ones(5)

    with pytest.raises(wp.InvalidValueError):
        DummyDof(empty, good_array)

    with pytest.raises(wp.InvalidValueError):
        DummyDof(good_array, empty)


def test_reject_different_array_sizes():
    with pytest.raises(wp.InvalidValueError):
        DummyDof(np.ones(3), np.ones(4))


def test_access_properties():
    dvr_array = np.ones(5)
    fbr_array = np.zeros(5)

    dof = DummyDof(dvr_array, fbr_array)

    assert dof.size == 5
    assert_array_equal(dof.dvr_points, dvr_array)
    assert_array_equal(dof.fbr_points, fbr_array)

    # Always return copies of the internal arrays.
    dof.dvr_points[0] = 5
    assert_array_equal(dof.dvr_points, dvr_array)
