import numpy as np
import wavepacket as wp
import pytest

from numpy.testing import assert_allclose


def test_reject_multidimensional_grids():
    good_array = np.ones(5)
    bad_array = np.ones([2, 3])

    with pytest.raises(wp.InvalidValueError):
        wp.DofBase(bad_array, good_array)

    with pytest.raises(wp.InvalidValueError):
        wp.DofBase(good_array, bad_array)

    # no exception
    wp.DofBase(good_array, good_array)


def test_reject_empty_grids():
    empty = np.ones(0)
    good_array = np.ones(5)

    with pytest.raises(wp.InvalidValueError):
        wp.DofBase(empty, good_array)

    with pytest.raises(wp.InvalidValueError):
        wp.DofBase(good_array, empty)


def test_access_properties():
    dvr_array = np.ones(5)
    fbr_array = np.zeros(4)

    dof = wp.DofBase(dvr_array, fbr_array)

    assert_allclose(dof.dvr_array, dvr_array)
    assert_allclose(dof.fbr_array, fbr_array)