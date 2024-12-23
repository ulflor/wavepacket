import numpy as np
import wavepacket as wp
import pytest

from numpy.testing import assert_allclose
from wavepacket.typing import ComplexData, RealData


class DummyDof(wp.DofBase):
    def __init__(self, dvr_array: RealData, fbr_array: RealData):
        super().__init__(dvr_array, fbr_array)

    def from_fbr(self, data: ComplexData, index: int, is_ket: bool = True) -> ComplexData:
        return data

    def to_dvr(self, data: ComplexData, index: int) -> ComplexData:
        return data

    def from_dvr(self, data: ComplexData, index: int) -> ComplexData:
        return data

    def to_fbr(self, data: ComplexData, index: int, is_ket: bool = True) -> ComplexData:
        return data


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
    assert_allclose(dof.dvr_array, dvr_array)
    assert_allclose(dof.fbr_array, fbr_array)
