import numpy as np
import pytest
import wavepacket as wp


def test_reject_empty_dof_list():
    with pytest.raises(wp.InvalidValueError):
        wp.grid.Grid([])

def test_require_consistent_dof_data():
    a1 = np.ones(3)
    a2 = np.ones(4)

    with pytest.raises(wp.InvalidValueError):
        wp.grid.Grid([(a1, a1, a1), (a1, a2, a1)])

    with pytest.raises(wp.InvalidValueError):
        wp.grid.Grid([(a1, a1, a2)])

    # this should work, though; we flatten dimensions later
    wp.grid.Grid([(a2, a2, np.reshape(a2, (2, 2)))])

def test_require_nonempty_data():
    a = np.empty(0)
    b = np.ones(1)
    with pytest.raises(wp.InvalidValueError):
        wp.grid.Grid([(b, b, b), (a, a, a)])
