import numpy as np
import pytest
import wavepacket as wp


plane_wave = wp.grid.DofType.PLANE_WAVE

def test_reject_empty_dof_list():
    with pytest.raises(wp.InvalidValueError):
        wp.grid.Grid([])


def test_require_consistent_dof_data():
    good = np.ones(4)
    bad_size = np.ones(3)
    bad_rank = np.ones((2, 2))

    good_dof = wp.grid.DegreeOfFreedom(plane_wave, good, good, good)
    bad_dvr = wp.grid.DegreeOfFreedom(plane_wave, bad_size, good, good)
    bad_fbr = wp.grid.DegreeOfFreedom(plane_wave, good, bad_size, good)
    bad_weight = wp.grid.DegreeOfFreedom(plane_wave, good, good, bad_size)
    bad_rank_dof = wp.grid.DegreeOfFreedom(plane_wave, bad_rank, bad_rank, bad_rank)

    with pytest.raises(wp.InvalidValueError):
        wp.grid.Grid([good_dof, bad_dvr])

    with pytest.raises(wp.InvalidValueError):
        wp.grid.Grid([bad_fbr, good_dof])

    with pytest.raises(wp.InvalidValueError):
        wp.grid.Grid([good_dof, bad_weight])

    with pytest.raises(wp.InvalidValueError):
        wp.grid.Grid([good_dof, bad_rank_dof])


def test_require_nonempty_data():
    empty_dof = wp.grid.DegreeOfFreedom(plane_wave, np.ones(0), np.ones(0), np.ones(0))

    with pytest.raises(wp.InvalidValueError):
        wp.grid.Grid([empty_dof])


def test_grid_can_take_dof_directly():
    dof = wp.grid.DegreeOfFreedom(plane_wave, np.ones(2), np.ones(2), np.ones(2))

    grid = wp.grid.Grid(dof)

    assert len(grid.dof) == 1
    assert grid.dof[0] == dof


def test_grid_offers_dof_as_property():
    dof_a = wp.grid.DegreeOfFreedom(plane_wave, np.ones(2), np.ones(2), np.ones(2))
    dof_b = wp.grid.DegreeOfFreedom(plane_wave, np.ones(5), np.ones(5), np.ones(5))

    grid = wp.grid.Grid([dof_a, dof_b])

    assert len(grid.dof) == 2
    assert grid.dof[0] == dof_a
    assert grid.dof[1] == dof_b


def test_shapes():
    dof_a = wp.grid.DegreeOfFreedom(plane_wave, np.ones(2), np.ones(2), np.ones(2))
    dof_b = wp.grid.DegreeOfFreedom(plane_wave, np.ones(5), np.ones(5), np.ones(5))

    grid = wp.grid.Grid([dof_a, dof_a, dof_b])

    assert grid.shape == (2, 2, 5)
    assert grid.operator_shape == (2, 2, 5, 2, 2, 5)