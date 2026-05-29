import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest
import wavepacket as wp
from wavepacket.testing import assert_close


def test_channel_rejects_invalid_arguments(grid_1d):
    channel_grid = wp.grid.Grid(wp.grid.ChannelDof(["a", "b", "c"]))

    with pytest.raises(wp.BadGridError):
        wp.operator.Channel(grid_1d, 0)

    wp.operator.Channel(channel_grid, -3)
    with pytest.raises(wp.InvalidValueError):
        wp.operator.Channel(channel_grid, -4)

    wp.operator.Channel(channel_grid, 2)
    with pytest.raises(wp.InvalidValueError):
        wp.operator.Channel(channel_grid, 3)

    with pytest.raises(wp.InvalidValueError):
        wp.operator.Channel(channel_grid, "d")


def test_coupling_rejects_invalid_arguments(grid_1d):
    channel_grid = wp.grid.Grid(wp.grid.ChannelDof(["a", "b", "c"]))

    with pytest.raises(wp.BadGridError):
        wp.operator.Coupling(grid_1d, 0, 1)

    with pytest.raises(wp.InvalidValueError):
        # from != to
        wp.operator.Coupling(channel_grid, -2, 1)

    with pytest.raises(wp.InvalidValueError):
        wp.operator.Coupling(channel_grid, -4, 1)
    with pytest.raises(wp.InvalidValueError):
        wp.operator.Coupling(channel_grid, 3, 1)
    with pytest.raises(wp.InvalidValueError):
        wp.operator.Coupling(channel_grid, "d", 1)

    with pytest.raises(wp.InvalidValueError):
        wp.operator.Coupling(channel_grid, "a", -4)
    with pytest.raises(wp.InvalidValueError):
        wp.operator.Coupling(channel_grid, "a", 3)
    with pytest.raises(wp.InvalidValueError):
        wp.operator.Coupling(channel_grid, "a", "x")


def test_operators_are_time_independent():
    channel_grid = wp.grid.Grid(wp.grid.ChannelDof(3))

    op = wp.operator.Channel(channel_grid, 0)
    assert not op.time_dependent

    op = wp.operator.Coupling(channel_grid, 0, 1)
    assert not op.time_dependent


def test_channel_apply_to_wave_function():
    channel_grid = wp.grid.Grid(
        [wp.grid.PlaneWaveDof(1, 2, 3), wp.grid.ChannelDof(4), wp.grid.PlaneWaveDof(1, 2, 5)]
    )
    op = wp.operator.Channel(channel_grid, 2)

    psi = wp.testing.random_state(channel_grid, 42)
    got = op.apply_to_wave_function(psi.data, 0.0)
    test_shape = (got.shape[0], got.shape[2])

    assert_array_equal(got[:, 0, :], np.zeros(test_shape))
    assert_array_equal(got[:, 1, :], np.zeros(test_shape))
    assert_array_equal(got[:, 2, :], psi.data[:, 2, :])
    assert_array_equal(got[:, 3, :], np.zeros(test_shape))


def test_channel_apply_to_density_operator():
    channel_grid = wp.grid.Grid(
        [wp.grid.PlaneWaveDof(1, 2, 3), wp.grid.ChannelDof(4), wp.grid.PlaneWaveDof(1, 2, 5)]
    )
    op = wp.operator.Channel(channel_grid, 2)

    ket = wp.testing.random_state(channel_grid, 42)
    bra = wp.testing.random_state(channel_grid, 43)
    prj_ket = op.apply(ket, 0.0)
    prj_bra = op.apply(bra, 0.0)

    rho = wp.builder.direct_product(ket, bra)
    prj_left = op.apply_from_left(rho.data, 0.0)
    prj_right = op.apply_from_right(rho.data, 0.0)

    assert_allclose(prj_left, wp.builder.direct_product(prj_ket, bra).data, atol=1e-12, rtol=0)
    assert_allclose(
        prj_right, wp.builder.direct_product(ket, prj_bra).data, atol=1e-12, rtol=0
    )


def test_reference_channel_by_name_or_index():
    channel_dof = wp.grid.ChannelDof(["a", "b", "c", "d"])
    grid = wp.grid.Grid(channel_dof)
    psi = wp.testing.random_state(grid, 42)

    op_index = wp.operator.Channel(grid, 1)
    op_name = wp.operator.Channel(grid, "b")

    assert_close(op_index.apply(psi, 0.0), op_name.apply(psi, 0.0), 1e-12)


def test_coupling_apply_to_wave_function():
    channel_grid = wp.grid.Grid(
        [wp.grid.PlaneWaveDof(1, 2, 3), wp.grid.ChannelDof(4), wp.grid.PlaneWaveDof(1, 2, 5)]
    )
    op = wp.operator.Coupling(channel_grid, 1, -1)

    psi = wp.testing.random_state(channel_grid, 42)
    got = op.apply_to_wave_function(psi.data, 0.0)
    test_shape = (got.shape[0], got.shape[2])

    assert_array_equal(got[:, 0, :], np.zeros(test_shape))
    assert_array_equal(got[:, 1, :], psi.data[:, -1, :])
    assert_array_equal(got[:, 2, :], np.zeros(test_shape))
    assert_array_equal(got[:, 3, :], psi.data[:, 1, :])


def test_coupling_apply_to_density_operator():
    channel_grid = wp.grid.Grid(
        [wp.grid.PlaneWaveDof(1, 2, 3), wp.grid.ChannelDof(4), wp.grid.PlaneWaveDof(1, 2, 5)]
    )
    op = wp.operator.Coupling(channel_grid, 1, -1)

    ket = wp.testing.random_state(channel_grid, 42)
    bra = wp.testing.random_state(channel_grid, 43)
    rho = wp.builder.direct_product(ket, bra)

    cpl_ket = op.apply(ket, 0.0)
    cpl_bra = op.apply(bra, 0.0)
    cpl_left = op.apply_from_left(rho.data, 0.0)
    cpl_right = op.apply_from_right(rho.data, 0.0)

    assert_allclose(cpl_left, wp.builder.direct_product(cpl_ket, bra).data, atol=1e-12, rtol=0)
    assert_allclose(
        cpl_right, wp.builder.direct_product(ket, cpl_bra).data, atol=1e-12, rtol=0
    )
