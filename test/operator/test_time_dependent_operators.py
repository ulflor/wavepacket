from numpy.testing import assert_allclose

import wavepacket as wp


def test_time_dependent_operator():
    dof = wp.grid.PlaneWaveDof(1, 2, 3)
    grid = wp.grid.Grid(dof)
    op = wp.operator.TimeDependentOperator(grid, lambda t: 2j * t)

    psi = wp.testing.random_state(grid, 47)
    rho = wp.builder.pure_density(psi)
    psi_result = op.apply(psi, 5)

    wp.testing.assert_close(10j * psi, psi_result, 1e-12)

    ket_result = op.apply_from_left(rho.data, 5)
    ket_expected = wp.builder.direct_product(psi_result, psi).data
    assert_allclose(ket_expected, ket_result, rtol=0, atol=1e-12)

    bra_result = op.apply_from_right(rho.data, 5)
    bra_expected = wp.builder.direct_product(psi, psi_result).data
    assert_allclose(bra_expected, bra_result, rtol=0, atol=1e-12)


def test_real_valued_function():
    dof = wp.grid.PlaneWaveDof(1, 2, 3)
    grid = wp.grid.Grid(dof)
    op = wp.operator.TimeDependentOperator(grid, lambda t: 5*t)

    psi = wp.testing.random_state(grid, 48)
    rho = wp.builder.pure_density(psi)

    psi_result = op.apply_to_wave_function(psi.data, 5)
    ket_result = op.apply_from_left(rho.data, 5)
    bra_result = op.apply_from_right(rho.data, 5)

    assert_allclose(psi.data * 25, psi_result, rtol=0, atol=1e-12)
    assert_allclose(ket_result, bra_result, rtol=0, atol=1e-12)
