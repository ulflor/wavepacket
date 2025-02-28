import wavepacket as wp


def test_logging(grid_1d):
    psi = wp.builder.product_wave_function(grid_1d, wp.Gaussian(0, fwhm=2.0))

    wp.log(0.0, psi)
    wp.log(0, psi)
