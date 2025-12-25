from typing import Optional

import numpy as np

import wavepacket.typing as wpt


def clip_real(data: wpt.ComplexData,
              lower: Optional[float] = None, upper: Optional[float] = None) -> wpt.ComplexData:
    """
    Clips only the real part of a potentially complex-valued array.

    The background is that operators, for example potentials,
    can have real and imaginary values, where only the real values should be clipped.

    Parameters
    ----------
    data: wpt.ComplexData
        The data to be clipped
    lower: Optional[float], default=None
        The lower bound of the clipping interval. Default ist not to clip from below.
    upper: Optional[float], default=None
        The upper bound of the clipping interval. Default is not to clip from above

    Returns
    -------
    wpt.ComplexData
        The input data with the real part clipped accordingly.
        If the input data is real (data.dtype == "real floating"), this function forwards to numpy.clip().
    """

    if np.isdtype(data.dtype, 'real floating'):
        return np.clip(data, lower, upper)
    else:
        imag_data = np.imag(data)
        real_data = np.real(data)

        return np.clip(real_data, lower, upper) + 1j * imag_data
