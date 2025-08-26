import math

from .exceptions import InvalidValueError


class SinSquare:
    """
    Shape function for a squared sinusoidal laser pulse.

    Usually, you use this function to describe a smooth laser pulse
    with a definite start and end point (which a Gaussian does not have).
    The exact shape is :math:`cos^2(\\pi \\frac{t - t_0}{2 \\Delta}`, and
    zero outside the interval :math:`[t_0-\\Delta, t_0+\\Delta]`.

    Parameters
    ----------
    t0: float
        The center of the pulse.
    half_width
        The half-width of the laser pulse, :math:`\\Delta`.

    Raises
    ------
    wp.InvalidValueException
        if the half-width is not positive.
    """

    def __init__(self, t0: float, half_width: float):
        if half_width <= 0:
            raise InvalidValueError(f"Half width '{half_width}' must be positive.")

        self._t0 = t0
        self._half_width = half_width
        self._scale = math.pi / (2 * half_width)

    def __call__(self, t: float) -> float:
        dt = abs(t - self._t0)
        if dt < self._half_width:
            return math.cos(self._scale * dt)
        else:
            return 0.0


class SoftRectangularFunction:
    """
    Shape function for a rectangular pulse with soft (cosine) turn-on.

    This shape function is a rectangular function with a given half-width,
    with an added soft turn-on and turn-off of the form
    :math:`cos(\\frac{t}{B})` with the border width B.

    Parameters
    ----------
    t0: float
        Center of the rectangular pulse.
    half_width: float
        Half-width of the rectangular part of the pulse.
    border: float = half_width/10
        Optional width of the turn-on / turn-off region.

    Raises
    ------
    wp.InvalidValueError
        If the half-width or the border region is not positive.
    """

    def __init__(self, t0: float, half_width: float, border: float = None):
        if border is None:
            border = half_width / 10

        if half_width <= 0 or border <= 0:
            raise InvalidValueError(f"Half-width '{half_width}' and soft border '{border}'must be positive")

        self._scale = math.pi / (2 * border)
        self._min = t0 - half_width - border
        self._rect_min = self._min + border
        self._rect_max = self._rect_min + 2 * half_width
        self._max = self._rect_max + border

    def __call__(self, t: float) -> float:
        if t <= self._min:
            return 0.0
        elif t <= self._rect_min:
            return math.cos(self._scale * (self._rect_min - t))
        elif t <= self._rect_max:
            return 1.0
        elif t <= self._rect_max:
            return math.cos(self._scale * (t - self._rect_max))
        else:
            return 0.0
