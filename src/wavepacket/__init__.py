"""
A package for solving Schroedinger and Liouville von Neumann equations.
"""

__version__ = "0.4.0"

__all__ = ['__version__', 'log', 'BadFunctionCall', 'BadGridError', 'BadStateError',
           'ExecutionError', 'InvalidValueError',
           'SinSquare', 'SoftRectangularFunction',
           'Gaussian', 'PlaneWave', 'SphericalHarmonic',
           'builder', 'expression', 'grid', 'operator', 'solver', 'testing', 'typing']

# order matters, because subpackages depend on each other at times.
# Though that should only matter for typing, so we could choose an alphabetic order again
# once we drop support for Python < 3.13 where type annotation are stored as strings, not types.
from . import typing
from . import grid
from . import builder
from . import operator
from . import expression
from . import solver
from . import plot
from . import testing

from ._utilities import log

from .exceptions import (BadFunctionCall, BadGridError, BadStateError, ExecutionError,
                         InvalidValueError)
from .functions import SinSquare, SoftRectangularFunction
from .generators import Gaussian, PlaneWave, SphericalHarmonic
