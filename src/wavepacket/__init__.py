"""
A package for the propagation of quantum-mechanical wave functions.
"""

__version__ = "0.1.0"

__all__ = ['__version__', 'log', 'BadFunctionCall', 'BadGridError', 'BadStateError',
           'ExecutionError', 'InvalidValueError',
           'SinSquare', 'SoftRectangularFunction',
           'Gaussian', 'PlaneWave', 'SphericalHarmonic',
           'builder', 'expression', 'grid', 'operator', 'solver', 'testing', 'typing']

from . import builder
from . import expression
from . import grid
from . import operator
from . import solver
from . import testing
from . import typing


from .exceptions import (BadFunctionCall, BadGridError, BadStateError, ExecutionError,
                         InvalidValueError)
from .functions import SinSquare, SoftRectangularFunction
from .generators import Gaussian, PlaneWave, SphericalHarmonic
from .logging import log
