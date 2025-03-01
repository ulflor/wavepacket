"""
A package for the propagation of quantum-mechanical wave functions.
"""

__version__ = "0.1.0"

from . import builder
from . import expression
from . import grid
from . import operator
from . import solver
from . import typing


from .exceptions import (BadFunctionCall, BadGridError, BadStateError, ExecutionError,
                         InvalidValueError)
from .generators import Gaussian, PlaneWave
from .logging import log


__all__ = ['__version__', 'log', 'BadFunctionCall', 'BadGridError', 'BadStateError',
           'ExecutionError', 'InvalidValueError',
           'Gaussian', 'PlaneWave',
           'builder', 'expression', 'grid', 'operator', 'solver', 'testing', 'typing']
