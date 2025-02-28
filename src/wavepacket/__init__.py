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
from . import utils


from .exceptions import (BadFunctionCall, BadGridError, BadStateError, ExecutionError,
                         InvalidValueError)
from .logging import log


__all__ = ['__version__', 'log', 'BadFunctionCall', 'BadGridError', 'BadStateError',
           'ExecutionError', 'InvalidValueError',
           'builder', 'expression', 'grid', 'operator', 'solver', 'typing', 'utils']
