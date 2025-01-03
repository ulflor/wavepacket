"""
A package for the propagation of quantum-mechanical wave functions.
"""

__version__ = "0.1.0"

from . import builder
from . import expression
from . import grid
from . import operator
from . import solver
from . import utils

from .builder import *
from .expression import *
from .grid import *
from .operator import *
from .solver import *
from .utils import *

from .logging import log

# not imported into the top-level namespace.
# It seems to make sense to use a separate package for that.
from . import typing

__all__ = list({'__version__', 'log'} |
               set(builder.__all__) |
               set(expression.__all__) |
               set(grid.__all__) |
               set(operator.__all__) |
               set(solver.__all__) |
               set(utils.__all__))
