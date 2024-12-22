"""
A package for the propagation of quantum-mechanical wave functions.
"""

__version__ = "0.1.0"

from . import grid
from . import utils

from .grid import *
from .utils import *

# not imported into the upper namespace. It seems to make sense to use a separate package for that.
from . import typing

__all__ = list({'__version__'} |
               set(grid.__all__) |
               set(utils.__all__))


