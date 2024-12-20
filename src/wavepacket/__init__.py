"""
A package for the propagation of quantum-mechanical wave functions.
"""

__version__ = "0.1.0"

from . import grid
from . import utils

from .grid import *
from .utils import *

__all__ = list({'__version__'} |
               set(grid.__all__) |
               set(utils.__all__))


