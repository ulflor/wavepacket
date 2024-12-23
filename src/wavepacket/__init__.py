"""
A package for the propagation of quantum-mechanical wave functions.
"""

__version__ = "0.1.0"

from . import builder
from . import grid
from . import utils

from .builder import *
from .grid import *
from .utils import *

# not imported into the upper namespace. It seems to make sense to use a separate package for that.
from . import typing

__all__ = list({'__version__'} |
               set(builder.__all__) |
               set(grid.__all__) |
               set(utils.__all__))
