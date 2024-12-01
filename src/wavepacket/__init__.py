"""
A package for the propagation of quantum-mechanical wave functions.
"""

__version__ = "0.1.0"

__all__ = ["__version__",
           "BadStateError",
           "InvalidValueError",
           "State"]

from . import grid
from .exceptions import *
from .state import State