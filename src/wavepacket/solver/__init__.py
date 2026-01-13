"""
This module contains classes to solve the equations of motion set up by the
classes in the :py:mod:`wavepacket.expression` module.
"""
__all__ = ['ChebychevSolver', 'OdeSolver', 'RelaxationSolver', 'SolverBase',
           'diagonalize']

from .chebychev import ChebychevSolver, RelaxationSolver
from .odesolver import OdeSolver
from .solverbase import SolverBase
from .tise import diagonalize
