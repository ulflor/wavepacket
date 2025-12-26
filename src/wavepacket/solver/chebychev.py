import itertools
import math
from typing import Tuple

import scipy

import wavepacket as wp
from .solverbase import SolverBase
from ..grid import State
from ..expression import ExpressionBase


class ChebychevSolver(SolverBase):
    """
    Solver that expands the time evolution operator into Chebychev polynomials.

    The advantage of using Chebychev polynomials is that the series converges continuously.
    That is, for a given order of the expansion, you can find a strict upper bound for the numerical error,
    no matter the initial state. This solver is about an order of magnitude faster than a generic ODE solver.

    However, there are two significant limitations.
    Only time-independent systems are supported. Time-dependence introduces time-ordering in the operator product,
    which is difficult to handle. Also, the expansion is only defined for an operator with a spectrum
    inside the interval [-1,1]. Hence, the Hamiltonian needs to be mapped to a "normalized" operator;
    this process requires the spectral range of the Hamiltonian or corresponding lower/upper bounds.
    These can be principally obtained with the power method, but doing so can be a little cumbersome.

    Parameters
    ----------
    expression: wp.expression.Expression
        The Hamiltonian / Liouvillian that describes the right-hand side of the differential equation.
    dt: float
        The time step that is propagated in one go.
    spectrum: Tuple[float, float]
        Lower and upper bound of the spectrum of the Hamiltonian or Liouvillian.
        If the bound is generous, the solver is less efficient. If the spectrum extends outside of the bounds,
        the solution will diverge.
    cutoff: float, default=1e-12
        The smallest coefficient that is summed up. Basically an upper bound of the numerical error.

    References
    ----------
    .. [1] https://sourceforge.net/p/wavepacket/cpp/blog/2021/04/convergence-2
    """

    def __init__(self, expression: ExpressionBase, dt: float,
                 spectrum: Tuple[float, float], cutoff: float = 1e-12):
        super().__init__(dt)

        if spectrum[1] <= spectrum[0]:
            raise wp.InvalidValueError("Spectrum is not monotonous")
        if cutoff <= 0.0:
            raise wp.InvalidValueError("Cutoff must be positive")
        if expression.time_dependent:
            raise wp.InvalidValueError("Polynomial solvers do not work for time-dependent operators.")

        self._expression = expression
        self._spec_min = spectrum[0]
        self._spec_range = spectrum[1] - spectrum[0]

        self.alpha = self._spec_range * self._dt / 2.0

        # Setup coefficients and prefactor
        self._prefactor = math.e ** (-1j * (spectrum[0] + spectrum[1]) / 2.0 * dt)
        self._coeffs = [scipy.special.j0(self.alpha)]
        for n in itertools.count(1):
            c = 2 * scipy.special.jv(n, self.alpha)
            self._coeffs.append(c)

            if abs(c) < cutoff and n > 2:
                break

    @property
    def order(self):
        return len(self._coeffs) - 1

    def step(self, state: State, t: float) -> State:
        # Note that we solve a differential equation of the form
        # dX/dt = L[X]
        # with X the density operator or wave function, L[] some linear operator ("expression").
        # The usual Schrödinger equation has an additional "i" on the left side, though.
        term_minus2 = state
        term_minus1 = self._apply_normalized(state)

        result = self._coeffs[0] * term_minus2 + self._coeffs[1] * term_minus1
        for n in range(2, len(self._coeffs)):
            term = 2 * self._apply_normalized(term_minus1) + term_minus2
            result += self._coeffs[n] * term

            term_minus2 = term_minus1
            term_minus1 = term

        return self._prefactor * result

    def _apply_normalized(self, state: wp.grid.State) -> wp.grid.State:
        # result = 2.0/spec_diff * (H[input] - (spec_min + spec_diff/2) * input)
        # Common expressions add a "-1j", so we need to do the same to the second factor for consistency.
        h_input = self._expression.apply(state, t=0.0)

        factor1 = 2.0 / self._spec_range
        factor2 = -1j * (2.0 / self._spec_range * self._spec_min + 1)

        return factor1 * h_input - factor2 * state
