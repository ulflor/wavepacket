import itertools
import math
from typing import override

import scipy

import wavepacket as wp
from .solverbase import SolverBase
from ..grid import State
from ..expression import ExpressionBase
from ..operator import OperatorBase


class ChebychevSolver(SolverBase):
    """
    Solver that expands the time evolution operator into Chebychev polynomials.

    This solver is fast and accurate, but works only for time-independent closed systems, and
    requires explicit bounds for the spectrum of the Hamiltonian or Liouvillian, which is more
    setup work.

    There is a dedicated tutorial for this solver, see :doc:`/tutorials/chebychev_solvers`.

    Parameters
    ----------
    expression: wp.expression.Expression
        The Schrödinger equation or Liouvillian that describes the right-hand side of the differential equation.
    dt: float
        The time step that is propagated in one go.
    spectrum: tuple[float, float]
        Lower and upper bound of the spectrum of the Hamiltonian or Liouvillian.
        If the bound is generous, the solver is less efficient. If the spectrum extends beyond the bounds,
        the solution may diverge.

    Attributes
    ----------
    alpha
        The Kosloff alpha number "spectral range * dt / 2". Ideally, it should be around 40.
        Smaller values reduce efficiency, much larger values can cause problems with overflows.
    order
        The order of the expansion.

    Notes
    -----

    The name of this class deviates from the nowadays official transcription, which is "Chebyshev".
    The original references by Tal-Ezer and Kosloff, however, use the outdated spelling, so we tried
    to be consistent with these references.
    """

    def __init__(self, expression: ExpressionBase, dt: float,
                 spectrum: tuple[float, float]) -> None:
        super().__init__(dt)

        if spectrum[1] <= spectrum[0]:
            raise wp.InvalidValueError("Spectrum is not monotonous")
        if expression.time_dependent:
            raise wp.InvalidValueError("Polynomial solvers do not work for time-dependent operators.")

        self._expression = expression
        self._spec_min = spectrum[0]
        self._spec_range = spectrum[1] - spectrum[0]

        self.alpha = self._spec_range * self._dt / 2.0

        self._prefactor = math.e ** (-1j * (spectrum[0] + spectrum[1]) / 2.0 * dt)
        self._coeffs = [scipy.special.j0(self.alpha)]
        for n in itertools.count(1):
            c = 2 * scipy.special.jv(n, self.alpha)
            self._coeffs.append(c)

            # We fix the cutoff to 1e-12, because there is little point in varying it.
            if abs(c) < 1e-12 and n > 2:
                break

    @property
    def order(self) -> int:
        return len(self._coeffs) - 1

    @override
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


class RelaxationSolver(SolverBase):
    """
    Solver for propagation in imaginary time.

    This class is similar to the :py:class:`wp.solver.ChebychevSolver`, but in imaginary time.
    It relaxes an initial wave function to the ground state of a system or a density operator to
    the density operator of a system at finite temperature.

    There is a dedicated tutorial for this solver, see :doc:`/tutorials/relaxation`.

    Parameters
    ----------
    hamiltonian: wp.operator.OperatorBase
        The Hamiltonian operator that describes the right-hand side of the differential equation.
    dt: float
        The time step that is propagated in one go.
    spectrum: tuple[float, float]
        Lower and upper bound of the spectrum of the Hamiltonian or Liouvillian.
        If the bound is generous, the solver is less efficient. If the spectrum extends beyond the bounds,
        the solution may diverge.

    Attributes
    ----------
    alpha
        The Kosloff alpha number "spectral range * dt / 2". Ideally, it should be around 40.
        Smaller values reduce efficiency, much larger values can cause problems with overflows.
    order
        The order of the expansion.
    """

    def __init__(self, hamiltonian: OperatorBase, dt: float,
                 spectrum: tuple[float, float]) -> None:
        super().__init__(dt)

        if spectrum[1] <= spectrum[0]:
            raise wp.InvalidValueError("Spectrum is not monotonous")
        if hamiltonian.time_dependent:
            raise wp.InvalidValueError("Polynomial solvers do not work for time-dependent operators.")

        self._hamiltonian = hamiltonian
        self._spec_min = spectrum[0]
        self._spec_range = spectrum[1] - spectrum[0]

        self.alpha = self._spec_range * self._dt / 2.0

        self._prefactor = math.e ** (-(spectrum[0] + spectrum[1]) / 2.0 * dt)
        self._coeffs = [scipy.special.i0(self.alpha)]
        for n in itertools.count(1):
            c = 2 * scipy.special.iv(n, self.alpha)
            self._coeffs.append(c)

            # There is almost no use in playing with the cutoff, so we fix it at 1e-12
            if abs(c) < 1e-12 and n > 2:
                break

    @property
    def order(self) -> int:
        return len(self._coeffs) - 1

    @override
    def step(self, state: State, t: float) -> State:
        term_minus2 = state
        term_minus1 = -self._apply_normalized(state)

        result = self._coeffs[0] * term_minus2 + self._coeffs[1] * term_minus1
        for n in range(2, len(self._coeffs)):
            term = -2 * self._apply_normalized(term_minus1) - term_minus2
            result += self._coeffs[n] * term

            term_minus2 = term_minus1
            term_minus1 = term

        return self._prefactor * result

    def _apply_normalized(self, state: wp.grid.State) -> wp.grid.State:
        # result = 2.0/spec_diff * (H[input] - (spec_min + spec_diff/2) * input)
        h_input = self._hamiltonian.apply(state, t=0.0)

        factor1 = 2.0 / self._spec_range
        factor2 = (2.0 / self._spec_range * self._spec_min + 1)

        return factor1 * h_input - factor2 * state
