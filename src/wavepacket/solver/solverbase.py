from abc import ABC, abstractmethod

import wavepacket as wp
from ..grid import State


class SolverBase(ABC):
    """
    Abstract base class for all differential equation solvers.

    Each solver must take in the constructor the size of an
    elementary time step, and it must implement a method :py:meth:`step`
    to evolve a state by one elementary time step forward in time.

    Usually, a solver does not need to know what step it evolves or
    how a differential equation looks in detail. The form of the
    differential equation is encapsulated in a
    :py:class:`wavepacket.expression.ExpressionBase` instance that
    is usually supplied to the specific implementations.

    A solver may, however, implicitly assume properties of the
    differential equation. It may also take additional arguments,
    these details are documented in the specific implementations.

    Parameters
    ----------
    dt: float
        The size of an elementary time step.


    Attributes
    ----------
    dt

    Raises
    ------
    wp.InvalidValueError
        If the timestep is not positive.
    """

    def __init__(self, dt: float):
        if dt <= 0:
            raise wp.InvalidValueError(f"Require positive timestep, got {dt}")

        self._dt = dt

    @property
    def dt(self) -> float:
        """
        Returns the size of the elementary time step.
        """
        return self._dt

    @abstractmethod
    def step(self, state: State, t: float) -> State:
        """
        Evolves the given state for one elementary time step.

        Given a wave function or density operator at time t,
        this function should return the propagated wave function
        or density operator at the new time t+dt, where dt is
        the elementary time step.

        Parameters
        ----------
        state: wp.grid.State
            The state to be evolved in time
        t: float
            The time at which the time evolution starts

        Returns
        -------
        wp.grid.State
            The state at the new time t+dt.
        """
        pass

    def propagate(self, state0: State, t0: float,
                  num_steps: int, include_first: bool = True):
        """
        Generator function that yields the propagated wave function at multiple time steps.

        Parameters
        ----------
        state0: wp.grid.State
            The initial state to be propagated in time.
        t0: float
            The initial time at which the state is given
        num_steps: int
            For how many elementary time steps the state should be propagated.
        include_first: bool
            If true, the function will start by yielding the initial state.

        Yields
        ------
        float
            The time at which the state is yielded. Starts optionally with t0 and progresses in units of dt.
        wp.grid.State
            The propagated state at the given time.

        Raises
        ------
        wp.InvalidValueError
            If num_steps is negative.
        """
        if num_steps < 0:
            raise wp.InvalidValueError("Cannot propagate for negative number of steps.")

        if include_first:
            yield t0, state0

        state = state0
        for step in range(num_steps):
            t = t0 + step * self._dt
            state = self.step(state, t)
            yield t + self._dt, state
