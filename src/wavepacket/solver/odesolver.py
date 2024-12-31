import numpy as np
from scipy.integrate import solve_ivp

import wavepacket as wp
import wavepacket.typing as wpt
from .solverbase import SolverBase
from ..grid import Grid, State
from ..expression import ExpressionBase


def _inner_solver(t: float, y: wpt.ComplexData,
                  eq: ExpressionBase, grid: Grid) -> wpt.ComplexData:
    # 1. Reconstruct a state from the data array y
    if len(y) == grid.size:
        state = State(grid, np.reshape(y, grid.shape))
    else:
        state = State(grid, np.reshape(y, grid.operator_shape))

    # 2. Insert into the equation expression
    dot_state = eq.apply(state, t)

    # 3. Extract the result as a vector again and return
    return dot_state.data.ravel()


class OdeSolver(SolverBase):
    def __init__(self, expr: ExpressionBase, dt: float, **kwargs):
        super().__init__(dt)

        self._expression = expr
        self._kwargs = kwargs

        # The default error tolerance is typically too generous for our use cases. Override
        self._kwargs.setdefault('rtol', 1e-6)

    def step(self, state: State, t: float) -> State:
        t_span = (t, t + self.dt)
        y0 = state.data.ravel()
        args = (self._expression, state.grid)

        solution = solve_ivp(_inner_solver, t_span, y0, t_eval=[t + self._dt], args=args, **self._kwargs)
        if solution.status != 0:
            raise wp.ExecutionError("Bad return value from integrating"
                                    f"Status: {solution.status} != 0; Message: {solution.msg}")

        return State(state.grid, np.reshape(solution.y, state.data.shape))