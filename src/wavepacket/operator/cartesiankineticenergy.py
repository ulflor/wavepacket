import wavepacket as wp
from .planewavefbroperator import PlaneWaveFbrOperator
from ..grid import Grid


class CartesianKineticEnergy(PlaneWaveFbrOperator):
    def __init__(self, grid: Grid, dof_index: int, mass: float):
        if mass <= 0:
            raise wp.InvalidValueError(f"Particle mass must be positive, but is {mass}")

        super().__init__(grid, dof_index, lambda fbr_points: fbr_points ** 2 / (2 * mass))
