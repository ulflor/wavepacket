---
file_format: mystnb
kernelspec:
    name: python3
---

# Schroedinger cat states


This demo demonstrates one of the core features of Wavepacket:
Switching between wave functions and density operators with minimal overhead

We consider a simple free particle as a coherent or incoherent sum of two Gaussians.
This shows the main features without too much detracting physics.


```{code-cell}
import math

import matplotlib.pyplot as plt
import numpy as np
import wavepacket as wp
```


Both descriptions, wave function and density operator, share the grid, and
the operator structure.

```{code-cell}
degree_of_freedom = wp.grid.PlaneWaveDof(-15, 15, 128)
grid = wp.grid.Grid(degree_of_freedom)
hamiltonian = wp.operator.CartesianKineticEnergy(grid, dof_index=0, mass=1.0)
```


For wave functions, we set up an initial wave function,
and create a Schroedinger equation that we then solve.
The dynamics are not terribly surprising:
The initial Gaussians only broaden over time.

In the plots, note the small peak in the density around the zero coordinate,
and some wiggles as soon as the two wave functions come into contact with each other.
Both features arise from the coherent addition of the two wave functions

```{code-cell}
rms = math.sqrt(0.5)
psi_left = wp.builder.product_wave_function(grid, wp.Gaussian(-3, rms=rms))
psi_right = wp.builder.product_wave_function(grid, wp.Gaussian(3, rms=rms))
psi_0 = math.sqrt(0.5) * (psi_left + psi_right)

schroedinger_eq = wp.expression.SchroedingerEquation(hamiltonian)
solver = wp.solver.OdeSolver(schroedinger_eq, dt=math.pi/5)

for t, psi in solver.propagate(psi_0, t0=0.0, num_steps=5):
    # TODO: Better and more convenient plotting
    plt.plot(grid.dofs[0].dvr_points, wp.grid.dvr_density(psi))
```


We can equally use a density operator description.
For that, we have to setup the initial state as an operator.
Also, the equations are motion are now governed by a Liouvillian.

Besides these unavoidable changes, the interface works the same
as for the wave function case.
Convenience functions, such as `wavepacket.grid.dvr_density()` work for both states.

Note how the small peak at the zero coordinate and the wiggles are gone now.
This is the effect of the summation;
there is no coherence between the left and right Gaussian.

```{code-cell}
rho_0 = 0.5 * (wp.builder.pure_density(psi_left) + wp.builder.pure_density(psi_right))

liouvillian = wp.expression.CommutatorLiouvillian(hamiltonian)
solver = wp.solver.OdeSolver(liouvillian, dt=math.pi/5)

for t, rho in solver.propagate(rho_0, t0=0.0, num_steps=5):
    # TODO: Better and more convenient plotting
    plt.plot(grid.dofs[0].dvr_points, wp.grid.dvr_density(rho))
```
