---
file_format: mystnb
kernelspec:
    name: python3
---

# Calculating eigenstates and -energies of operators

There are a number of use cases where we want to calculate the eigenstates of a Hamiltonian,
for example, if we start a calculation from an ensemble of low-temperature states.
Here, we show some basic and advanced use cases around energy eigenstates.

Note that there exist other approaches for calculating eigenstates of the Hamiltonian.
In particular, for few low-energy eigenstates, you can try imaginary-time propagation,
as detailed in{doc}`relaxation`.


## Basic eigenstate calculation

As example system, we choose a modified double-well potential (Razavy potential).
Diagonalization is done with :py:func:`wp.solver.diagonalize`,
which yields the eigenstates and -energies sorted by the latter.

```{code-cell}
import numpy as np
import wavepacket as wp

def razavy_potential(dvr_points):
    val = np.cosh(dvr_points)
    return -0.7 * val + 0.01 * val**2
    
grid = wp.grid.Grid(wp.grid.PlaneWaveDof(-7, 7, 128))
hamiltonian = wp.operator.CartesianKineticEnergy(grid, 0, 0.5) + wp.operator.Potential1D(grid, 0, razavy_potential)

for energy, state in wp.solver.diagonalize(hamiltonian):
    # placeholder for doing something with the eigenstate
    print(f"E = {energy} a.u., |psi|^2 = {wp.grid.trace(state)}")
```

Note that the direct diagonalization does not do fancy tricks.
It only extracts the (dense) matrix representation of the operator, diagonalizes it using `numpy.linalg.eigh`,
and transforms the output into Wavepacket data structures.
It is therefore rather robust, but not terribly efficient.

Computation times of such standard solvers scale with the third power of the number of grid points,
memory with the square of the number of grid points.
At a size of 8 bytes per data point, you need approximately 1 GB of memory for 10,000 grid points (100x100 grid),
with several minutes of effort required for the diagonalization.
This is about the maximum size that you can reasonably use this procedure for.

## Advanced usage: Largest and smallest eigenstates

In some applications, you are not interested in all eigenstates, but only some of them.
Usually, you start by wrapping the calculation results in a list.
This is slightly inefficient if you do not need all states,
but the wasted effort is much smaller than the effort for the eigenvalue calculation in the first place.

```{code-cell}
results = list(wp.solver.diagonalize(hamiltonian))
len(results)
```

We now have a list of (energy, state) pairs, and can manipulate them with usual Python list comprehensions.
For example, to get all energies up to a cut-off,

```{code-cell}
results_with_negative_energy = [(energy, state) for energy, state in results if energy < 0]
len(results_with_negative_energy)
```

The same approach can be used with slicing, sorting etc.
The resulting list can always be iterated over like the initial generator

```{code-cell}
sliced_results = results[0:10]

inverted_order = sliced_results
inverted_order.sort(reverse=True)

for energy, state in inverted_order:
    print(f"E = {energy} a.u., |psi|^2 = {wp.grid.trace(state)}")
```
