---
file_format: mystnb
kernelspec:
    name: python3
---

# Background theory: ODE solvers

This web page can be downloaded as notebook: {nb-download}`ode_solvers.ipynb` (Jupyter)
or {download}`ode_solvers.md` (Markdown)

Here, we want to have a look at conventional ODE solvers.
In particular, how do you check calculations, and how do you speed up a calculation?

For the impatient, the main results of the following treatise are:

* Monitor the norm of the wave function. Poor convergence usually makes the norm diverge.
* The required time step, hence the performance,
  is dictated by the largest (absolute) eigenvalue, not by the typical energies.
* Truncate aggressively and shift energies to improve performance.

If these results sound suspiciously like the take-home advice from {doc}`polynomial_solvers`,
this is not a coincidence; after all, all solvers tackle the same problem.

## Some basics

### ODE solvers theory

We start from the Schrödinger equation

$$
    \imath \dot \psi(t) = \hat H(t) \psi(t)
$$

and expand the wave function in a complete basis with time-dependent coefficients,
$|\psi(t)\rangle = \sum_n c_n(t) |\phi_n\rangle$.
Inserting and multiplying with $\langle \phi_m |$ gives the formula for the coefficients,

$$
    \imath \dot c_m(t) = \sum_n \langle \phi_m| \, \hat H \, |\phi_n \rangle \ c_n(t)
$$

This is just a system of ordinary differential equations, so any solver can be used for the time evolution,
for example Runge-Kutta algorithms.
A similar derivation can be done for the Liouville von-Neumann equation for density operators.
Note that the typical solvers do not have unitary time evolution built in.
As such, they do not preserve the norm, which can therefore be used as proxy for the quality of convergence.

ODE solvers, such as Runge-Kutta have the big advantage that they are robust work horses.
You can throw any problem at them: time-dependent Hamiltonians, complex-valued Hamiltonians,  open systems, ...
They are mostly guaranteed to give you a solution.
They are also easy to use.
With adaptive solvers, you do not even need to specify a time step, just an error bound.

The big drawback is performance; ODE solvers do not exploit any knowledge of the system,
and this makes better adapted solvers like polynomial solvers more efficient for specific classes of problems.

### Model system

As an example system, we choose a one-dimensional harmonic oscillator
with a shifted Gaussian as initial state.

```{code-cell}
import wavepacket as wp

grid = wp.grid.Grid(wp.grid.PlaneWaveDof(-10, 10, 128))
hamiltonian = (wp.operator.CartesianKineticEnergy(grid, 0, 1.0)
              + wp.operator.Potential1D(grid, 0, lambda x: 0.5 * x ** 2))
equation = wp.expression.SchroedingerEquation(hamiltonian)

psi0 = wp.builder.product_wave_function(grid, wp.special.Gaussian(x=-3, rms=2))
```

Let us have a look at the projection onto the Hamiltonian eigenstates:

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np

eigenstates = [state for _, state in wp.diagonalize(hamiltonian)]
projection = [wp.population(psi0, state) for state in eigenstates]

plt.semilogy(np.arange(128), projection, '+')
```

In theory, the population should decay exponentially.
In practice, things start to happen around the 50th eigenstate.
You can check that this is an artifact of the finite grid extent and the finite number of grid points,
we will skip it here to reduce the run time of the notebook.
This is a common pattern and discussed in depth in {doc}`plane_wave_grid`.
We will come back to this observation later.

## Convergence behavior

Intuitively, we would assume that the time step is determined by our initial wave function.
After all, a low-energy state evolves slower in time than a high-energy state,
hence it should require fewer time steps.
Unfortunately, this intuition is incorrect.

To understand what happens let us produce a non-converged calculation.
This is a contrived example; scipy's ODE solvers always use adaptive stepping to keep the error manageable,
we go an extra mile here to render the mechanism useless.
To understand where exactly things go awry,
we can plot the populations of the individual eigenstates while the solution diverges.

```{code-cell}
solver = wp.solver.OdeSolver(equation, 0.1, atol=1e10, first_step=0.01)
x = np.arange(128)

for t, psi in solver.propagate(psi0, 0, 2):
    print(f"t = {t}      Trace = {wp.trace(psi)}")

    populations = [wp.population(psi, state) for state in eigenstates]
    plt.semilogy(x, populations, '+')
```

The plot tells a clear story: The divergence is caused only by the high-energy contributions.
These are completely irrelevant for the actual dynamics,
but require small timesteps for accurate time evolution.

In practice, you do not normally observe divergence,
because the ODE solvers use adaptive time steps to keep the error within a given bound.
However, this problem does show up as inefficiency:
We need more time steps, hence more compute power than we would like to spend.

Assuming we want to stick with ODE solvers,
efficiency gains can come only from optimizing the operator spectrum.
There are two options: We can shift the spectrum, or we can truncate it.

### Shifting the spectrum

The most expensive task for the solver is the correct propagation of the populated
states with the highest energies.
The burden can be reduced by simply making the (absolute values of the) energies smaller,
for example by shifting the operator spectrum.
Simple scaling laws suggest that the required time step scales with the inverse of the largest energy,
so shifting the spectrum to be balanced around zero would ideally halve the computation time.

To see what happens, let us repeat the previous analysis with the spectrum shifted by 50 atomic units
(approximately the energy of the 50th eigenstate)

```{code-cell}
shifted_equation = wp.expression.SchroedingerEquation(hamiltonian - 50)

shifted_solver = wp.solver.OdeSolver(shifted_equation, 0.1, atol=1e10, first_step=0.01)
x = np.arange(128)

for t, psi in shifted_solver.propagate(psi0, 0, 2):
    print(f"t = {t}      Trace = {wp.trace(psi)}")

    populations = [wp.population(psi, state) for state in eigenstates]
    plt.semilogy(x, populations, '+')
```

The good news is that this helps visibly with the divergent trace.
Alas, now the populations of the low-energy states start to diverge.
This causes problems with adaptive step sizes,
because the low-energy have much larger populations and therefore affect the step error to a larger degree. 

Shifting by half of the spectral range is probably not optimal.
to figure out the optimal value, we introduce a "Counting expression" that measures how
often our Hamiltonian is applied (the most expensive operation in the algorithm).

```{code-cell}
class CountingExpression(wp.expression.ExpressionBase):
    def __init__(self, wrapped_expression):
        super().__init__(False)

        self._wrapped_expression = wrapped_expression
        self.count = 0

    def apply(self, state, t):
        self.count += 1
        return self._wrapped_expression.apply(state, t)
```

Then we can try out the computational effort for different shifts:

```{code-cell}
for shift in range(0, 51, 10):
    shifted_equation = wp.expression.SchroedingerEquation(hamiltonian - shift)
    counter = CountingExpression(shifted_equation)
    solver = wp.solver.OdeSolver(counter, 0.5)

    solver.step(psi0, 0)
    print(f"Shift = {shift} a.u.      count = {counter.count}")
```

For the optimal shift around 40 a.u., the computational effort is reduced by about 25%.
Such savings are not groundbreaking,
but somewhat attractive since you get them almost for free.

### Truncating the spectrum

The second manipulation is a direct truncation of the operator spectrum.
Why this may seem clumsy on first encounter, there are good reasons to truncate aggressively:

1. As the previous analyses have shown, high-energy eigenstates are poorly represented by the grid anyway.
   In the example here, this affects everything above about the 50th eigenstate.
   Aggressive truncation therefore only invalidates results that were already incorrect.

   Alternatively, we might intuitively try to extend the grid to improve the accuracy results,
   but this quickly leads to diminishing returns for significantly increased computational cost.
2. Whenever we simulate systems, we use model systems,
   which are usually low-energy approximations of real systems.
   These are typically wrong in the high-energy regime.
   In the particular example here, we can safely assume that, say, for the 50th excited state,
   the system will definitely not be harmonic any longer.

   Again, this means that by truncating the Hamiltonian,
   we merely trade an incorrect approximation for a different incorrect result, no harm done.

Truncating the spectrum of the Hamiltonian requires an expensive diagonalization,
so we truncate the individual components instead.

Similar to the shifting of the spectrum, we can study the effect of truncation on the efficiency.

```{code-cell}
for cutoff in range(50, 201, 50):
    truncated_hamiltonian = (wp.operator.CartesianKineticEnergy(grid, 0, 1.0, cutoff=cutoff)
                            + wp.operator.Potential1D(grid, 0, lambda x: 0.5 * x ** 2, cutoff=cutoff))
    truncated_equation = wp.expression.SchroedingerEquation(truncated_hamiltonian)
    counter = CountingExpression(truncated_equation)
    solver = wp.solver.OdeSolver(counter, 0.5)

    solver.step(psi0, 0)
    print(f"Cutoff = {cutoff} a.u.      count = {counter.count}")
```

The computational effort is proportional to the cutoff,
which supports the original thesis that the effort is spent on propagating high-energy contributions.
Truncating the Hamiltonian to 50 a.u. halves the computational effort.
