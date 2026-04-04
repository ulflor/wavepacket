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

* Poor convergence can show up as a divergent norm.
  In practice, however, ODE solvers use adaptive step sizes;
  for reasonable parameters, they just take forever.
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

This is a system of ordinary differential equations,
and any explicit solver can be used for the time evolution, most notably Runge-Kutta algorithms.
A similar derivation can be done for the Liouville von-Neumann equation for density operators.

ODE solvers have the big advantage that they are robust work horses.
You can throw any problem at them: time-dependent Hamiltonians, complex-valued Hamiltonians,  open systems, ...
they will eventually produce a solution.
They are also easy to use.
You typically get adaptive solvers, which choose the time step based on an error bound that you supply.

The big drawback is performance; ODE solvers do not exploit any knowledge of the system,
and this makes better adapted solvers like polynomial solvers more efficient for specific classes of problems.
Typically, ODE solvers are not unitary, so the norm is not a conserved quantity and diverges on poor convergence.
However, because of the adaptive steps, this does not usually happen,
instead the algorithm chooses tiny step sizes.

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
In practice, strange things start to happen around the 50th eigenstate.
You can check that this is an artifact of the finite grid extent and the finite number of grid points,
we will skip it here to reduce the run time of the notebook.
This finite grid limitation is a common issue and discussed in depth in {doc}`plane_wave_grid`.
We will come back to this observation later.

## Convergence behavior

Intuitively, we would assume that the time step is determined by the system dynamics,
i.e., the initial wave function in a time-independent system.
For example, a low-energy state evolves slower in time than a high-energy state,
hence it should require fewer time steps.
Unfortunately, this intuition is incorrect.

To understand what happens let us enforce a non-converged calculation.
This is a contrived example; scipy's ODE solvers always use adaptive stepping,
so we go an extra mile to render the mechanism useless.
To understand where exactly things go awry,
we can plot the populations of the individual eigenstates

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

Efficiency gains thus require an optimization of the operator spectrum.
There are two options: We can shift the spectrum, or we can truncate it.

### Shifting the spectrum

The solver spends most effort on the correct propagation of the populated
states with the highest energies.
This burden can be reduced by shifting the operator spectrum,
thereby making the (absolute values of the) energies smaller.
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

The good news is that the shift helps visibly with the divergent trace.
Alas, now the populations of the low-energy states start to diverge.
This causes problems with adaptive step sizes,
because the low-energy states have larger populations,
therefore they contribute more to the step error. 

As a consequence, shifting by half of the spectral range is probably not optimal.
To figure out the optimal value, we use a "Counting expression" that measures
the most expensive operation in the algorithm, how often our Hamiltonian is applied.

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

Then we can try out the computational effort for different energy shifts:

```{code-cell}
for shift in range(0, 51, 10):
    shifted_equation = wp.expression.SchroedingerEquation(hamiltonian - shift)
    counter = CountingExpression(shifted_equation)
    solver = wp.solver.OdeSolver(counter, 0.5)

    solver.step(psi0, 0)
    print(f"Shift = {shift} a.u.      count = {counter.count}")
```

We find an optimal shift of around 40 a.u., which reduces the computational effort by about 25%.
The drawback is the required guesswork to find a suitable energy shift;
choosing incorrectly yields smaller gains.

### Truncating the spectrum

The second manipulation is a direct truncation of the operator spectrum.
Why this may seem clumsy on first encounter, there are good reasons to truncate aggressively:

1. As the previous analyses have shown, high-energy eigenstates are poorly represented by the grid anyway.
   In the example here, this affects everything above about the 50th eigenstate.
   Aggressive truncation therefore only invalidates results that were already incorrect.

   In theory, we could remove these artifacts by extending the grid,
   but then we pay twice: The larger grid makes all computations slower,
   and to faithfully propagate the large-energy components, we need smaller step sizes.
   And this all for diminishing gains, because the populations are small, after all.
2. Whenever we simulate existing physical systems,
   we use models that are usually low-energy approximations of the real system.
   For example, we neglect anharmonic contributions,
   or we ignore the coupling to higher excited electronic states.
   In the particular example here, we can safely assume that, say, for the 50th excited state,
   the system will definitely not be harmonic any longer.

   Again, this means that by truncating the Hamiltonian,
   we merely trade an incorrect approximation for another incorrect result, no harm done.

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

The computational effort is mostly proportional to the cutoff,
which supports the original thesis that most effort is spent on propagating high-energy contributions.
Truncating the Hamiltonian to 50 a.u. in our example halves the computational effort.

How bad is the approximation for the dynamics?
To answer this question, we can look at the projection of the truncated result on the untruncated result;
any deviation from one can serve as a figure of merit of the additional error.

```{code-cell}
import math

truncated_hamiltonian = (wp.operator.CartesianKineticEnergy(grid, 0, 1.0, cutoff=50)
                        + wp.operator.Potential1D(grid, 0, lambda x: 0.5 * x ** 2, cutoff=50))
truncated_equation = wp.expression.SchroedingerEquation(truncated_hamiltonian)
truncated_solver = wp.solver.OdeSolver(truncated_equation, math.pi / 2)
truncated_result = truncated_solver.step(psi0, 0)

solver = wp.solver.OdeSolver(equation, math.pi/2)
result = solver.step(psi0, 0)

print(f"Overlap after pi/2: {wp.population(truncated_result, result)}")
```

The additional error in the sixth decimal is of a similar magnitude
and therefore consistent with the grid error.
