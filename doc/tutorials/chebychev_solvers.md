---
file_format: mystnb
kernelspec:
    name: python3
---

# Using Chebychev solvers

This web page can be downloaded as notebook: {nb-download}`chebychev_solvers.ipynb` (Jupyter)
or {download}`chebychev_solvers.md` (Markdown)


```{note}
This tutorial focuses on the *usage* of the {py:class}`wavepacket.solver.ChebychevSolver`.
If you want to know more about the theory, see {doc}`/advanced/polynomial_solvers` or the original paper
by Tal-Ezer and Kosloff [^ChebychevReal]
```

Before deciding on the solver to use, be aware of the different tradeoffs between ODE solvers and the Chebychev solver:

Advantages of the Chebychev solver:

* much faster, something like a factor of 4-5.
* highly accurate with errors on the order of the numerical precision.

Drawbacks of the Chebychev solver:

* does not work for time-dependent systems
* does not work with complex eigenvalues (open systems, absorbing boundary conditions)
* setup is more complex

## Setting up a Chebychev solver

To set up the solver, you need three data items:

1. The expression that describes the equation of motion.
2. The spectrum of the Hamiltonian / Liouvillian, or rather upper and lower bounds for the maximum and minimum
   eigenvalue, respectively.
3. The length of one elementary time step.

The expression is trivial, you always need to define the differential equation to solve.

The tricky part is the determination of the spectral bounds.
If these are too generous, we lose efficiency proportional to the quality of the guess;
for example, if the Hamiltonian's spectrum fits twice into our guessed interval, computation takes twice as long as
for a perfect guess.
If, however, we choose them too tight, that is if the Liouvillian or Hamiltonian has an eigenvalue outside the
supplied bounds, the solution usually diverges.
We will discuss this in the next section.

The length of the time step is finally given through the alpha value:

$$
   \alpha = \frac{\Delta t \ \delta E}{2},
$$

where $\delta E$ is the difference between our guessed upper and lower bound of the spectrum
(the "spectral range" of the Hamiltonian/Liouvillian).
The alpha value should be larger than 40, otherwise the efficiency goes down.
If it is much larger than say 100, you might run into numerical problems.

```{note}
The Chebychev solver is only efficient for rather large time steps.
If you need the wave function at small timesteps, it is possible to implement (exact) interpolation, but that has
not been done yet.
Open a ticket or drop me a mail if this is an issue for your application.
```

### Determining the spectrum

Let us come to the tricky part: How do we obtain good bounds for the spectrum of the Hamiltonian?
As an example, let us set up a simple harmonic oscillator.

```{code-cell}
import wavepacket as wp

grid = wp.grid.Grid(wp.grid.PlaneWaveDof(-10, 10, 128))

kinetic = wp.operator.CartesianKineticEnergy(grid, 0, mass=1)
potential = wp.operator.Potential1D(grid, 0, lambda x: 0.5 * x ** 2)
hamiltonian = kinetic + potential

equation = wp.expression.SchroedingerEquation(hamiltonian)
liouvillian = wp.expression.CommutatorLiouvillian(hamiltonian)
```

For didactic reasons, we will discuss three items in the following:

1. How is the spectrum of the Liouvillian related to that of the Hamiltonian?
2. How can I estimate the bounds of a Hamiltonian's spectrum?
3. Are there better ways to increase the efficiency of the solver?

#### How is the spectrum of the Liouvillian related to that of the Hamiltonian?

If the spectrum of the Hamiltonian is inside the interval
$[E_\mathrm{min}, E_\mathrm{min} + \delta E]$, then the corresponding
{py:class}`wp.expression.CommutatorLiouvillian` has the spectrum inside $[-\delta E, \delta E]$,
because the coherence terms with the fastest oscillations $\sim \exp(\pm \imath \delta E t)$
occur between the eigenstates with the largest and smallest eigenvalues, respectively.

An important corollary is that a {py:class}`wp.solver.ChebychevSolver`
can normally only be used for wave functions *or* density operators.

#### How can I estimate the bounds of a Hamiltonian's spectrum?

First, we suggest not to spend too much effort on the exact spectrum, but take reasonable bounds instead.
Your time is usually more valuable than the additional computing time
from guessing generous bounds to the spectrum.

A simple lower bound of the Hamiltonian is given by the smallest value of the potential,
that is, zero in our harmonic oscillator example.

For an estimate of the largest eigenvalue, we can use the power iteration: Repeatedly applying the Hamiltonian
to an almost arbitrary state converges exponentially towards the eigenvector with the largest (absolute) eigenvalue.
The implementation is straight forward:

```{code-cell}
import math

psi = wp.builder.unit_wave_function(grid)
for iteration in range(10):
    psi = equation.apply(psi, t=0)
    psi = wp.grid.normalize(psi)

energy_guess = wp.operator.expectation_value(hamiltonian, psi).real
energy_guess = 1.2 * energy_guess
print(f"Energy guess = {energy_guess:.4}")
```

The initial state must contain the highest-energy eigenstate as a non-negligible component,
but has negligible relevance otherwise.
The factor of 1.2 is a guessed safety margin because the result may not have been converged yet.
Thus, we arrive at an estimate of the spectrum of about [0, 280].
For an alpha value of 40 or more, our time step must be at least $2 \cdot 40 / 280 = 2/7$.
Let us evolve a Gaussian wave packet with such values:

```{code-cell}
psi0 = wp.builder.product_wave_function(grid, wp.Gaussian(-5, 0, rms=1))
x_op = wp.operator.Potential1D(grid, 0, lambda x: x)
solver = wp.solver.ChebychevSolver(equation, math.pi/10, (0, energy_guess))

for t, psi in solver.propagate(psi0, t0=0.0, num_steps=10):
    trace = wp.grid.trace(psi)
    x = wp.operator.expectation_value(x_op, psi).real

    print(f"t = {t:.4}, trace = {trace:.4}, <x> = {x:.4}")
```

We can also check what happens if we get the spectrum wrong. Let us say we cut at 200 a.u.:

```{code-cell}
bad_solver = wp.solver.ChebychevSolver(equation, math.pi/10, (0, 200))

for t, psi in bad_solver.propagate(psi0, t0=0.0, num_steps=10):
    trace = wp.grid.trace(psi)
    print(f"t = {t:.4}, trace = {trace:.4}")
```

The footprint of our wrong spectrum is readily apparent if we monitor the trace.
So that is something you should always do in practice.

#### Are there better ways to increase the efficiency of the solver?

So far, we have taken the Hamiltonian for granted and tried to estimate bounds of the spectrum.
However, we may just as well *define* the spectral bounds by truncating the operator.
To motivate this approach, let us print the average and standard deviation of the initial state's energy:

```{code-cell}
psi0 = wp.builder.product_wave_function(grid, wp.Gaussian(-5, 0, rms=1))
energy = wp.operator.expectation_value(hamiltonian, psi0).real
energy2 = wp.operator.expectation_value(hamiltonian*hamiltonian, psi0).real

print(f"E = {energy},   dE^2 = {energy2 - energy**2}")
```

From this data, it seems safe to assume that the state is well represented
using only energy eigenstates with energies <= 30 a.u.
This is *much* less than the estimate of 280 a.u. that we have guessed in the preceding section.
In other words, 90% of our computing time is wasted on the correct propagation of high-energy eigenstates.
These are irrelevant for our dynamics, but would blow up the calculation if we chose the bounds too small.
Such an imbalance between the spectrum that the Hamiltonian can support and the spectrum that we actually need
is unfortunately common.

To get rid of this waste, we truncate the Hamiltonian.
Because this is difficult, we actually truncate the kinetic and potential energy operators instead.
The question is: At what values?
Here we need some intuition again (or guesswork or plain trial-and-error).

Our Gaussian (x0 = -5, rms = 1) extends maybe three rms up to x=8, where the potential value is
$8^2/2 = 32 \ \mathrm{a.u.}$.
Hence, we might truncate at 35 a.u., which is just a little less than the maximum potential of 50 a.u.
For the kinetic energy, we do the same, which has a much larger effect.
The system setup changes to

```{code-cell}
truncated_kinetic = wp.operator.CartesianKineticEnergy(grid, 0, mass=1, cutoff=35)
truncated_potential = wp.operator.Potential1D(grid, 0, lambda x: 0.5 * x ** 2, cutoff=35)
truncated_hamiltonian = truncated_kinetic + truncated_potential
truncated_equation = wp.expression.SchroedingerEquation(truncated_hamiltonian)
```

We do not know nor care about the exact spectrum of this Hamiltonian,
but we know that it cannot be larger than 35 + 35 = 70, which is one fourth of the original Hamiltonian.
This allows us to increase the time step by a factor of 4.

```{code-cell}
solver = wp.solver.ChebychevSolver(truncated_equation, 4 * math.pi/10, (0, 70))

for t, psi in solver.propagate(psi0, t0=0.0, num_steps=10):
    trace = wp.grid.trace(psi)
    x = wp.operator.expectation_value(x_op, psi).real

    print(f"t = {t:.4}, trace = {trace:.4}, <x> = {x:.4}")
```

[^ChebychevReal]: H. Tal Ezer and R. Kosloff, J. Chem. Phys. 81:3967 (1986)
                       <https://openscholar.huji.ac.il/sites/default/files/ronniekosloff/files/jcp1.448136.pdf>

