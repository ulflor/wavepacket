---
file_format: mystnb
kernelspec:
    name: python3
---

# Background theory: Reduced density operators

This web page can be downloaded as notebook: {nb-download}`reduced_density.ipynb` (Jupyter)
or {download}`reduced_density.md` (Markdown)

In this notebook, we investigate a toy problem that qualitatively demonstrates the effect of
an environment on a quantum measurement.
Basically, we discuss the mechanism behind coherent and incoherent superpositions,
that is shown in {doc}`/tutorials/schroedinger_cat`.

## Theory: Partial trace

Let us consider an excerpt of the universe whose state can be described by a wave function.
At some point, we want to make a measurement, so we calculate an expectation value of the corresponding operator.
By inserting a summation over an orthonormal basis {math}`\phi`, we can rewrite this as a trace over
a (pure) density operator,

\begin{align*}
\langle \hat A \rangle &= \langle \psi | \hat A | \psi \rangle\\
  &= \langle \psi | \hat A \sum_n|\phi_n\rangle \langle \phi_n | \psi \rangle\\
  &= \sum_n \langle \phi_n | \psi \rangle \langle \psi | \hat A |\phi_n\rangle\\
  &= \mathrm{Tr}(|\psi\rangle\langle\psi | \hat A) = \mathrm{Tr} (\hat \rho \hat A)
  .
\end{align*}

Now we decompose our small universe into system and environment degrees of freedom, $\mathcal{H} = S \otimes E$
with respective orthonormal basis $\varphi, \eta$.
As "system", we denote those degrees of freedom along which our operator(s) act,
the rest we call "environment" or "bath".
By definition, the operator can be rewritten as $\hat A = \hat A_S \otimes \hat 1_E$,
where the subscripts denote action or integration over the system or environment part.
The trace now becomes

\begin{align*}
\langle \hat A \rangle &= \sum_{k,l} \Bigl(\langle \varphi_k | \otimes \langle \eta_l|\Bigr) \  \hat \rho
    \hat A_S \otimes \hat 1_E \ \Bigl(|\varphi_k\rangle_S \otimes |\eta_l \rangle_E\Bigr)\\
  &= \sum_k \langle \varphi_k | \ \Bigl( \sum_l \langle \eta_l | \hat \rho |\eta_l\rangle_E \Bigr)
    \ \hat A_S |\varphi_k\rangle_S \\
  &= \sum_k \langle \varphi_k | \hat \rho_\mathrm{red} \hat A_S | \varphi_k \rangle \\
  &= \mathrm{Tr}_S ( \hat \varrho_\mathrm{red} \hat A_S) \qquad \qquad
    \mathrm{with} \qquad \hat \varrho_\mathrm{red} = \mathrm{Tr}_E \hat \rho
\end{align*}

Even though our original state is pure and can be described by a wave function,
the reduced density may be impure/mixed and describe a statistical ensemble.
It can easily be checked that

\begin{align*}
    \psi = (\varphi_0 + \varphi_1) \otimes \eta_0 \quad &\rightarrow \quad
        \varrho_\mathrm{red} = |\varphi_0+\varphi_1\rangle\langle\varphi_0+\varphi_1|\\
    \psi = \varphi_0 \otimes \eta_0 + \varphi_1 \otimes \eta_1 \quad &\rightarrow \quad
        \varrho_\mathrm{red} = |\varphi_0\rangle\langle\varphi_0| + |\varphi_1\langle\rangle \varphi_1|
\end{align*}

That is, we get a pure reduced density if $\psi$ is a product state,
and a mixed reduced density if $\psi$ contains correlations between the system and environment.

## The system under study

With the basic theory result spelled out, let us look at a toy example, where we can observe these effects,
yet still understand the full dynamics.
We use one-dimensional harmonic oscillators for the system and environment, respectively.
Let us start with the grid definitions:

```{code-cell}
import wavepacket as wp

system_dof = wp.grid.PlaneWaveDof(-10, 10, 96)
environment_dof = wp.grid.PlaneWaveDof(-5, 5, 96)
full_grid = wp.grid.Grid([system_dof, environment_dof])

trafo = wp.grid.PartialTraceTransformation(full_grid, 0)
system_grid = trafo.target_grid
```

We exploit here that the transformation object creates a one-dimensional "target grid"
containing only the first (/system) degree of freedom during setup.
We will use this transformation later to calculate the reduced density operator
from a state defined on the system+environment grid.

For system and environment, we define different harmonic oscillators

```{code-cell}
kinetic_sys = wp.operator.CartesianKineticEnergy(system_grid, 0, mass=1, cutoff=40)
potential_sys = wp.operator.Potential1D(system_grid, 0, lambda x: 0.5 * x**2, cutoff=40)

kinetic_full = (wp.operator.CartesianKineticEnergy(full_grid, 0, mass=1, cutoff=40)
               + wp.operator.CartesianKineticEnergy(full_grid, 1, mass=1, cutoff=40))
potential_full = (wp.operator.Potential1D(full_grid, 0, lambda x: 0.5 * x**2, cutoff=40)
                 + wp.operator.Potential1D(full_grid, 1, lambda x: 0.5 * (2*x)**2, cutoff=40))
```

## Dynamics with and without environment

### Product state without coupling

To demonstrate the theoretical results, we start with an environment not coupled to the system,
and with a product state.
For the initial system states, we choose a squeezed and shifted Gaussian that oscillates in the system potential.
For the environment state, we choose some Gaussian centered around the potential minimum.

```{code-cell}
import math

left_gauss = wp.special.Gaussian(-5, 0, rms=0.5)
right_gauss = wp.special.Gaussian(5, 0, rms=0.5)
env_gauss = wp.special.Gaussian(0, 0, rms=0.5)

psi0_sys = math.sqrt(0.5) * (
             wp.builder.product_wave_function(system_grid, left_gauss)
             + wp.builder.product_wave_function(system_grid, right_gauss)
           )

psi0_full = math.sqrt(0.5) * (
              wp.builder.product_wave_function(full_grid, [left_gauss, env_gauss])
              + wp.builder.product_wave_function(full_grid, [right_gauss, env_gauss])
            )
```

As reference, we start with a simulation of only the system degree of freedom.

```{code-cell}
equation_sys = wp.expression.SchroedingerEquation(kinetic_sys + potential_sys)

solver = wp.solver.ChebychevSolver(equation_sys, math.pi/6, (0, 80))
plotter = wp.plot.StackedPlot1D(4, psi0_sys, potential_sys, kinetic_sys + potential_sys)
for t, psi in solver.propagate(psi0_sys, 0.0, 3):
    plotter.plot(t, psi);
```

The two Gaussians breathe and oscillate in the potential and show interference when they cross paths.
This is just textbook quantum mechanics, no surprises so far.

Now what happens if we add an environment degree of freedom?
We start with a product state, and because of the missing system-environment coupling,
our solution remains a product state.
Hence, we expect a pure reduced density matrix that just replicates the
system-only result including interferences.

```{code-cell}
equation_uncoupled = wp.expression.SchroedingerEquation(kinetic_full + potential_full)

solver = wp.solver.ChebychevSolver(equation_uncoupled, math.pi/6, (0, 160))
plotter = wp.plot.StackedPlot1D(4, psi0_sys, potential_sys, kinetic_sys + potential_sys)
for t, psi in solver.propagate(psi0_full, 0.0, 3):
    reduced_density = trafo.transform(psi)
    plotter.plot(t, reduced_density);
```

This result also makes sense for philosophical reasons.
If the addition of an arbitrary, uninteresting degree of freedom would change the results,
we could modify any calculation results by, say, considering the phase of the moon
which is an obviously silly proposition.

### Couplings and correlations

For the environment to have an effect on the system dynamics, we need to introduce a coupling.
To understand what happens now, let us first have a look at the two-dimensional dynamics.

```{code-cell}
coupling = -0.5 * (wp.operator.Potential1D(full_grid, 0, lambda x: x)
                  * wp.operator.Potential1D(full_grid, 1, lambda x: x))
equation_coupled = wp.expression.SchroedingerEquation(kinetic_full + potential_full + coupling)

# store the results, we iterate twice over them, but want to calculate only once.
solver = wp.solver.ChebychevSolver(equation_coupled, math.pi/6, (0, 210))
results = [(t, psi) for t, psi in solver.propagate(psi0_full, 0.0, 3)]

plotter = wp.plot.StackedContourPlot2D(2, 2, psi0_full, potential_full + coupling)
plotter.contours /= 2
for t, psi in results:
    plotter.plot(t, psi)
```

Adding a bilinear coupling to a harmonic oscillator results in a harmonic oscillator
with different frequencies, and rotated normal modes.
For the parameters here, we see that the new normal modes are roughly
$n_0 \approx x_s + 0.5 x_e$ and $n_1 \approx x_s - 0.5 x_e$.

The two Gaussians that form our initial state move along these new normal modes, and now it gets interesting.
From a purely formal perspective, we get correlations between the system and environment.
That is, the total wave function is clearly no longer a direct product of a system and an environment wave function.
As discussed in the theory section, we expect a mixed density operator and a suppression of interferences.

We can also look at the situation from a different angle.
Because the two Gaussians move along the new normal modes, they mostly miss each other,
and barely overlap / interfere in the system-environment space.
The operation of tracing out the environment just projects this situation onto the system axis;
basically as if you "looked onto a projection of the plot while standing on the system axis".

```{code-cell}
plotter = wp.plot.StackedPlot1D(4, psi0_sys, potential_sys, kinetic_sys + potential_sys)
for t, psi in results:
    reduced_density = trafo.transform(psi)
    plotter.plot(t, reduced_density);
```

In a "real" system, we define observables for a few degrees of freedom at most,
while the environment around the system has a gazillion degrees of freedom.
There is then much more "space" for different parts of the total wave function to avoid each other.
After a while (the "decoherence time"), the likelihood of different parts of
the wave function overlapping in the system+environment space is negligible,
and interferences vanish.

Note, however, that we introduced a pretty strong system-environment coupling to force visible effects.
In practical systems, such couplings may be weak, giving significant experimental time scales during which you
can safely ignore environment effects.

But even the weakest coupling affects the initial state.
Typically, at the start of an experiment, the system has spent plenty of time interacting with the environment;
it is "thermalized".
And this interaction has introduced correlations between the system and the environment into the initial state,
whose reduced density matrix is therefore no longer pure.
Let us set up such a prototypical situation here again: We use a correlated initial state,
and ignore coupling between system and environment.

```{code-cell}
psi0_correlated = math.sqrt(0.5) * (
              wp.builder.product_wave_function(full_grid, [left_gauss, wp.special.Gaussian(-1.5, 0, rms=0.5)])
              + wp.builder.product_wave_function(full_grid, [right_gauss, env_gauss])
            )


# store the results, we iterate twice over them, but want to calculate only once.
solver = wp.solver.ChebychevSolver(equation_uncoupled, math.pi/6, (0, 180))
results = [(t, psi) for t, psi in solver.propagate(psi0_correlated, 0.0, 3)]

plotter = wp.plot.StackedContourPlot2D(2, 2, psi0_correlated, potential_full)
plotter.contours /= 2
for t, psi in results:
    plotter.plot(t, psi)
```

The "right" Gaussian oscillates only along the system degree of freedom,
while the "left" Gaussian follows a trajectory from the "bottom left" to the "center top".
As a consequence, these two wave packets again miss each other, and the interference pattern is suppressed.
If we look at the system degree of freedom again, we find exactly this pattern:

```{code-cell}
plotter = wp.plot.StackedPlot1D(4, psi0_sys, potential_sys, kinetic_sys + potential_sys)
for t, psi in results:
    reduced_density = trafo.transform(psi)
    plotter.plot(t, reduced_density);
```

While interference patterns _within_ each Gaussian are retained during the simulation (no coupling),
there is no interference _between_ the two Gaussians.
Or in other words, even though we can ignore the environment during the simulation, we need to
take it into account in the initial state.
Following this thread of thought much further eventually leads to the
expression of thermal states with mixed density operators,
without interference terms between the system Hamiltonian's eigentstates,
but that goes well beyond such a small Jupyter notebook.
