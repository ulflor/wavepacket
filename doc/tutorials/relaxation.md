---
file_format: mystnb
kernelspec:
    name: python3
---

# Ground-state relaxation and finite temperatures

## Introduction

In imaginary time, {math}`t \to \imath \tau`, the Schrödinger equation becomes

```{math}
    \frac{\partial \psi}{\partial \tau} = - \hat H \psi(\tau)
```

with the formal solution of the time evolution operator

```{math}
    \hat U(\tau) = \mathrm{e}^{-\hat H \tau}.
```

There are two main applications for such a time evolution:

1. If you apply this time evolution operator to some wave function, the norm decays.
   The decay is faster for high-energy contributions, so that after sufficient propagation time,
   the wave function is dominated by the ground state.
   Hence, imaginary-time propagation is a quick and easy way to get the ground state of a system
   (and also some excited states).
2. The time evolution operator has the shape of a density operator for the system at finite temperature.
   Hence, you can use imaginary-time propagation to prepare a system at a finite temperature.
   There are actually different techniques to do so, this is detailed in {doc}`/advanced/thermal_states`.

## Relaxing wave functions

### Calculating the ground state

At its heart, the relaxation solver is just a modified version [^ChebychevImag]
of the {py:class}`wavepacket.solver.ChebychevSolver`,
whose use is detailed in {doc}`chebychev_solvers`.
Most remarks in the reference apply here as well, in particular the spectrum guess and operator truncation.

We take the harmonic oscillator example of {doc}`chebychev_solvers`,

```{code-cell}
import numpy as np
import wavepacket as wp

grid = wp.grid.Grid(wp.grid.PlaneWaveDof(-10, 10, 128))
kinetic = wp.operator.CartesianKineticEnergy(grid, 0, mass=1, cutoff=35)
potential = wp.operator.Potential1D(grid, 0, lambda x: 0.5 * x ** 2, cutoff = 35)
hamiltonian = kinetic + potential

solver = wp.solver.RelaxationSolver(hamiltonian, 1.5, (0, 70))
```

In contrast to the real-time evolution, however, we always apply the Hamiltonian from the left,
also for density operators.
Hence, the solver expects the Hamiltonian directly, not a Schrödinger equation or Liouvillian.

Because of the exponential dynamics, most details of the initial state are irrelevant,
as long as it has *some* overlap with the true ground state.
In practice, you just take something simple like a Gaussian somewhere around the potential minimum and
let the solver optimize the state.

Typically, we do not care deeply about intermediate results, only about the final ground state,
and only use the intermediate results to monitor convergence.
Note, though, that the result needs to be normalized before further use!

```{code-cell}
def relax(solver, psi0):
    for t, psi in solver.propagate(psi0, t0=0, num_steps=5):
        psi_normalized = wp.grid.normalize(psi)

        energy = wp.operator.expectation_value(hamiltonian, psi_normalized).real
        energy2 = wp.operator.expectation_value(hamiltonian*hamiltonian, psi_normalized).real
        print(f"t = {t}:  E = {energy},   dE^2 = {energy2 - energy**2:.4}, |psi|^2 = {wp.grid.trace(psi):.4}")
    return psi

# to demonstrate convergence, let us take a Gaussian that is too wide (rms=1 would be exact)
psi0 = wp.builder.product_wave_function(grid, wp.Gaussian(rms=2))
relax(solver, psi0);
```

Note how the energy uncertainties decay exponentially by about a factor of five for each time step.
In the example here, we are basically converged after about three steps,
but the relaxation is usually the least expensive part of the calculation,
so you can afford to spend a few more CPU cycles on a converged solution.

Beware that the initial state must have overlap with the ground state!
This most often fails because the symmetry is incorrect.
In our harmonic oscillator example, the ground state has even parity, {math}`\psi(x) = \psi(-x)`.
Let us check what happens if we choose an initial state with odd parity, for example a sign function:

```{code-cell}
psi0 = wp.builder.product_wave_function(grid, lambda x: np.sign(x))
relax(solver, psi0);
```

Neat, isn't it?
We converged, but to the first excited state, because the ground state has exactly zero overlap with an odd function.
In general, you should, however, not rely on this behavior.
If the wave function has or acquires even the smallest asymmetry, for example through finite-precision arithmetics,
you will eventually converge to the ground state, albeit slowly.

However, you can exploit this behavior in a different scenario.
The convergence time is given by the smallest excitation energy,
because that state's contribution decays second-slowest after the ground-state contribution.
In some systems, this excitation energy is small, leading to rather slow convergence.
A prototypical example is the double-well potential, where low-energy states always come in pairs with small energy
gaps within the pair.
In most such cases, however, the two close-lying energy eigenstates differ by their symmetry.
By selecting an initial state with the same symmetry as the ground state, you can already suppress the
unwanted state from the start, leading to faster convergence.

### Calculating excited states

With a bit more machinery, we can also calculate excited energy eigenstates.
Imaginary-time propagation works because contributions from excited energy eigenstates decay faster than the
ground-state contribution.

Now we do the following:

1. We calculate the ground state with imaginary-time propagation.
2. We run another imaginary-time propagation, and repeatedly remove the ground state component.

What happens? Well, the resulting wave function is dominated by its slowly decaying ground state component.
If we remove it, we are left with the second-slowest decaying component, which is the first excited state.
In this fashion, we can calculate successive excited states.
Let us write the code before a discussion:

```{code-cell}
def print_data(state, n, step):
    energy = wp.operator.expectation_value(hamiltonian, state).real
    energy2 = wp.operator.expectation_value(hamiltonian*hamiltonian, state).real
    print(f"step {step}: E_{n} = {energy},   dE^2 = {energy2 - energy**2:.4}")

def relax_v2(solver, found_states):
    psi = wp.builder.product_wave_function(grid, wp.Gaussian(x=1, rms=2))
    print_data(psi, len(found_states), 0)

    for step in range(5):
        psi = solver.step(psi, 0)
        psi = wp.grid.orthonormalize(found_states + [psi])[-1]
        print_data(psi, len(found_states), step+1)

    print("----------------------")
    return psi

ground = relax_v2(solver, [])
exc1 = relax_v2(solver, [ground])
exc2 = relax_v2(solver, [ground, exc1])
# and so on
```

A few notes about the code:

* I have reduced the number of steps to 5 to compress the output, and because the results are
  already sufficiently converged.
* Note that the initial state is now a *shifted* Gaussian. This is a deliberate choice. Because the eigenstates
  of the harmonic oscillator toggle between even and odd parity, we need an asymmetric initial state that contains
  contributions with both parities.
  If I had chosen an even or odd function as initial state, I would only get excited states of that parity.
* The code exploits a semi-internal detail of the orthonormalization function; it normalizes the first entry, then
  orthonormalizes the second etc., so that the last entry is our wave function with the other components removed.
* We manipulate the state while we evolve it in time. Therefore we can no longer `propagate()` in one go, but must
  explicitly step through the solution.

In theory, you can now go on to get arbitrary excited states with this technique.
In practice, you need a full propagation for each subsequent excited state, 
so this approach becomes very expensive very quickly.
Also, errors may add up in larger systems.
Any error in the ground state, because you did not propagate long enough, will lead to
an incorrect orthogonalization of the first excited state and so on.
The recovered states hence become more and more incorrect.
Unfortunately, it is difficult to quantify this latter error even with a lot of hand waving, and at least this harmonic
oscillator example seems pretty robust against this issue.

In conclusion, you may only want to recover a few excited states of the Hamiltonian and monitor the
convergence carefully.

## Propagating density operators in imaginary time

The calculation of the density operator for finite temperatures is derived in a roundabout way.
The Liouville-von-Neumann (LvNe) equation

```{math}
    \frac{\partial \hat \varrho}{\partial \beta} = - \mathcal{L}(\hat \varrho) = - \hat H \hat \varrho
        \qquad\mathrm{with}\qquad \hat \varrho(\beta = 0) = \hat 1
```

can be shown, in analogy to ordinary differential equations and Schrödinger equations, to have the solution

```{math}
    \hat \varrho (\beta) = \mathrm{e}^{-\beta \mathcal{L}} (\hat \varrho(0))
        = \mathrm{e}^{-\beta \hat H}.
```

So we just set up the LvNe, and evolve the unit density in time until the requested inverse temperature.
The step should be chosen to fit the requested inverse temperature(s) and keep the alpha value
in a reasonable range.
as an example, let us calculate the partition sum of our harmonic oscillator.

```{code-cell}
delta_beta = 1.0
solver = wp.solver.RelaxationSolver(hamiltonian, delta_beta, (0, 70))
rho =  wp.builder.unit_density(grid)

for step in range(5):
    rho = solver.step(rho, 0)

    Z = wp.grid.trace(rho)
    print(f"beta = {delta_beta * (step+1)},  Z = {wp.grid.trace(rho):.4}")
```

Because imaginary-time propagation only makes sense for a particular Liouvillian, you supply again
the Hamiltonian directly, not some Liouvillian.
Side note: because this Liouvillian has the same spectrum as the Hamiltonian, we can use the
same RelaxationSolver for density operators and wave functions, in contrast to the real-time ChebychevSolver.

[^ChebychevImag]: R. Kosloff and H. Tal-Ezer, Chem. Phys. Lett. 127(3) 223 (1986)
                       <https://openscholar.huji.ac.il/sites/default/files/ronniekosloff/files/cpl86.pdf>

