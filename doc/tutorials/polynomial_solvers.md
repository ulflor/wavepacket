---
file_format: mystnb
kernelspec:
    name: python3
---

# Using polynomial solvers (Chebychev, Relaxation)

This tutorial should give you enough background information to use the {py:class}`wavepacket.solver.ChebychevSolver`
and {py:class}`wavepacket.solver.RelaxationSolver`.

For those interested in a bit more in-depth knowledge, an addendum drills a bit deeper into some aspects.

## Overview: Polynomial solvers

The basic idea of polynomial solvers is to write the time evolution operator in a polynomial sum,

```{math}
\mathrm{e}^{-\imath \hat H \Delta t} \propto \sum_n c_n P_(-\imath \hat H \Delta t)
.
```

We gloss over a lot of technical details here, which we will only pick up in the addendum.
While the choice of polynomial is arbitrary, in practice, particular properties are desired.
In particular, Chebychev polynomials are a useful choice, because they converge continuously.

Compared to general purpose solvers like the Runge-Kutta procedure, the Chebychev solver has three advantages.
First, it is much faster, up to an order of magnitude.
Second, for large n the sum terms decay exponentially with n, which makes the solver extremely accurate.
And finally, because of the continuous convergence, you can (in theory) define an exact upper bound of the error
in the whole spectrum.
In practice, you can set a small threshold (say, 1e-12), sum up all terms with coefficients above the threshold,
and you are guaranteed that the result is accurate for all possible input states, no further tuning required.

On the other hand, there are two major drawbacks.
First, polynomial solvers fail for time-dependent systems, because the expansion contains time-ordered products,
and that makes the whole scheme too difficult to work with.
In particular those based on Chebychev polynomials also require real eigenvalues, hence closed systems.
Second, the expansion has a finite radius of convergence, hence you must supply reasonably accurate bounds
for the spectrum of the Hamiltonian, which can be cumbersome.
The spectrum of a Liouvillian differs from that of a Hamiltonian, so the solver cannot be used for
wave functions and density operators.
Finally, the solver is only efficient for large time steps (Note: This is actually wrong, interpolation is possible,
but has not been implemented for simplicity. If you do need interpolation, drop me a mail or raise a ticket!).

## Using the Chebychev solver for real-time time evolution

In order to use a ChebychevSolver, you need four pieces of information:

1. The expression that describes the equation of motion.
2. The spectrum of the Hamiltonian / Liouvillian.
3. The length of one elementary time step.
4. The cutoff, that is, the smallest coefficient included in the summation.

Only the spectrum has severe consequences if chosen wrong and is difficult to obtain here.
We will discuss that parameter in a moment.

The expression is obviously needed for all solvers, so nothing new here.
In general, you never need to touch the default cutoff value of 1e-12.
Even the use of questionably large values of 1e-6 speeds up typical calculation by something like 25%.
Finally, the time step should be something like {math}`80 / \Delta E`, where {math}`\Delta E` is the range of
the spectrum that you supply, but you need not be extremely precise.
Choosing a smaller value decreases efficiency, for example {math}`40/\Delta E` roughly doubles the computational cost.
Choosing much larger values, beyond {math}`150/\Delta E` may overflow double precision values and has diminishing
efficiency gains, about 30% in this example.
These rather small gains are usually not worth the trouble.

All these magic numbers are explained in the addendum. For using Chebychev solvers, just take them at face value.

### Determining the spectrum

As an example, let us take the simple harmonic oscillator example from {doc}`plotting`.

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

1.What is the spectrum of the Liouvillian relative to the spectrum of the Hamiltonian?
2.Given a Hamiltonian (Schrödinger equation), how can I estimate its spectrum?
3.Are there better ways to increase the efficiency of the solver?

#### What is the spectrum of the Liouvillian relative to the spectrum of the Hamiltonian?

If the Hamiltonian has a spectrum
{math}`[E_\mathrm{min}, E_\mathrm{max}] = [E_\mathrm{min}, E_\mathrm{min} + \Delta E]`, then the corresponding
{py:class}`wp.expression.CommutatorLiouvillian` has the spectrum {math}`[-\Delta E, \Delta E]`,
because the coherence terms between the fastest oscillations {math}`\sim \mathrm{e}^{\pm \imath \Delta E t}`
occur between the eigenstates with the largest and smallest eigenvalues, respectively.
An important corollary is that, in general, a {py:class}`wp.solver.ChebychevSolver` and
{py:class}`wp.solver.RelaxationSolver` can only be used for wave functions or density operators, not both.

#### Given a Hamiltonian (Schrödinger equation), how can I estimate its spectrum?

To estimate the spectrum, let us first point out that you do not need the exact minimum/maximum eigenvalue,
but only a lower/upper bound to it.
The estimated spectral range is inversely proportional to the time step and hence proportional to the
computational cost, so do not spend too much effort in figuring out the exact values.
With that said, a simple lower bound of the Hamiltonian is given by the smallest value of the potential,
that is, an energy of zero in the example.

For an estimate of the largest eigenvalue, we can use the power iteration: Repeatedly applying the Hamiltonian
to an almost arbitrary state converges exponentially towards the eigenvector with the largest (absolute) eigenvalue.
That is straight forward to do:

```{code-cell}
import math

psi = wp.builder.zero_wave_function(grid) + 1
for iteration in range(10):
    psi = equation.apply(psi, t=0)
    psi = psi / math.sqrt(wp.grid.trace(psi))

energy_guess = wp.operator.expectation_value(hamiltonian, psi).real
energy_guess = 1.2 * energy_guess
print(energy_guess)
```

The initial state should contain the highest-energy eigenstate as a non-negligible component, but is otherwise
rather arbitrary.
The factor of 1.2 is an arbitrary safety margin to compensate that we approach the highest-energy
eigenstate from below and may not have been converged yet.
Thus, we arrive at an estimate for the spectrum of about [0, 280], and a time step of around 80/280 = 2/7.
Let us plug it in and watch a Gaussian wave packet evolve.

```{code-cell}
psi0 = wp.builder.product_wave_function(grid, wp.Gaussian(-5, 0, rms=1))
x_op = wp.operator.Potential1D(grid, 0, lambda x: x)
solver = wp.solver.ChebychevSolver(equation, math.pi/10, (0, energy_guess))

for t, psi in solver.propagate(psi0, t0=0.0, num_steps=10):
    trace = wp.grid.trace(psi)
    x = wp.operator.expectation_value(x_op, psi).real
    
    print(f"t = {t:.4}, trace = {trace:.4}, <x> = {x:.4}")
```

For completeness, we can finally check what happens if we get the spectrum wrong. Let us say we cut at 200 a.u.
The footprint of our wrong spectrum is readily apparent if we print the trace.

```{code-cell}
bad_solver = wp.solver.ChebychevSolver(equation, math.pi/10, (0, 200))

for t, psi in bad_solver.propagate(psi0, t0=0.0, num_steps=10):
    trace = wp.grid.trace(psi)
    print(f"t = {t:.4}, trace = {trace:.4}")
```

#### Are there better ways to increase the efficiency of the solver?

So far, we have taken the Hamiltonian for granted and tried to figure out the spectrum.
A much better approach is to turn this procedure around and tailor the operators to the spectrum.
To motivate this change, let us print the average and standard deviation of the initial state's energy:

```{code-cell}
    energy = wp.operator.expectation_value(hamiltonian, psi0).real
    energy2 = wp.operator.expectation_value(hamiltonian*hamiltonian, psi0).real
    
    print(f"E = {energy},   dE^2 = {energy2 - energy**2}")
```

Even if we give room for several standard deviations, we can safely claim that only eigenstates with energy <= 30 a.u.
are relevant to carry this Gaussian.
This is much, much less than the Hamiltonian's spectral range.
In other words: 90% of the computing time is spent to ensure the correct propagation of high-energy states
that we actually do not care about at all.
This imbalance between the spectrum that the grid/operator can support and the spectrum that we need is rather
typical.

What we do then is simply to truncate the Hamiltonian, or rather the kinetic and potential energy operators.
Truncating the potential gives just a little.
Our Gaussian (x0 = -5, rms = 1) extends maybe up to x = 8, where the potential value is
{math}`8**2/2 = 32 \mathrm{a.u.}`, so we might truncate at 35 a.u., a little less than the maximum potential of 50 a.u.
As a good first guess, we can do the same for the kinetic energy, which supports _much_ larger energies.
The truncated system is now

```{code-cell}
kinetic = wp.operator.CartesianKineticEnergy(grid, 0, mass=1, cutoff=35)
potential = wp.operator.Potential1D(grid, 0, lambda x: 0.5 * x ** 2, cutoff = 35)
hamiltonian = kinetic + potential
equation = wp.expression.SchroedingerEquation(hamiltonian)
```

The spectrum of this Hamiltonian has a guaranteed upper limit of 35 + 35 = 70, one fourth of the original Hamiltonian.
This allows us to increase the time step / lower the computing cost by a factor of 4.

```{code-cell}
solver = wp.solver.ChebychevSolver(equation, 4 * math.pi/10, (0, 70))

for t, psi in solver.propagate(psi0, t0=0.0, num_steps=10):
    trace = wp.grid.trace(psi)
    x = wp.operator.expectation_value(x_op, psi).real
    
    print(f"t = {t:.4}, trace = {trace:.4}, <x> = {x:.4}")
```

## Relaxing a state in imaginary time

The Schrödinger equation in imaginary time is given by

```{math}
\frac{\partial \psi}{\partial t} = - \hat H \psi
```

with the formal solution in terms of energy eigenstates {math}`\phi_i`

```{math}
\psi(t) = \mathrm{e}^{-\hat H t} \psi(t=0) = \sum_i c_i \mathrm{e}^{-E_i t} \phi_i
```

As the formula demonstrates, each eigenstate decays exponentially depending on the energy.
Hence, for sufficiently long time evolution with repeated normalization, our solution is the ground state if the
initial state has any overlap with the true ground state.
Propagation in imaginary time is thereby similar to the power iteration where you repeatedly apply the
Hamiltonian to an almost arbitrary state and the eigenstate with the largest absolute energy grows the fastest.

Besides being a convenient tool to find the ground state, we can also use the same equation with a density operator.
Solving the initial-value problem

```{math}
\frac{\partial \rho}{\partial \beta} = - \hat H \rho \qquad \mathrm{where} \qquad \rho(\beta=0) = \mathbb{1}
```

has the thermal density operator as a solution up to a normalization:

```{math}
\rho(\beta) = \mathrm{e}^{-\beta \hat H}
```

So imaginary-time propagation with density operators is also a convenient way to get the thermal density operator
without an explicit exponentiation of the Hamiltonian.

### Relaxing a wave function

At its heart, the relaxation solver is just a modified version of the {py:class}`wavepacket.solver.ChebychevSolver`
implemented along [^ref-chebychev-imag].
As such, all the remarks regarding the ChebychevSolver hold.
In particular, you should carefully consider how much you want to truncate your Hamiltonian.
Let us repeat the demo example here:

```{code-cell}
import math
import wavepacket as wp

grid = wp.grid.Grid(wp.grid.PlaneWaveDof(-10, 10, 128))
kinetic = wp.operator.CartesianKineticEnergy(grid, 0, mass=1, cutoff=35)
potential = wp.operator.Potential1D(grid, 0, lambda x: 0.5 * x ** 2, cutoff = 35)
hamiltonian = kinetic + potential

solver = wp.solver.RelaxationSolver(hamiltonian, 1.5, (0, 70))
```

There are two noteworthy differences with respect to the real-time solver.
First, we always apply the Hamiltonian from the left, so the solver takes the Hamiltonian, not a
{py:class}`wavepacket.equation.Equation`.
Second, we do not care about the intermediate results.
Instead, we may have to normalize the wave function every now and again to avoid numeric inaccuracies.
Last, we do not care deeply about the initial state, as long as there is a significant overlap with the ground state.
A good guess starts with more overlap, but you do not need to be overly precise.

```{code-cell}
# to demonstrate convergence, let us take a Gaussian that is too wide (rms=1 would be exact)
psi = wp.builder.product_wave_function(grid, wp.Gaussian(rms=2))
for step in range(5):
    energy = wp.operator.expectation_value(hamiltonian, psi).real
    energy2 = wp.operator.expectation_value(hamiltonian*hamiltonian, psi).real
    print(f"step = {step}:  E = {energy},   dE^2 = {energy2 - energy**2:.4}")

    psi = solver.step(psi, 0)
    psi = wp.grid.normalize(psi)
```

In the example here, we are basically converged after about three steps.
Note however, that in general the convergence depends on the energy gap between the ground state and the
first excited state, because the latter is the component that is the slowest to be suppressed.

### Propagating density operators in imaginary time

The relaxation of density operators is essentially the same as for wave functions.
The step should be chosen to conveniently access the requested inverse temperature(s) and keep the alpha value
in a reasonable range

```{code-cell}
delta_beta = 1.0
solver = wp.solver.RelaxationSolver(hamiltonian, delta_beta, (0, 70))
rho =  wp.builder.unit_density(grid)

for step in range(5):
    rho = solver.step(rho, 0)

    Z = wp.grid.trace(rho)
    print(f"beta = {delta_beta * (step+1)},  Z = {wp.grid.trace(rho):.4}")
```

## Addendum: Some more theory

For more details, see the [original paper by Ronnie Kosloff [^ref-chebychev-real], [^ref-chebychev-imag].

For a time-independent closed system, the time evolution operator U is defined by

```{math}
    \psi(t+\Delta t) = \hat U(t + \Delta t, t) \psi(t) = \mathrm{e}^{-\imath \hat H \Delta t} \psi(t).
```

In a similar fashion, we can define a time evolution super operator for density operators and open systems by
using a Liouvillian in place of the Hamiltonian and a density operator instead of the wave function.

The basic idea of polynomial solvers is to expand this exponential in a convenient series of classical polynomials.
In principle, we could use any polynomials, for reasons that we discuss in a moment, Chebychev polynomials are used.
Our first problem is that the polynomial series converges only in a specific domain.
In terms of operators, this means that our Hamiltonian must have eigenvalues exclusively in the range [-1, 1].
Hence, our first step is a rescaling of the Hamiltonian:

```{math}
    \mathrm{e}^{-\imath \hat H \Delta t} = \mathrm{e}^{-\imath (E_{\mathrm{min}} + \Delta E / 2) \Delta t}
        \mathrm{e}^{-\imath \alpha \hat H_\mathrm{norm}}\\
    \mathrm{with}\\
    \alpha = \frac{\Delta E \  \Delta t}{2}\\
    \hat H_\mathrm{norm} = \frac{2}{\Delta E} (\hat H - E_\mathrm{min} + \frac{\Delta E}{2})
```

Here, {math}`E_\mathrm{min}, \Delta E` denote the Hamiltonian's smallest eigenvalue and the spectral range,
respectively.

It can be checked that the normalized Hamiltonian has eigenvalues in the requested range.
We then only expand the second exponential with the normalized Hamiltonian and the rescaled time {math}`\alpha`,
and obtain

```{math}
    \mathrm{e}^{-\imath \hat H_\mathrm{norm} \alpha} = \sum_{n=0}^N a_n(\alpha) T_n(-\imath \hat H_\mathrm{norm}),
```

where up to constants, the coefficients "a" are Bessel functions of the first kind and the functions "T" are
Chebychev polynomials of the first kind.

## Convergence

The huge advantage of Chebychev polynomials is the availability of an exact upper bound for the propagation error.
We start with two inequalities, see [^ref-abramowitz], equations 22.14.4 and 9.1.62 that readily generalize to
functions of operators:

```{math}
    |T_n(x)| \leq 1 (x \in [-1, 1]) \qquad \rightarrow \qquad \|T_n(\hat H_\mathrm{norm}) \psi\| \leq 1 (\|\psi\| = 1)\\
    |J_n(x)| \leq  \frac{|x/2|^n}{n!}
```

Now let us assume that we used the first N terms of the expansion and want to know the error when propagating a state.
We get for a normalized state

```{math}
    \| \mathrm{e}^{-\imath \alpha \hat H_\mathrm{norm}} \psi \ 
        - \ \sum_{n=0}^N a_n(\alpha) T_n(-\imath \hat H_\mathrm{norm}) \psi \|
        = \| \sum_{n=N+1}^\infty a_n(\alpha) T_n(-\imath \hat H_\mathrm{norm} \psi)\|
    \leq  \sum_{n=N+1}^\infty a_n(\alpha) \|T_n(-\imath \hat H_\mathrm{norm} \psi)\|
    \leq \sum_{n=N+1}^\infty a_n(\alpha)
    \leq \sum_{n=N+1}^\infty \frac{(\alpha/2)^n}{n!}
```

We do not care about the exact value of the error, just note two properties of the expression:

1. There is an upper bound for the error independent of the initial state. That is, the convergence is continuous,
   there are no individual states for which the approximation is poor.
2. For large enough N, the right-hand side decreases at least exponentially with increasing N.

In practice, we do not aim for a specific value of (the upper bound of) the propagation error,
but truncate the series as soon as the Bessel function has reached a certain cutoff.
Due to the monotonous decrease of the Bessel function,
this cutoff can still be interpreted as an order of magnitude estimate of the error, but is easy to calculate.

## Efficiency of the Chebychev solver and good alpha values

Now we can finally discuss good alpha values for acceptable efficiencies.

The computational cost is essentially proportional to the order of expansion n;
the Chebychev polynomials are calculated through a recursion relation, and each recursion requires one expensive
evaluation of the Hamiltonian / Liouvillian.
We determine the order of expansion indirectly by requiring the Bessel functions to drop below a certain threshold.
The size of the time step, that is, the gain, however, is proportional to the alpha value, i.e., the
argument to said Bessel function.

So we can rephrase the question: At which value of alpha do we get optimal efficiency?
Let us plot the behaviour of the Bessel functions for different values of alpha

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
import scipy

alpha, n = np.meshgrid([20, 40, 60, 80], np.arange(120))
y = np.abs(scipy.special.jv(n, alpha))

fig, ax = plt.subplots()
ax.set_ylim(1e-12, 1)
ax.semilogy(n, y)
```

We can clearly see that the Bessel functions are significant up to {math}`n \approx \alpha`, then they decay rapidly,
approaching values of 1e-12 within roughly 30 orders.
We have "pay" for this decay anyway to get a well-converged solution, so a good alpha value should be larger than
this constant decay.

As an example, let us compare alpha values of 20 and 60.
For the former, we need about 50 polynomials for a time step, but our time step is shorter, so we need to propagate
three times. Altogether that makes approximately 150 applications of the Hamiltonian.
For the latter, we need only a single time step, and approximately 90 polynomials, hence about 90 applications of
the Hamiltonian.
Thus, the latter alpha value is about 60% faster than the former.
The turnaround is approximately when alpha becomes larger than this constant decay, hence the proposition of using
an alpha value of at least 40 to avoid inefficiencies.

Note that as the values become much larger than 30, the additional efficiency gains become marginal.
The order of expansion increases roughly linearly with alpha, so the larger time steps are exactly compensated by
the larger order of expansion.
However, large orders of expansion may lead to artefacts from finite-precision arithmetic,
therefore I would recommend not increasing alpha beyond something like 100, but that is an equally vague limit.

As an additional side note, we want to remark that for normal use the precision has remarkably little influence on
the efficiency.
Even if we go for embarrassingly large values of 1e-6, the required polynomial orders are reduced only by about 10,
so there is little gain in changing the default value of 1e-12.

[^ref-abramowitz]: M. Abramowitz and I. Stegun "Handbook of Mathematical Functions"
[^ref-chebychev-real]: R. Kosloff, J. Phys. Chem. 92(8), 2087 (1988)
https://doi.org/10.1021/j100319a003
[^ref-chebychev-imag]: R. Kosloff and H. Tal-Ezer, Chem. Phys. Lett. 127(3) 223 (1986)
https://doi.org/10.1016/0009-2614(86)80262-7
