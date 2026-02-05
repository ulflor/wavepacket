---
file_format: mystnb
kernelspec:
    name: python3
---

# How to represent states at finite temperature

## Introduction

Sometimes, the system under study is not in a well-defined initial state, but has a well-defined temperature.
In such a case, we need either a density operator or an ensemble of wave functions to describe our system.
The textbook expression for a thermal state is a density operator

```{math}
    \hat \varrho_\mathrm{th} = \frac{1}{Z} \ \mathrm{e}^{- \beta \hat H}
    \qquad \mathrm{with} \qquad
     Z = \mathrm{Tr} ( \mathrm{e}^{-\beta \hat H} )
```

with the inverse temperature {math}`\beta = 1 / (k_B T)`
in terms of Boltzmann's constant and the system temperature.
Note that we use atomic units throughout this text.

Typically, the thermal state itself is not terribly interesting.
What we want to calculate is some expectation value, often the response to some perturbation,

```{math}
:name: eq_response
    \langle \hat R(t) \rangle = \mathrm{Tr}( \hat R(t) \hat \varrho_\mathrm{th} )
         = \mathrm{Tr}( \hat U^\dagger(t, t_0) \hat R \hat U(t, t_0) \hat \varrho_\mathrm{th} )
```

where the calculation of the expectation value may involve some time evolution of the density operator
with additional terms such as laser fields.

In this document, we present three different approaches to represent a thermal state
and to calculate expectation values.
For simplicity, we skip the time propagation and only calculate the system energy.
However, more complex setups should be straight forward extensions of this simple example.

## The system under study

As example system, we calculate the temperature-dependent energy of a Morse oscillator.
As parametrization, we take the values for an OH radical.
The system is simple and fast, yet not entirely trivial.
Because we are going to use the {py:class}`RelaxationSolver`,
we truncate our operators well beyond the dissociation energy.
This has the side effect of providing us bounds for the spectrum of the Hamiltonian that we can use later.

```{code-cell}
import numpy as np
import wavepacket as wp

D_e = 0.1994;
r_e = 1.821;
alpha = 1.189;
mass = 1728.539

def morse_potential(x):
    return D_e * (1 - np.exp(-alpha * (x - r_e))) ** 2

dof = wp.grid.PlaneWaveDof(0.7, 7, 128)
grid = wp.grid.Grid(dof)

hamiltonian = (wp.operator.CartesianKineticEnergy(grid, 0, mass=mass, cutoff=1)
               + wp.operator.Potential1D(grid, 0, morse_potential, cutoff=1))
spectrum = (0, 2)
beta_vals_to_calculate = [100, 25, 10]
```

Already the largest inverse temperature of 100 a.u corresponds to over thousand Kelvin.
The chosen values should not be understood as anything realistic, but as examples corresponding to

* beta = 100 as temperature on the order of the excitation energy
* beta = 25 as temperature larger than the excitation energy
* beta = 10 as temperature on the order of the binding energy (half the dissociation energy)

## Method I: Using the thermal density operator directly

### Theory

The first approach directly constructs the thermal density operator, {math}`\hat \varrho_\mathrm{th}`.
The exponentiation is difficult, therefore we use a roundabout way, see also {doc}`/tutorials/relaxation`.
It can be checked that, the initial-value problem

```{math}
    \frac{\partial \hat \varrho}{\partial t} = - \hat H \hat \varrho(t)
        \qquad \mathrm{with} \quad
        \hat \varrho(0) = \hat 1
```

has the solution

```{math}
    \varrho(t = \beta) = \mathrm{e}^{-\hat H \beta}
```

which is exactly our thermal density operator except for the trivially calculated normalization factor.
The differential equation is solved by the {py:class}`wavepacket.solver.RelaxationSolver`, so we only need to
prepare the initial state and let the solver run.

### Implementation

The implementation closely follows the theory.
We propagate the state and calculate the corresponding expectation values.

```{code-cell}
for beta in beta_vals_to_calculate:
    solver = wp.solver.RelaxationSolver(hamiltonian, beta, spectrum)

    unit_density = wp.builder.unit_density(grid)
    rho = solver.step(unit_density, 0)
    thermal_state = wp.grid.normalize(rho)
    
    Z = wp.grid.trace(rho)
    energy = wp.operator.expectation_value(hamiltonian, thermal_state).real
    
    print(f"beta = {beta}: Z = {Z:.4}   energy = {energy:.4}")
```

Note that the solver is mostly inefficient due to the small alpha values, but in practice you would prepare
the initial state exactly once, so we do not really need to care too much.
If you do not need the partition sum explicitly,
you can skip its calculation and simply normalize the resulting density operator.

### Conclusions

The calculation of a thermal density by imaginary-time propagation is the most straight-forward method.
It has a direct connection to the theory, where you also normally employ a density-operator formalism.
Finally, the implementation is much simpler than the alternative methods.

If you want to study open systems, you want a density operator to be able to describe interactions
with the environment, so this approach fits you perfectly and you are done.
If, however, you study closed systems and only needed to account for temperature, density operator are awkward.
if we look beyond the scope of this humble Python package, a density operator formalism might simply not exist,
for example when you study large systems with MCTDH.

Consider a system of N total grid points / basis functions.
To represent a density operator, you need store N^2 values, while a wave function only needs N values.
The cost of applying an operator is somewhere between pointwise multiplication (operators in DVR approximation)
and matrix multiplication (general, unoptimized operator).
For a density operator, your computational cost scales like N^2 to N^3, while the cost for a single wave function
the scaling is between N and N^2.

As a result, density operators are only applicable for rather small systems, otherwise the memory requirements
will kill you.
And their propagation in time is comparably slow compared to a small ensemble of wave functions.
So if we do not intrinsically *need* a density operator, we may want to avoid it if we can.
That is what the other approaches do.

## Method II: Using eigenstates of the Hamiltonian

### Theory

To motivate the use of energy eigenstates, let us use the cyclic property
and the definition of the trace to rewrite the [response expression](#eq_response).

```{math}
    \langle \hat R(t) \rangle &= \frac{1}{Z} \mathrm{Tr}( \hat R(t) \mathrm{e}^{-\hat H \beta} )\\
    &= \frac{1}{Z} \mathrm{Tr}( \mathrm{e}^{-\hat H \beta/2}
        \ \hat R(t) \ \mathrm{e}^{-\hat H \beta/2} )\\
    &= \frac{1}{Z} \sum_k \langle \psi_k | \mathrm{e}^{-\hat H \beta/2} \hat U^\dagger(t, t_0)
        \ \ \hat R \ \ \hat U(t, t_0) \mathrm{e}^{-\hat H \beta/2} | \psi_k \rangle
```

This equation directly suggests an alternative procedure for calculating thermal averages:

1. Choose *any* orthonormal basis.
2. For each basis function, first do an imaginary-time propagation up to half the inverse temperature,
   then propagate in real time up to the point where the observable is calculated.
3. For each such propagated wave function, calculate the expectation value, and sum up all the results.
4. Divide the sum by the partition sum. From its definition, we get the partition sum by calculating steps 1-3
   with the unit operator as response operator (so no real-time propagation).

Note that this procedure is valid for each and every orthonormal basis.
So, with some algebra, we succeeded in replacing the density operator by an ensemble of wave functions.
This helps with memory requirements,
because we only need to store one or few wave functions at a time instead of the full density operator.
However, we do not gain performance.
While a single wave function is faster to propagate, we need to do so for the full basis of N wave functions,
which turns out to be as expensive as propagating the density operator.

As a consequence, this scheme makes most sense if we can faithfully reproduce the results
using only few of the basis functions.
We then need to choose a basis where we know that the result converges (exponentially) quickly,
and one such set are the eigenfunctions of the Hamiltonian.
We find that

```{math}
    \mathrm{e}^{-\beta \hat H/2} \psi_k = \mathrm{e}^{-\beta E_k/2} \psi_k.
```

For energies {math}`E_k \gg 1/\beta \propto T`, the exponential term is essentially zero,
and such an eigenstate has a negligible contribution.
For sufficiently low temperatures, few eigenstates suffice, making this scheme rather efficient.
A further convenient benefit is that the exponential term is a simple calculation and requires no
explicit solver.


### Implementation

For simplicity, we choose to obtain the eigenstates here from diagonalising the Hamiltonian matrix.
This is silly performance-wise, but keeps the example focused.

Conceptually, we need two passes.
In a first pass, we calculate the partition sum as "expectation value of the unit operator".
Then we can calculate the expectation values of the response operator.
In practice, the partition sum can be trivially obtained from intermediate results.

```{code-cell}
energies_and_states = [val for val in wp.solver.diagonalize(hamiltonian)]

for num_states in [2, 4, 16, 64]:
    used_sample = energies_and_states[:num_states]
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"{num_states} states, max energy = {used_sample[-1][0]:.4}")
     
    for beta in beta_vals_to_calculate:
        Z = 0
        sum = 0
        
        for energy, psi in used_sample:
            relaxed_state = np.exp(-energy * beta / 2.0) * psi
            
            Z += wp.grid.trace(relaxed_state)
            sum += wp.operator.expectation_value(hamiltonian, relaxed_state).real
            
        print (f"beta={beta}: Z = {Z:.4}    E = {sum / Z:.4}") 
```

If we compare these values to the exact results for the density operator, we see a few trends.
Even for the lowest energy, we need to include several excited states;
even four states have a residual error of about 1%.
As soon as the thermal excitations {math}`k_BT` are similar in magnitude to the binding potential,
the number of required states grows substantially.
This situation is already the case for an inverse temperature of 25 (one fifth the dissociation energy)
An intuitive explanation is that as soon free or quasi-free states start to contribute,
there are a lot of them, and they compensate their small weights with numbers.

### Conclusions

The method based on energy eigenstates has superior efficiency compared to the density operator approach,
but is slightly more complex to implement.
For that reason, you will find it widely adopted in the literature.
However, it has two mighty limitations: you need to converge with few enough states, and you need
an efficient method to determine these states.
The principal feasibility of this scheme can usually be determined with back-of-an-envelope calculations
and a few test propagations.
In any case, you should always carefully monitor the convergence, not accept the result blindly.

The cost scales with the number basis functions that contribute to the requested quantity.
As soon as the thermal excitation probability {math}`\exp(-E \beta/2)` is large enough to populate weakly-bound
states, the number of wave functions for a converged result explodes.
Hence, this approach is most useful for *low temperatures* with {math}`k_BT` on the order of the
smallest excitation energy or less.
This is a rather common situation for molecular vibrations and typical experimental conditions.

Determining the eigenstates and -energies through diagonalizing the Hamiltonian,
as we did here, is usually out of question;
memory and computations are similar to those of the density approach of Method I.
You are probably better off sticking to the accurate density-operator formalism than blindly adding
complexity.

One solution is the use of imaginary-time relaxation to get the lowest few eigenstates.
This method is discussed in {doc}`/tutorials/relaxation`.
Curiously, the principal limitation of relaxation to a few excited eigenstates plays along very well
with the limitation of this method to a few excited eigenstates.

A completely different solution is to use another basis;
as long as you cover (most of) the contributing low-energy subspace, you get converged results.
As an example, take a rotating, vibrating molecule:
Calculating the ro-vibrational eigenstates is tricky,
but calculating the rotational *or* vibrational eigenstates is easy.
So you might just construct your basis from products of rotational and vibrational eigenstates.
This method has some caveats; for example you might want to relax the basis functions explicitly,
which might be too expensive, but it principally works equally well.

## Method III: Using random wave functions

Note: We use the notation from our own publication[^random], which may be a bit uncommon.
See the references therein for other works and also applications.

### Theory

The derivation is similar in spirit to Method II, but instead of an orthonormal basis,
we use special random wave functions,

```{math}
   |\tilde \psi\rangle = \sum_k \tilde c_k |\psi_k \rangle
      \qquad \mathrm{where} \quad E[\tilde c_k \tilde c_l^\ast] = \delta_kl
      .
```

To obtain a random wave function, we choose any basis,
and assign each basis function a random number as coefficient.
Here and in the following, we use a tilde to denote random variables.
Besides various algebras, these allow the calculation of an expected value, denoted "E[]", which then is a
"normal" number or object.
The expected value calculation trivially commutes with non-random factors,
and for practical applications boils down to an ensemble average.

We are particularly interested in normalized, uncorrelated coefficients, because the projection operator becomes
the normal unit operator under averaging,

```{math}
    E[ |\tilde \psi \rangle \langle \tilde \psi | ]
        = E[ \sum_{k,l} \tilde c_k \tilde c_l^\ast | \psi_k \rangle\langle \psi_l |]
        = \sum_{k,l} E[\tilde c_k \tilde c_l^\ast] |\psi_k \rangle \langle \psi_l|
        = \sum_k |\psi_k \rangle \langle \psi_l |
        = \hat 1 
        .
```

With these preparations, we can now take again our [response expression](#eq_response), split the exponential,
insert and expand a unit operator, rearrange the scalar products, and finally use the resolution of identity
to get rid of the trace summation,

```{math}
    \langle \hat R(t) \rangle &= \frac{1}{Z} \ \sum_k \langle \psi_k | \hat R(t)
        \mathrm{e}^{-\hat H \beta/2} \ \hat 1 \  \mathrm{e}^{-\hat H \beta/2}
        | \psi_k \rangle
        \\
    &= \frac{1}{Z} \ E[ \langle \tilde \psi| \mathrm{e}^{-\hat H \beta/2}
        \ \sum_k |\psi_k\rangle\langle\psi_k| \ \hat R(t)
        \mathrm{e}^{-\hat H \beta/2} |\tilde \psi \rangle ]
        \\
    &= \frac{1}{Z} \ E[ \langle \tilde \psi \mathrm{e}^{-\hat H \beta/2} \hat U^\dagger(t, t_0) \hat R
        \hat U(t, t_0) \mathrm{e}^{-\hat H \beta/2} |\tilde \psi \rangle ]
```

Similar to the previous discussion, this formula gives a simple interpretation
for the calculation of the response:

1. Generate an ensemble of random wave functions (with normalized, uncorrelated coefficients).
   More on that in a moment.
2. For each random wave function, propagate for half the inverse temperature in imaginary time.
3. Propagate the result in real time as required, and calculate the expectation value of the response operator.
4. Calculate the ensemble average, i.e., the mean, of the ensemble of the expectation values.
5. Divide by the partition sum. The partition sum can be calculated from steps 1 to 4 using a unit operator
   (i.e., just calculating the average trace).

On first view, this method may look like a slightly deranged variant of the previous method II.
So how, and especially why and when does it work?
The hand-waving answer: The key differences to method II are that
(i) the ensemble of random wave functions is not orthogonal (bad),
and (ii) that it covers the whole state space with a few wave functions (good).

Non-orthogonality is uniformly bad; it means for example that you may count contributions twice,
and this problem is only suppressed with large enough samples.
The spreading can be advantageous; instead of requiring many rigorous eigenstates, you just construct a
coherent superposition and get "all results in one go".
Depending on the relative importance of these two factors, the results are terrific or terrible.

The final bit of theory concerns the construction of the random wave functions.
A convenient basis is the DVR; that is, we just assign a random coefficient to each grid point,
that saves complicated transformations.
Then, we only need any scheme that is uncorrelated and normalized,
{math}`E[\tilde c_k \tilde c_l^\ast] = \delta_{kl}`.
A simple scheme assigns each coefficient a complex number {math}`\exp(i \tilde \phi)`
where the phase is drawn uniformly from the interval {math}`\phi \in [-\pi, pi]`.

### Implementation

The overall implementation is similar to the one with energy eigenstates with three differences.

1. Instead of calculating eigenstates, we generate random wave functions using
   built-in functionality of Wavepacket.
2. We need to explicitly propagate in imaginary time, because we have no energy eigenstates.
3. We do not sum the individual contributions, but average.

```{code-cell}
rng = np.random.default_rng(42)

for num_states in [1, 2, 4, 16, 64]:
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"{num_states} states")

    for beta in beta_vals_to_calculate:
        solver = wp.solver.RelaxationSolver(hamiltonian, beta/2, spectrum)
        Z = 0
        sum = 0

        for n in range(num_states):
            psi_0 = wp.builder.random_wave_function(grid, rng)
            relaxed_state = solver.step(psi_0, 0)

            Z += wp.grid.trace(relaxed_state) / num_states
            sum += wp.operator.expectation_value(hamiltonian, relaxed_state).real / num_states

        print (f"beta={beta}: Z = {Z:.4}    E = {sum / Z:.4}")
```

If you compare these number with each other and the exact results from the density operator (method I),
you can simultaneously see the good *and* the bad things about random wave functions.
Even with a single random wave function, the results are in the correct ballpark, with only a few 10% deviation.
The non-orthogonality does not seem to have as much effect as one might fear.
On the hand, even for as much as 64 random wave functions, the results are good, but not terrific.

### Discussion

The implementation demo already shows the good parts of random thermal wave functions.
It is a very cheap way to get reasonable results.
This has also been found in the literature again and again (see the references in [^random]):
You get pretty good results with ensembles as little as ten random wave functions.
And the results are uniformly ok for very different temperatures.

There are two caveats, though.

One is rather theoretical: Pseudo-random number generation even in 2025 is not a solved problem,
all existing generators have issues.
It is unlikely, but principally possible that your "random" numbers are less random than you think.
Such problems are difficult to notice, leave alone debug for a non-expert, so the only advise is to be
critical of ones results.

More severely, the results converge very slowly with an increasing number of random wave functions.
Consider the rigorous method II with energy eigenstates: It may be infeasible, but once you have sampled the
relevant subspace, the exponential factor guarantees that additional contributions drop off quickly;
you get at some point *exponential* convergence.
With random wave functions, once you have added enough functions to reasonably sample the relevant subspace,
you get *stochastic* convergence, and the relative error drops with the square root of the number of
wave functions.

This suggests the following algorithm for converging:

1. Always run multiple samples (e.g., different random seeds) to get a feeling for the spread of the
   results. As long as this spread drops noticeably, you might want to increase the sample size.
2. Once the results do not significantly improve further, you stop.
   Adding more functions will no longer improve the accuracy of your solution, just increase the cost.
   If the final accuracy is not good enough for your purpose,
   you probably cannot use random wave functions.

[^random]: U. Lorenz and P. Saalfrank, JCP 140:044106 (2014)
