---
file_format: mystnb
kernelspec:
    name: python3
---

# Using coupled channels

This web page can be downloaded as notebook: {nb-download}`channels.ipynb` (Jupyter)
or {download}`channels.md` (Markdown)

This tutorial gives an overview on the use of coupled channels in Wavepacket.
While channels are mostly just ordinary degrees of freedom,
they have some specialties:

* They have no "natural" representations; a channel is just an index.
    In contrast, for example a rotational degree of freedom has a "natural"
    representation in real space and in the angular momenta.
* The individual "points" (channels) have names,
    and we commonly want to reference individual channels, not use functions of grid points.
    As an example, we commonly want to project onto individual channels;
    this workflow is rather uncommon for, say, individual angular grid points.
* Some functionality, such as expectation value logging or plotting, should be done per channel.

## Introduction

A common motivation for the channel concept comes from the Born-Huang expansion:
When studying the dynamics of molecules, we split the degrees of freedom into
atomic or nuclear degrees of freedom $\mathbf{R}$, and into electronic degrees of freedom $\mathbf{r}$.
We further separate the full Hamiltonian into the atomic kinetic energy and an
electronic Hamiltonian that parametrically depends on the atomic coordinates,

\begin{gather*}
    \hat H(\mathbf{R}, \mathbf{r}) = \hat T_\mathbf{R}
        + \hat H_\mathrm{el}(\mathbf{r}; \mathbf{R}),
    .
\end{gather*}

"Normal" quantum chemistry, such as DTF or CI methods solve the electronic problem,
yielding electronic eigenstates and -energies

\begin{gather*}
    \hat H_\mathrm{el}(\mathbf{r}; \mathbf{R}) \varphi_n(\mathbf{r}; \mathbf{R})
        = V_n(\mathbf{R}) \varphi_n(\mathbf{r}; \mathbf{R})
\end{gather*}

We can expand our full wave function in these electronic eigenstates,

\begin{gather*}
    \Psi(\mathbf{R}, \mathbf{r}, t) = \sum_n \psi_n(\mathbf{R}, t) \varphi_n(\mathbf{r}; \mathbf{R})
\end{gather*}

Plugging this expansion into the full Schrödinger equation,
multiplying with $\varphi_k^\ast$ and integrating over the electronic coordinates yields

\begin{gather*}
    \imath \frac{\partial \psi_k(\mathbf{R}, t)}{\partial t} 
        = (\hat T_\mathbf{R} + V_n(\mathbf{R})) \psi_k(\mathbf{R}, t) + \sum_m \hat C_{km} \psi_m(\mathbf{R}, t)
\end{gather*}

The couplings C originate from the kinetic energy operator $\hat T_\mathbf{R}$ acting on
the $\mathbf{R}$-dependence of the electronic eigenstates (adiabatic couplings), or from
external couplings, for example from coupling electronic states with electromagnetic fields.

This formulation reduces the time evolution of the molecule
to that of a set of coupled atomic wave functions, the so-called channels.
This situation is common in molecular systems, which is why this is an important topic.
The only, though common, exception arises if you can neglect the couplings (Born-Oppenheimer approximation).
In that case, you typically ignore all channels but the initially populated electronic ground state.

## Model

As a model system for this tutorial, we choose the laser-excitation from the electronic ground state
to an excited electronic state of the HCl+ cation.
We ignore rotation, so we end up with two channels (electronic states),
and the interatomic distance as sole atomic coordinate.
Adiabatic couplings are neglected, but the channels have an electronic dipole coupling.

The two potential energy surfaces and the dipole coupling are interpolated from tabulated data,
taken from the original Matlab version's demo (click to expand for details)

```{code-cell}
:tags: [hide-input]
import numpy as np
import scipy as sp

data_pot_X = [(01.50, -1.46623), (01.60, -1.58853), (01.70, -1.67848), (01.80, -1.74431),
              (01.90, -1.79187), (02.00, -1.82554), (02.10, -1.84866), (02.20, -1.86377),
              (02.30, -1.87282), (02.40, -1.87728), (02.50, -1.87831), (02.60, -1.87677),
              (02.70, -1.87335), (02.90, -1.86281), (03.10, -1.84956), (03.30, -1.83532),
              (03.50, -1.82110), (03.75, -1.80417), (04.00, -1.78866), (04.50, -1.76250),
              (05.00, -1.74273), (05.50, -1.72858), (06.00, -1.71906), (07.00, -1.70982),
              (08.00, -1.70714), (09.00, -1.70640), (10.00, -1.70615), (12.50, -1.70599),
              (15.00, -1.70594), (17.50, -1.70592), (20.00, -1.70591)]
data_pot_A = [(01.50, -1.27768), (01.60, -1.40200), (01.70, -1.49469), (01.80, -1.56401),
              (01.90, -1.61571), (02.00, -1.65410), (02.10, -1.68242), (02.20, -1.70312),
              (02.30, -1.71805), (02.40, -1.72860), (02.50, -1.73583), (02.60, -1.74052),
              (02.70, -1.74329), (02.90, -1.74473), (03.10, -1.74264), (03.30, -1.73850),
              (03.50, -1.73322), (03.75, -1.72587), (04.00, -1.71833), (04.50, -1.70413),
              (05.00, -1.69227), (05.50, -1.68317), (06.00, -1.67664), (07.00, -1.66976),
              (08.00, -1.66790), (09.00, -1.66770), (10.00, -1.66819), (12.50, -1.66856),
              (15.00, -1.66876), (17.50, -1.66888), (20.00, -1.66894)]
data_dip = [(01.50,  0.2556), (01.60,  0.2455), (01.70,  0.2309), (01.80,  0.2192),
            (01.90,  0.2073), (02.00,  0.1948), (02.10,  0.1820), (02.20,  0.1618),
            (02.30,  0.1564), (02.40,  0.1438), (02.50,  0.1316), (02.60,  0.1296),
            (02.70,  0.1080), (02.90,  0.0860), (03.10,  0.0662), (03.30,  0.0489),
            (03.50,  0.0340), (03.75,  0.0183), (04.00,  0.0051), (04.50, -0.0159),
            (05.00, -0.0306), (05.50, -0.0389), (06.00, -0.0422), (07.00, -0.0400),
            (08.00, -0.0313), (09.00, -0.0065), (10.00, -0.0049), (12.50, -0.0024),
            (15.00, -0.0012), (17.50, -0.0001), (20.00,  0.0000)]

def interpolate(data, points):
    x = np.array([x for x, _ in data], dtype=float)
    y = np.array([y for _, y in data], dtype=float)
    spline = sp.interpolate.splrep(x, y)
    return sp.interpolate.splev(points, spline)

def potential_X(points):
    return interpolate(data_pot_X, points)
def potential_A(points):
    return interpolate(data_pot_A, points)
def dipole(points):
    return interpolate(data_dip, points)
```

We also define a few constants that we use for reproducing the original demo's output.

```{code-cell}
amu = 9.10938215e-28 * 6.0221408e23
fs = 41.341
eV = 0.0367493
```

## Basic usage

Channels are treated as a special degree of freedom (DOF).
You can initialize a channel DOF either with the number of channels,
or by explicitly labeling the individual channels.
These labels can be optionally used to reference channels in a more readable way,
so choose preferably short, descriptive names

```{code-cell}
import wavepacket as wp

dof = wp.grid.PlaneWaveDof(1.5, 7.5, 128)
channels = wp.grid.ChannelDof(['X', 'A'])
grid = wp.grid.Grid([dof, channels])
```

It is theoretically possible to have more than one channel DOF defined on a grid,
but this is not well supported;
the second channel DOF should be ignored by the convenience functionality described next.

Two special operators exist for convenience:
{py:class}`wavepacket.operator.Channel` projects onto a specific channel, while
{py:class}`wavepacket.operator.Coupling` describes a symmetric coupling between two channels.
We use them to define the ground and excited state potentials and the dipole coupling operator.
The kinetic energy operator is the same for all channels, so it needs no additional dressing.
Note how the channels can be interchangeably denoted through the index or the name.

```{code-cell}
kinetic = wp.operator.CartesianKineticEnergy(grid, 0, mass=0.97989 / amu, cutoff=0.7)
pot_X = wp.operator.Potential1D(grid, 0, potential_X, cutoff=-1.2) * wp.operator.Channel(grid, "X")
pot_A = wp.operator.Potential1D(grid, 0, potential_A, cutoff=-1.2) * wp.operator.Channel(grid, 1)
hamiltonian = kinetic + pot_X + pot_A

# The laser field is extremely strong, but at least that gives a visible effect.
dip = wp.operator.Potential1D(grid, 0, dipole) * wp.operator.Coupling(grid, 0, "A")
laser = wp.operator.LaserField(grid, max_field=0.5, shape=wp.special.SinSquare(1.5*fs, 1.5*fs),
                               omega=3.52*eV, phi = np.pi/2)

eq = wp.expression.SchroedingerEquation(hamiltonian - dip * laser)
```

The same goes for the wave function.
We approximate the ground vibrational state with a Gaussian,
and place the initial state on the ground state ("X") channel.

```{code-cell}
psi0 = wp.builder.product_wave_function(grid, [
        wp.special.Gaussian(2.519868, rms=np.sqrt(2) * 0.153688),
        "X"
    ])
```

Now let us take a small detour: How can we get rid of the channels?
As a somewhat pointless example,
let us calculate the trace of the excited-state channel of the initial wave function.
You have two options for that.

First, we can recognize that the Channel operators are projectors.
By applying them onto a state, we project out the respective channel.

```{code-cell}
projector_A = wp.operator.Channel(grid, "A")
projected_state = projector_A.apply(psi0, 0.0)
print(f"Trace of excited channel is {wp.trace(projected_state)}.")
```

Alternatively, you can set up a transformation to get rid of the channel degree of freedom.
{py:class}`wavepacket.grid.ChannelProjectionTransformation` takes a state,
projects out the requested channel, and moves this projected state onto a target grid without the channel DOF.
This is comfortable for some usages;
for example the plotting functionality always gets rid of the channel DOF in that way.
The approach is inconvenient for everything that involves operators, however,
because these are not trivially transformed.

```{code-cell}
transformation = wp.grid.ChannelProjectionTransformation(grid)
transformed = transformation.transform(psi0, channel=1)
assert transformed.grid.dofs == [dof] 
print(f"Trace of excited channel is {wp.trace(transformed)}.")
```

## Plotting etc.

Both the standard log output and the plotting objects are aware of channel DOFs.
Instead of treating them like a normal degree of freedom, the state is inspected separately for each channel.

Following the recommendation in {doc}`relaxation`, we first relax the initial state for some time.
This improves the initial state only marginally here,
but would be important if the two channels were coupled from the start.

```{code-cell}
relaxation = wp.solver.RelaxationSolver(hamiltonian, 30, (-1.9, 0.3))
psi = psi0
for i in range(5):
    psi = relaxation.step(psi, 0)

solver = wp.solver.OdeSolver(eq, 0.6 * fs)
plotter = wp.plot.StackedPlot1D(6, psi0, pot_X+pot_A, hamiltonian)
plotter.xlim = [1.5, 4]
plotter.ylim = [-1.9, -1.45]

for t, psi in solver.propagate(psi0, 0.0, 5):
    plotter.plot(t, psi)
    wp.log(t, psi)
```

```{note}
If you have channel coupling, the eigenstates of the Hamiltonian,
even the ground state, usually cover multiple channels.
The population of excited state channels (squared wave function) is typically small, on the order of 1e-3,
but the interference terms with the ground state scale with the wave function magnitude,
reaching multiple percent.
Erroneously placing the initial wave function onto a single channel then causes fast oscillations
with a few percent amplitude, which has caused confusion more than once.

The solution is to always relax the wave function before propagating.
The relaxation drives the wave function towards the correct eigenstate,
and spreads the initial state over multiple channels.
```

The dynamics of the HCl+ cation are rather dull. 
The laser couples the channels and excites the ground state wave function to the electronic excited state.
Because the excited state has a different potential, the wave function starts to move.
On the simulation time scale of 3 femtoseconds, however, it does not get very far.

A final side note: The laser pulse is extremely strong here; it not only excites the ground state,
but even depopulates the excited state channel again.
This suggests that a first-order perturbation theory picture, where the molecule absorbs only one photon
to reach the excited state, is not sufficient.
You need to take multi-photon excitation into account and consider higher excited states as well.
In fact, the field is so strong that you should expect significant multiphoton ionisation.

So as usual: Always be critical of your simulation results.
