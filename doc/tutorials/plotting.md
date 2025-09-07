---
file_format: mystnb
kernelspec:
    name: python3
---

(tutorial-plotting)=

# Plotting wave functions and density operators

This tutorial teaches some basics about plotting Wavepacket states.

## Basics of plotting with Matplotlib

The standard for plotting under Python is Matplotlib, see `https://matplotlib.org` for the full documentation.
However, while Matplotlib is powerful and flexible, it does not behave uniformly in all environments.
Hence, before we plot any wave function, we should manage to plot a simple line first.

Consider the following code

```{code-cell}
import matplotlib.pyplot as plt

figure, axes = plt.subplots()
axes.plot([1, 2, 3], [1, 2, 3])
```

A figure is a widget on which things can be drawn and can contain one or more axes, which are the actual plotting
areas.

What happens if you execute this code?

* Jupyter notebooks implicitly show all figures when a cell is executed, so you see a plot.
  However, once a figure is shown, it becomes read-only, so you must produce the plot in one go inside a cell,
  and you must use a new figure in each cell.
* Under a Windows e.g., command line, you do not see a plot.
  You need to add an explicit `figure.show()` to open a new window containing the plot display.
* Under Linux when, say, running Python in an xterm, you do not see a plot.
  You should only get an empty window when typing `figure.show()`.
  Now it becomes interesting:
    * You need to show the plot and add an event loop to have it displayed.
      Either type `plt.show(block=True)` to block Python execution until the plot window is closed,
      or type `plt.pause(<pause_in_seconds>)` to show the plot for some time before resuming execution.
      After the pause is over, the plot window stops being updated.
    * Depending on your Python installation, you may get no plot, but a warning about a non-interactive backend.
      So you need to install an interactive backend.
    * What should work most of the time is a GTK-based backend.
      Install PyGObject with, e.g., `pip install PyGObject`. Probably Matplotlib will now use it as default.
      If not, set the environment variable `export MPLBACKEND=GTK3Agg`. Now you should finally get a plot display.
* Other environments, such as PyCharm of VSCode plugins may have their own particular behavior.

If you managed to show a simple plot, we can go on plotting the real stuff.
Wavepacket does not handle these behavior differences, so you typically have to `show()` the plot
after plotting some wave function.

## Demo system

For this plotting demo, we use a one-dimensional harmonic oscillator as example system.

```{code-cell}
import math
import wavepacket as wp

grid = wp.grid.Grid(wp.grid.PlaneWaveDof(-10, 10, 128))

kinetic = wp.operator.CartesianKineticEnergy(grid, 0, mass=1)
potential = wp.operator.Potential1D(grid, 0, lambda x: 0.5 * x ** 2)

psi_0 = (wp.builder.product_wave_function(grid, wp.Gaussian(-3, 0, rms=1))
        - wp.builder.product_wave_function(grid, wp.Gaussian(3, 0, rms=1)))

equation = wp.expression.SchroedingerEquation(kinetic + potential)
solver = wp.solver.OdeSolver(equation, dt=math.pi/5) 
```

## Wavepacket plotting helpers

Wavepacket offers utility classes to make plotting of states easier.
Their purpose is a useful visualisation of the dynamics with minimal setup.
These classes are opinionated and may not be as flexible and configurable as you need them.
If they do not fulfill your needs, you might want to write your own plotting code, as discussed in the next section.

As of version 0.2, the only available helper is a class that stacks plots on top of each other,
and which plots the potential and the state's density (offset by the energy of the state) together.
It is mainly intended for Jupyter notebooks, where plotting is slightly restricted.

The constructor gets the number of plots, some wave function for guessing defaults, and the
potential and hamiltonian, then you just call plot repeatedly to fill the individual plots.
If you plot more often than there are axes available, the last plot is overwritten.

```{code-cell}
stacked_plot = wp.plot.StackedPlot1D(6, psi_0, potential=potential, hamiltonian=kinetic+potential)
stacked_plot.conversion_factor /= 2

for t, psi in solver.propagate(psi_0, t0=0.0, num_steps=5):
    stacked_plot.plot(psi, t)

# stacked_plot.figure.show() or similar is needed outside of Jupyter notebooks
```

In the example here, we have reduced the density scale (density to energy units), to fit the large spikes on the plot.
Further customization includes `stacked_plot.ylim`, which gives the lower and upper range of the y (energy) axis,
or the same for the x-axis.

## Manual plotting

You need to write your own plotting functions if, for example, the default functionality does not cover your use case,
or you need a specific styling for your publication. This requires deeper access and familiarity with
Wavepacket data structures, but is well-supported.

As an example, let us assume you want to plot the density in the plane-wave expansion (the FBR).
Ideally, you write a plot function first.

```{code-cell}
import matplotlib.pyplot as plt

figure, axes = plt.subplots()

def plot_in_fbr(psi):
    # Note: using global variables can be highly problematic,
    # but you often need a lot of different variables for plotting,
    # so we use them instead of function parameters here.
    global figure, axes
    
    fbr_grid = psi.grid.dofs[0].fbr_points
    fbr_density = wp.grid.fbr_density(psi)
    axes.plot(fbr_grid, fbr_density, 'k-')
    
    # Magic numbers; basically guesswork to make the plot cover only the interesting area.
    # You want a fixed range for all plots to avoid the plot zipping around.
    axes.set_xlim(-5, 5)
    axes.set_ylim(0, 0.8)

    # Beautification to make the plot look complete.
    axes.set_xlabel("k (a.u.)")
    axes.set_ylabel("|\psi|^2")
    
    # again, outside of notebooks, you need to add 'figure.show()' or similar here

# Then you plot like
plot_in_fbr(psi_0)
```

Of course, you can go ahead and plot further states, we only skip this here because this code is executed in a Notebook.
Outside of notebooks, you could recycle the figure and use, e.g., `plt.pause(1)`, to make a crude animation.

## Exporting images and animations

You can save individual plots and animations.
For the former, just call `figure.savefig()`, for example with the StackedPlot from above

```{code-cell}
stacked_plot.figure.savefig(f"harmonic_oscillator_stacked.png")
```

Showing the figure is not required prior to saving.
