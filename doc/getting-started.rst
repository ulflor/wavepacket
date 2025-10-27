Getting started
===============

To demonstrate the usage of Wavepacket, let us dive right into
a simulation of a one-dimensional free particle.

.. code-block:: python

    import math
    import wavepacket as wp

    dof = wp.grid.PlaneWaveDof(-20, 20, 128)
    grid = wp.grid.Grid(dof)

    hamiltonian = wp.operator.CartesianKineticEnergy(grid, 0, mass=1.0)
    equation = wp.expression.SchroedingerEquation(hamiltonian)

    psi0 = wp.builder.product_wave_function(grid,
                                            wp.Gaussian(x=0, p=0, fwhm=1))

    solver = wp.solver.OdeSolver(equation, dt=math.pi/5)
    for t, psi in solver.propagate(psi0, t0=0, num_steps=20):
        # do something with the solution psi

This program already highlights the basic structure of a Wavepacket simulation:

1. You first need to set up a grid / basis expansion for your system.
   For that, you need to define the grid along each degree of freedom,
   and then form the multidimensional grid as the direct product of the
   one-dimensional grids. Note that Wavepacket uses exclusively the DVR / pseudo-spectral method,
   see :doc:`representations` or [#dvr]_.
2. Given a grid, you can define your equations of motion.
   Again, this step consists of two parts: First, you define all relevant
   operators, for example the Hamiltonian, then you wrap these operators
   into components for your equation of motion. For Wavepacket dynamics,
   you usually want to setup a Schroedinger equation, but for density operators,
   you may compose your equation from various commutators, anticommutators,
   Lindblad Liouvillians etc.
3. Next you specify your initial state that you want to evolve in time.
4. Finally, you set up the solver for your equations of motion, and propagate
   your initial state in time [#solvers]_.

This programmatic approach is rather complex and verbose when compared to more rigid programs.
For example, Matlab Wavepacket only requires you to set the various parameters, which makes the scripts
shorter and simpler.
However, the Python version allows more flexibility when you use more complex setups.
For example, you can almost seamlessly switch between density operator and wavepacket descriptions,
see the :doc:`/tutorials/schroedinger_cat` tutorial, or you can propagate an ensemble of random thermal
wave functions to replicate a thermal system (TODO: translate demo).


.. [#dvr] See the explanation of the DVR method in the
   `Wavepacket wiki <https://sourceforge.net/p/wavepacket/wiki/Numerics.DVR>`_.
.. [#solvers] See the
   `discussion of ODE solvers <https://sourceforge.net/p/wavepacket/cpp/blog/2021/04/convergence-2/>`_.
