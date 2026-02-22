Architecture overview
=====================

Common concepts
---------------

The focus of Wavepacket is on flexible setups of simple quantum systems.
To get there, the code ended up with a few common concepts.

* To abstract away the difference between wave functions and density operators,
  a class :py:class:`wavepacket.grid.State` has been introduced.
* Most utility functions and classes can operate on wave functions as well as
  density operators transparently.

  Where the difference can lead to confusion and errors, functionality may not be offered.
  For example, the wave function norm is the square root of the
  of the corresponding density operator (trace) norm. This has been extremely confusing
  when switching between the two objects, so Wavepacket does not offer a `norm()`
  function, it offers :py:func:`wavepacket.trace` instead,
* Classes in Wavepacket are usually immutable after creation.
  This allows you to recycle objects without side effects; for example,
  when you have set up a Hamiltonian for the field-free case, you can trivially use it
  also sum it with the laser interaction to get the Hamiltonian  for the system with a laser field.
  All classes and packages follow a strict hierarchy as described in the subsequent section.


Packages
--------

Wavepacket is split into several subpackages with hierarchical dependencies.
From lowest to highest layer, these are:

:py:mod:`wavepacket.grid`
    Contains all classes that describe the representation of a state (the grid),
    These include degrees of freedom, :py:class:`wavepacket.grid.Grid` itself,
    and the class :py:class:`wavepacket.grid.State`.

:py:mod:`wavepacket.builder`
    Contains functions to create an initial wave function or density operator.
    These functions require a grid on which to create the state.

:py:mod:`wavepacket.special`
    Contains special functions. These are mostly typical pulse shapes
    like squared sine shapes, or typical shapes for initial wave functions,
    such as Spherical harmonics.

:py:mod:`wavepacket.operator`
    Contains operator classes. Operators are always defined on a specific grid,
    hence this module requires the grid module.

:py:mod:`wavepacket.expression`
    Contains classes that define differential equation.
    These can be a Schroedinger equation or various Liouvillians.
    As these entities wrap an operator, this module requires the operator module.

:py:mod:`wavepacket.solver`
    Contains actual Solvers for the differential equations.
    Hence, this module requires knowledge of the expression module.


Two other modules stand apart from this hierarchy:

* :py:mod:`wavepacket.typing` contains definitions for type hinting.
* :py:mod:`wavepacket.testing` contains some test helpers,
  for example :py:func:`wavepacket.testing.assert_close` to compare two states with each other.
* The top-level package offers the exception classes and general utility functions,
  such as py:func:`wavepacket.trace` or :py:func:`wavepacket.expectation_value`.

With these concepts in mind, most functionality should be readily explorable.
