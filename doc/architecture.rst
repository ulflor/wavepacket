Architecture overview
=====================

Common concepts
---------------

The focus of Wavepacket is on flexible setups of simple quantum systems.
To get there, the code ended up with a few common concepts.

* To abstract away the difference between wave functions and density operators,
  a class :py:class:`wavepacket.grid.State` has been introduced.
* Most utility functions and classes can operate on wave functions as well as
  density operators, wherever this is possible and sensible.

  Where the difference can lead to confusion and errors, functionality may not be offered.
  For example, the wave function norm is the square root of the
  of the corresponding density operator (trace) norm. This has been extremely confusing
  when switching between the two objects, so Wavepacket does not offer a `norm()`
  function, it offers :py:func:`wavepacket.grid.trace` instead,
* Classes in Wavepacket normally use the Value Pattern, and are immutable after
  creation. This allows you to recycle objects without side effects; for example,
  when you have set up a Hamiltonian for the field-free case, you can trivially use it
  also as a component for a system interacting with a laser field.
  Also, all classes and packages follow a strict hierarchy as described in the
  subsequent section.


Packages
--------

Wavepacket is split into several subpackages with hierarchical dependencies.
From lowest to highest layer, these are:

:py:mod:`wavepacket.grid`
    Contains all classes that describe the representation of a state (the grid),
    and all functionality that operates only on this representation. These include
    degrees of freedom, :py:class:`wavepacket.grid.Grid` itself, the class
    :py:class:`wavepacket.grid.State`, as well as supporting functions and classes.

:py:mod:`wavepacket.builder`
    Contains utility functions to create an initial wave function or density operator.
    These functions require a grid on which to create the state.

:py:mod:`wavepacket.operator`
    Contains operator classes. Operators are always defined on a specific grid,
    hence this module requires the grid module. The module also contains some utility
    functions related to operators, such as :py:func:`wavepacket.operator.expectation_value`

:py:mod:`wavepacket.expression`
    This module contains classes that wrap an operator for a differential
    equation. These can be a Schroedinger equation or various Liouvillians.
    As these entities wrap an operator, this module requires the operator module.

:py:mod:`wavepacket.solver`
    Here, you can find actual Solvers for the differential equations.
    Hence, this module requires knowledge of the expression module.


Two other modules stand apart from this hierarchy:

* :py:mod:`wavepacket.typing` contains definitions for type hinting.
* :py:mod:`wavepacket.testing` contains some test helpers,
  for example :py:func:`wavepacket.testing.assert_close` to compare two states with each other.
* A few grid-independent utilities are found in the top-level namespace, such as
  callables, exceptions, logging functions etc.

With these concepts in mind, most functionality should be readily findable.
