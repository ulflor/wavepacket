Architecture overview
=====================

Common concepts
---------------

The focus of Wavepacket is on flexible setups of simple quantum systems.
To get there, the code ended up with a few common concepts.

* To abstract away the difference between wave functions and density operators,
  a class :py:class:`wavepacket.grid.State` has been introduced.
* Most utility functions and classes can operate on wave functions as well as
  density operators. In some cases, this is principally not possible. For
  example, an operator can only operate on a wave function, but can be applied
  from the left or right onto a density operator. This turns out to be no
  problem in practice, because you hardly ever use operators directly.

  Where the difference is confusing, functionality may be offered differently.
  For example, the norm of a wave function and the equivalent density operator
  are different (the latter is the square of the former). For this reason, you
  do not find a function `norm()`, but :py:func:`wavepacket.grid.trace`,
  because the trace has the same definition for both cases.
* The classes in Wavepacket all use the Value Pattern, and are immutable after
  initialization. This allows you to recycle objects without side effects.
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
    :py:class:`wavepacket.grid.State`, as well as supporting functions and classes
    that do not require information about operators, for example various transformations.

:py:mod:`wavepacket.builder`
    Contains utility functions to create an initial state.
    These functions require a grid on which to create the state.

:py:mod:`wavepacket.operator`
    Contains operator classes. Operators are always defined on a specific grid,
    hence this module requires the grid module. Also, this module contains some utility
    functions related to operators, such as :py:func:`wavepacket.operator.expectation_value`

:py:mod:`wavepacket.expression`
    This module contains classes that inserts an operator into a differential
    equation. These can be a Schroedinger equation or various Liouvillians.
    As these entities wrap an operator, this module requires the operator module.

:py:mod:`wavepacket.solver`
    Here, you can find The solver module contains actual Solvers for the differential equations.


Two other modules stand apart from this hierarchy:

* :py:mod:`wavepacket.typing` contains definitions for type hinting.
* :py:mod:`wavepacket.testing` contains some test helpers,
  for example :py:func:`wavepacket.testing.assert_close` to compare two states with each other.

With these concepts, most functionality should be readily available.