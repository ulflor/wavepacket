Wavepacket documentation
========================

Wavepacket allows you to easily set up and simulate smaller quantum
systems. It is particularly well suited for small molecule molecular dynamics
or for teaching. See :doc:`tutorials/schroedinger_cat` for an introduction into
the code.

The current Python package is the offspring of a longer history of
development, there is also a maintained Matlab implementation and a
superseded C++ implementation.

Features
--------

- Uses the DVR approximation, which allows you to define
  potentials directly as functions in real space.
  See :doc:`representations` for more information.
- Directly solves the differential equations numerically.
  This is slower than clever methods like MCTDH, but easier to use.
- Most functions accept wave functions as well as density operators,
  allowing you to move between closed and open quantum systems with
  few code changes.
- You can easily define complex setups:

  Want to use an ensemble of random thermal wave functions?
  Can be easily done (see :doc:`/advanced/thermal_states`).

  Summing over all magnetic quantum numbers m for a given initial
  angular momentum? No problem (see :doc:`/advanced/pendular_states`).


Bugs / Requests
---------------

If you find a bug, have a feature request, or even need support, feel
free to use the `issue tracker <https://github.com/ulflor/wavepacket/features>`_.


.. toctree::
    :caption: General
    :hidden:

    architecture
    representations
    license

.. toctree::
    :caption: Introductory tutorials
    :hidden:

    tutorials/schroedinger_cat
    tutorials/eigenstates
    tutorials/plotting
    tutorials/chebychev_solvers
    tutorials/relaxation

.. toctree::
    :caption: Advanced usage and theory
    :hidden:

    advanced/pendular_states
    advanced/thermal_states
    advanced/polynomial_solvers

.. toctree::
    :caption: Other links

    Project homepage <https://github.com/ulflor/wavepacket>
    Wavepacket wiki <https://sourceforge.net/p/wavepacket/wiki>

.. toctree::
    :caption: API
    :maxdepth: 4
    :hidden:

    autoapi/index
