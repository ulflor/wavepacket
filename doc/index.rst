Wavepacket documentation
========================

Wavepacket allows you to easily set up and simulate smaller quantum
systems. It is particularly well suited for small molecule molecular dynamics
or for teaching.

The current Python package is the offspring of a longer history of
development, there is also a maintained Matlab implementation and a
superseded C++ implementation.


.. toctree::
    :caption: Introduction

    getting-started
    architecture

.. toctree::
   :caption: Demos for typical usages

   demos/schroedinger_cat

.. toctree::
    :caption: Useful links

    Project homepage <https://github.com/ulflor/wavepacket>
    Wavepacket wiki <https://sourceforge.net/p/wavepacket/wiki>

.. toctree::
   :caption: API
   :maxdepth: 4

   autoapi/index


Features
--------

- Uses the DVR approximation, which allows you to define
  potentials directly as functions in real space.
- Directly solves the differential equations numerically.
  This scales poorly to large systems, but is easy to use.
- Most of the code handles wave functions and density operators
  on the same footing, allowing you to move easily between closed
  and open quantum systems.
- You can easily define complex setups: Want to use an ensemble of
  random thermal wave functions? Summing over all quantum numbers m
  for a given initial angular momentum? No problem.


Bugs / Requests
---------------

If you find a bug, have a feature request, or even need support, feel
free to use the `issue tracker <https://github.com/ulflor/wavepacket/features>`_.
