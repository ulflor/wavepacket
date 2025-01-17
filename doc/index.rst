Wavepacket documentation
========================

Wavepacket allows you to easily set up and simulate smaller quantum
systems. It is particularly well suited for small molecule molecular dynamics
or for teaching.

The current Python package is the offspring of a longer history of
development, there is also a maintained Matlab implementation and a
superseded C++ implementation.


Links
-----

- The project homepage <https://github.com/ulflor/wavepacket>
- The Wavepacket wiki <https://sourceforge.net/p/wavepacket/wiki>
    contains many more examples and numerical background articles.


Features
--------

- Uses the DVR approximation, which allows you to define
  potentials directly as functions in real space.
- Directly solves the differential equations numerically.
  This scales poorly to large systems, but is easy to use.
- Most of the code can handle wave functions and density operators,
  allowing you to move easily between closed and open quantum systems.
- You can easily define complex setups: Want to use an ensemble of
  random thermal wave functions? No problem.


Bugs / Requests
---------------

If you find a bug, have a feature request, or even need support, feel
free to use the `issue tracker <https://github.com/ulflor/wavepacket/features>`_.
