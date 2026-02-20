Description
-----------

Wavepacket is a Python package to simulate small quantum systems.
Technically, it allows you to numerically solve Schrödinger and
Liouville-von-Neumann equations for distinguishable particles.

The full documentation can be found under https://wavepacket.readthedocs.io.
See also the advanced use cases for examples of what can be done.

Wavepacket focuses on a particular niche:

- Direct solution of Schrödinger / Liouville von Neumann equations.
  No electronic structure, no MCTDH, no semiclassics, at least for now.
- Grid-based representation of operators (DVR approximation).
- Accessibility is an overriding goal. Wavepacket should be directly usable in teaching.
  We try to provide good tutorials / docs, but also goodies like simple plotting.
- Exotic use cases should be as frictionless as possible.
  For example, switching between wave functions and density operators is supported well.
  Or random thermal wave functions are not much more complex to use than Gaussian states.
- Performance is a subordinate goal.
  For specific systems, you can easily improve the performance by integer factors.


Links
-----

Documentation:
    https://wavepacket.readthedocs.io
Source:
    https://github.com/ulflor/wavepacket
Issue tracker:
    https://github.com/ulflor/wavepacket/issues


Support
-------

If you lack a feature, create an issue in the issue tracker.
Depending on the complexity of the feature, this will lead to an immediate,
rapid, or prioritized implementation.

For support requests, also use the issue tracker for the time being.


Contribution
------------

I currently lack a formal procedure for new contributors, but you are
very welcome to contribute to the project. If you do not know what to do,
contact one of the developers; there is enough work for
multiple developers, also for non-coding skills (there is never enough
documentation).


History
-------

The original long-time version of Wavepacket was written in Matlab and is still
actively maintained under https://sourceforge.net/p/wavepacket/matlab. It is stable,
battle-tested and works.

However, Matlab is expensive, and not all users had easy
access to it. Also, the project's architecture did not support some
advanced use cases well. Finally,
C++11 had just come out and looked cool, so I started a
reimplementation in C++ around 2013, adding Python bindings as an
afterthought. The C++ project will be superseded by this Python package, but
can be found under https://sourceforge.net/p/wavepacket/cpp.

This worked really well, except that deploying C++ code is difficult.
In particular, there was no cheap route towards building a "good"
Python package, especially under Windows, which restricted accessibility.

The solution is this Python-only version based on numpy.
The Python version is slower by a factor of two to three compared
to C++-backed code. This is, however, often cancelled by a
parallelization of the tensor operations.
