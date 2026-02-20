=========
ChangeLog
=========

0.4
---


0.3
---

- (#32) added ChebychevSolvers
  - added a ChebychevSolver for real-time propagation
  - added RelaxationSolver for imaginary-time propagation
  - added function to normalize a state
  - Added various documentation on Chebychev solver use, relaxation,
    and polynomial solver theory

- (#24) added ExpressionSum and OneSidedLiouvillian
        for more complex expressions

- (#31) added diagonalize() function for operator eigenstates and -energies

- (#11) operator truncation and systematic handling of operator time-dependence

- (#12) Added some documentation on the construction of thermal states

- (#12) Changed the implementation of builder.random_wave_function();
  it now constructs random numbers exp(i * phi) because these are more useful.

- (#13) Tutorials and demos also can be downloaded as Jupyter notebooks
  and mostly render fine as notebooks.

- significantly improved type information, simplified attributes,
  implemented some annotation best practices


0.2
---

- (#18) added spherical harmonics expansion:
  degree of freedom, operator, generator

- (#19) added time-dependent functions and laser fields as operators,
  including helper functions for sin**2 and rectangular pulses with soft turn-on

- (#20) added functionality related to projections
  - projection operator for projecting onto a state or subspace
  - population() function to easily calculate populations of target states
  - Gram-Schmidt orthogonalization and normalization

- (#21) added more initial states
  random wave functions, zero densities and wave functions, unit density
- (#21) added function grid.fbr_density() to calculate density in FBR

- (#22) 1D plotting helpers
  - added plotting classes StackedPlot1D and SimplePlot1D
  - added a tutorial on plotting

- (#23) Added a notebook with the PendularStates demos converted from the older versions

- (#15) added CI build using tox for different Python versions
        Wavepacket now supports every Python >= 3.11


0.1
---

Initial release.

Scope is the simulation of a harmonic oscillator with an ordinary ODE solver,
plus a bit of introductory documentation and a demo.

That is, infrastructure, not so much content.
