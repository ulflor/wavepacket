=========
ChangeLog
=========

0.3
---

- (#24) added ExpressionSum and OneSidedLiouvillian
  for more complex expressions


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
