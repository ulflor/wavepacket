Description
-----------

Wavepacket is a Python package to define and simulate small
quantum systems. Or, more technically, it allows you to numerically
solve Schrödinger and Liouville-von-Neumann equations for
distinguishable particles.

There are many different quantum systems and consequently approaches
to solve them. Here we focus on a particular niche:

- Wavepacket solves the differential equations directly. This simplifies
  the maths, but limits the system size to few degrees of freedom.
  If you want to deal with larger systems, look out for MCTDH.
- Wavepacket uses the DVR approximation heavily. This allows you to
  directly define your potentials as functions of real-space coordinates
  instead of setting up opaque operator matrices.
  The latter approach is simpler an more concise if you are only
  interested in harmonic oscillators or qubits, though.
- Wavepacket is a Python-only package relying chiefly on numpy.
  This is slower than natively implemented code, but you gain
  great tooling support, for example matplotlib, Jupyter notebooks or
  integrated documentation.
- Most of the code can handle both wave functions and density operators.
  This allows you to convert a closed system into an open
  system with minimal fuss.

For example use cases, we have been using various precursors of this
package for simulating small molecular systems and for teaching.
Besides examples shipped with this package, see
https://sourceforge.net/p/wavepacket/wiki/Demos.Main for more applications.

The project is currently a first iteration to flesh out everything. Once
0.1 is released, I plan to quickly translate the existing C++ code from
a precursor project and reach a stable state. More can be found on the
project homepage https://github.com/ulflor/wavepacket


Support
-------

If you lack a feature that you would like to have, open an issue at
`our issue tracker <https://github.com/ulflor/wavepacket/issues>`_.
Depending on the complexity of the feature, this will lead to an immediate,
rapid, or prioritized implementation.


Contribution
------------

I currently lack a formal procedure for new contributors, but you are
very welcome to contribute to the project. If you do not know what to
do, feel free to email me at `ulf@wppy.org`; there is enough work for
more than one, just not fleshed out yet.


History
-------

The original version of Wavepacket was written in Matlab and is still
maintained under https://sourceforge.net/p/wavepacket/matlab. It is stable,
battle-tested and works.

However, Matlab is pretty expensive and not all interested user had
access to it. Also, the project's architecture did not support some
advanced use cases without digging deep into the code. Finally,
C++11 had just come out and looked cool, so I started a
reimplementation in C++ around 2013, adding Python bindings as an
afterthought. This project is getting out of maintenance, but can be
found under https://sourceforge.net/p/wavepacket/cpp.

This worked really well. However, deploying C++ code is difficult.
In particular, there was no cheap route towards building a "good"
Python package. Also, the underlying tensor library was slowly
getting less and less commits over the years, so I am currently
moving to a Python-only package, which makes most coding tasks
way simpler and faster. The main drawback is performance;
The Python version is slower by a factor of two to three compared
to C++-backed code. This is, however, often cancelled by a
parallelization of the tensor operations.
