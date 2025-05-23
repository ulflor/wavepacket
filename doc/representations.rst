Representation of states
========================

A more formula-heavy introduction of the DVR method can be found in the
`Wavepacket wiki <https://sourceforge.net/p/wavepacket/wiki/Numerics.DVR>`_.
Here we only introduce the most important elements. The takeaway message is:

* States in Wavepacket are represented by a coefficient vector in a
  special basis. For most purposes, you should transform the state into
  a more suited basis. For example, you should transform to the DVR if
  you want to plot a state.
* A grid is built as the direct product of one or more one-dimensional grids
  called degrees of freedom.

Let us start with a single degree of freedom only, and briefly look
at the different representations for a wave function.

FBR representation
------------------

The Finite Basis Representation (FBR) is what we always do in quantum mechanics:
Take a set of orthonormal basis functions, and expand your wave function,

.. math::

   \psi(x) = \sum_k c_k f_k(x),

where in practice, the sum is always truncated.
The coefficient vector provides a unique representation of the wave function.
Each degree of freedom (DOF) class in Wavepacket is modelled around
specific basis functions, for example
:py:class:`wavepacket.grid.PlaneWaveDof` uses plane waves as underlying basis.
Physical quantities can be calculated from the the coefficient vector; for example the trace
is given by :math:`\|\psi\|^2 = \sum_k |c_k|^2`.

We only use the FBR for special applications, such as operators
that are diagonal in the underlying basis. In general, we much prefer the DVR.

DVR representation
------------------

For the Discrete Variable Representation, we again expand in an orthonormal basis

.. math::
   :label: representation_dvr

   \psi(x) = \sum_i \psi_i g_i(x),

but choose the basis functions such that the coefficients are the values of the wave function at some
fixed grid points, :math:`\psi_i = \psi(x_i)`. It can be shown 
that basis expansions with N FBR coefficients can usually be converted
into a DVR with some choice of N grid points without loss of information.

The DVR is used for numerics with the DVR approximation: You apply
a space-local operator like a potential by multiplying the operator
values with the wave function values at the grid points. But most importantly,
if your expansion coefficients are the wave function values, you can directly
plot them!

.. note::

    The DVR is not inherently simple. For example, the density
    is given in the DVR as
    :math:`|\psi(x)|^2 = \sum_{i,j=1}^N \psi_i \psi_j^\ast g_i(x;N) g_j^\ast(x;N)`,
    where the basis functions also depend on the number of grid points.
    In practice, we side-step these problems by plotting
    the density only at the grid points, :math:`(x_i, |\psi(x_i)|^2)`.
    
    This gives artefacts if the grid has few grid points. However, in these cases,
    it is usually more insightful to study the system in a well-chosen FBR instead.

For actual computations, the DVR requires additional weights
for the grid points. For example, the trace of the wave function
is calculated as :math:`\|\psi\|^2 = \sum_i w_i |\psi(x_i)|^2`. Carrying around
the weights is somewhat annoying, and the additional
multiplication makes the calculations slower than they need to be.

For that reason, we do not quite use the DVR as native representation in Wavepacket.

Weighted DVR representation
---------------------------

A more numerically convenient representation can be obtained if we use a
modified DVR expansion :eq:`representation_dvr` with
:math:`\tilde g_i = g_i / \sqrt{w_i}` and
:math:`\tilde \psi_i = \psi_i \sqrt{w_i}`.
This allows us to ditch the weights.

For example, the trace is now simply given as
:math:`\|\psi\|^2 = \sum_i |\tilde \psi_i|^2`.

This weighted DVR is the default representation in Wavepacket.
Unless explicitly noted otherwise, all wave functions are expected and returned
in this representation. If you need the wave function, for example for plotting
the DVR density, you should explicitly transform it with available helper
functions, for example :py:func:`wavepacket.grid.dvr_density`.

Multidimensional grids
----------------------

A multidimensional :py:class:`wavepacket.grid.Grid` is constructed as the direct
product of one-dimensional degrees of freedom using the weighted DVR. Hence,
for example a two-dimensional wave function is given as

.. math::
   :label: representation_2dgrid

   \psi(x,y) = \sum_{i=1}^N \sum_{j=1}^M \tilde \psi_{ij}
   \tilde g_i(x) \tilde h_j(y)

where the functions are the weighted DVR basis of the two one-dimensional grids,
and where the coefficients are now given as a matrix of weighted DVR values.

Density operators
-----------------

A similar extension holds for density operators. For the example of a pure
density operator on the two-dimensional grid of :eq:`representation_2dgrid`,
the result is:

.. math::

   \rho(x_1, y_1, x_2, y_2) = \sum_{i,k=1}^N \sum_{j,l=1}^M
   (\tilde \psi_{ij} \tilde \psi_{kl}^\ast)
   \tilde g_i(x_1) \tilde h_j(y_1) \tilde g_k^\ast(x_2) \tilde h_l^\ast(y_2)

so that the resulting density operator is a four-dimensional tensor
:math:`\rho_{ijkl} = \psi_{ij}\psi_{kl}^\ast`, also for non-pure states.
You still benefit from the weighted DVR, for example the
trace is given by :math:`Tr[\hat \rho] = \sum_{i=1}^N \sum_{k=1}^M \rho_{ikik}`.
