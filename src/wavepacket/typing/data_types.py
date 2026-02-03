from typing import Callable, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

RealData: TypeAlias = npt.NDArray[np.float64]
"""
Type for real-valued input or output data.
"""

ComplexData: TypeAlias = npt.NDArray[np.float64] | npt.NDArray[np.complex128]
"""
Type for complex-valued input or output data.

Note that all client places should be able to consume real-valued data as well.
"""

# TODO: Replace by modern generic syntax once we drop support for Python 3.11
AnyData = TypeVar("AnyData", npt.NDArray[np.float64], npt.NDArray[np.complex128])
"""
Generic helper for functions that return real/complex output on real/complex input, respectively.
"""

Generator: TypeAlias = Callable[[RealData], ComplexData]
"""
A callable that transforms an real-valued input into complex-valued output.

Typical applications are transformations from grids or a time series, for
example for creating initial wave functions or potentials.
"""

RealGenerator: TypeAlias = Callable[[RealData], RealData]
"""
Similar to `Generator`, but outputs real-value data.

This is meant for cases where we deliberately want to constrain
the values.
"""
