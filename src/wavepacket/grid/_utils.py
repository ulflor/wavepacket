from typing import overload

import numpy as np

import wavepacket.typing as wpt


@overload
def broadcast(data: wpt.ComplexData, ndim: int, index: int) -> wpt.ComplexData: ...
@overload
def broadcast(data: wpt.RealData, ndim: int, index: int) -> wpt.RealData: ...
def broadcast(data, ndim, index):
    """
    Reshape an array by appending dummy indices.

    We basically want to calculate the product A_ijklm * b_k, where both the rank
    of the tensor A and the position of the running index k are only known at
    runtime. This seems to be outside numpy's standard use cases.

    As a workaround, we can reshape the array b to a multidimensional tensor
    b_k -> B_k00 and calculate A * B with numpy's broadcasting rules. That is what
    this function does.
    """
    shape = np.ones(ndim, dtype=int)
    shape[index] = len(data)

    return np.reshape(data, shape)


