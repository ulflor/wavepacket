import pytest
import wavepacket as wp

def test_reject_negative_range():
    with pytest.raises(Exception):
        wp.grid.PlaneWaveDof(10, 5, 10)