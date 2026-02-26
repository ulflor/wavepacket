import wavepacket as wp
import wavepacket.typing as wpt


def get_potential_values(potential: wp.operator.OperatorBase, t: float) -> wpt.RealData:
    unit_wave_function = wp.builder.unit_wave_function(potential.grid)
    dummy_state = potential.apply(unit_wave_function, t)
    return dummy_state.data.real

