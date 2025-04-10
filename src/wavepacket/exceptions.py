class BadFunctionCall(Exception):
    """
    Signals that a function was called incorrectly.

    A typical but rare use-case would be a function that was
    called with incorrect parameters.
    """
    pass


class BadGridError(Exception):
    """
    An invalid grid was supplied.

    Most often, you attempt an operation between objects that must be defined on the
    same grid. For example, the addition of two operators defined on different grids
    is not a useful operation. The grid may also miss required properties, for example
    an operation may expect a specific degree of freedom type along some index.
    """
    pass


class BadStateError(Exception):
    """
    An invalid state was supplied.

    Either the state is completely invalid (neither wave function nor density operator),
    or you supplied the wrong type of state to a function, for example passing a
    density operator where a wave function was required.
    """
    pass


class ExecutionError(Exception):
    """
    An unrecoverable problem was encountered in foreign code.

    The main example is the :py:class:`wavepacket.solver.odesolver` getting
    an error back while integrating.
    """
    pass


class InvalidValueError(Exception):
    """
    A function argument was incorrect, for example out of bounds.
    """
    pass
