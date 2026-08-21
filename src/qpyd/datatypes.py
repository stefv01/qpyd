import numpy as np
from typing import TypedDict


class StateProperties(TypedDict):
    """
    Dictionary structure containing the physical properties of a computed eigenstate.

    Attributes
    ----------
    eigenstate : str
        A rigidly ordered string representation of the many-body state.
    single_particle_kets : list of tuple of (float or complex, str, np.ndarray)
        A list of tuples representing the constituent many-body states expressed 
        in the single-particle basis. Each tuple is formatted as 
        `(amplitude, ket_string, ket_array)`.
    energy : float
        The calculated eigenenergy of the state.
    S : float
        The total spin quantum number.
    Sz : float
        The total spin projection quantum number.
    """
    eigenstate: str
    single_particle_kets: list[tuple[float | complex, str, np.ndarray]]
    energy: float
    S: float
    Sz: float