import importlib.metadata

try:
    __version__ = importlib.metadata.version("qpyd")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

# Parameter Classes
from .parameters import (
    BaseParameters,
    DirectParameters,
    DerivedParameters
)

# Simulator Engines & Factory
from .simulator import (
    SimulatorBase,
    DirectSimulator,
    DerivedSimulator,
    Simulator
)

# Plotting Utilities
from .plotter import Plotter

__all__ = [
    "__version__",
    "BaseParameters",
    "DirectParameters",
    "DerivedParameters",
    "SimulatorBase",
    "DirectSimulator",
    "DerivedSimulator",
    "Simulator",
    "Plotter"
]