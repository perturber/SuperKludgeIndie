"""
The ode module of the trajectory package implements the stock trajectory models supported in FEW and provides
the tools necessary for users to implement their own trajectory models.
"""

from .flux import SchwarzEccFlux, KerrEccEqFlux, SuperKludgeFlux
from .pn5 import PN5

_STOCK_TRAJECTORY_OPTIONS = {
    "SchwarzEccFlux": SchwarzEccFlux,
    "KerrEccEqFlux": KerrEccEqFlux,
    "SuperKludgeFlux": SuperKludgeFlux,
    "PN5": PN5,
}
