"""
ALBERT Kerr Metric Module
Implements the Kerr metric for rotating black holes
"""

import sympy as sp
from typing import List, Dict, Any
from albert.core.tensors import TensorField


def kerr_metric(coords: List[str] = ['t', 'r', 'theta', 'phi'], 
                M: float = 1, 
                a: float = 0.5) -> TensorField:
    """
    Create the Kerr metric tensor for a rotating black hole
    
    Args:
        coords: Coordinate system names
        M: Mass of the black hole
        a: Angular momentum parameter (a = J/M where J is angular momentum)
    
    Returns:
        TensorField representing the Kerr metric
    """
    # Create coordinate symbols
    t, r, th, ph = sp.symbols(coords)
    
    # Boyer-Lindquist coordinates
    Sigma = r**2 + a**2 * sp.cos(th)**2
    Delta = r**2 - 2 * M * r + a**2
    
    # Create metric tensor (0 contravariant, 2 covariant = metric tensor)
    g = TensorField("g_Kerr", (0, 2), coords)
    
    # Helper function to get coordinate index
    def idx(s): 
        return coords.index(s)
    
    # Set non-zero components of the Kerr metric
    # g_tt
    g.set_component((idx('t'), idx('t')), -(1 - 2*M*r/Sigma))
    
    # g_rr
    g.set_component((idx('r'), idx('r')), Sigma/Delta)
    
    # g_θθ
    g.set_component((idx('theta'), idx('theta')), Sigma)
    
    # g_φφ
    g.set_component((idx('phi'), idx('phi')), 
                   (r**2 + a**2 + (2*M*a**2*r*sp.sin(th)**2)/Sigma)*sp.sin(th)**2)
    
    # g_tφ = g_φt (off-diagonal terms for frame dragging)
    g.set_component((idx('t'), idx('phi')), -2*M*a*r*sp.sin(th)**2/Sigma)
    g.set_component((idx('phi'), idx('t')), -2*M*a*r*sp.sin(th)**2/Sigma)
    
    return g
