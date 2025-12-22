"""
MiCA: Microglial CRISPR Analysis
================================

A framework for projecting CRISPR knockout effects onto 
cell state transition networks.

Usage
-----
>>> from mica import EdgeProjector, ProjectionPlotter
>>> 
>>> proj = EdgeProjector(sc_data, ko_data)
>>> proj.build_graph()  # Default: Gabriel graph in state space
>>> results = proj.project_all()
>>> 
>>> plotter = ProjectionPlotter(proj, results)
>>> plotter.plot_projection(node_color_method='raw')
>>> plotter.plot_state_profiles(auto_scale=True)
"""

__version__ = '0.1.0'
__author__ = 'Ken Xie'

from .projector import EdgeProjector, ProjectionResult, load_data
from .plotting import ProjectionPlotter
from .utils import (
    STATE_COLORS,
    STATE_ORDER,
    get_cluster_colors,
    get_dominant_state_per_cluster,
    plot_state_legend_vertical,
    compute_cluster_state_zscore
)

__all__ = [
    # Core classes
    'EdgeProjector',
    'ProjectionResult', 
    'ProjectionPlotter',
    # IO
    'load_data',
    # Constants
    'STATE_COLORS',
    'STATE_ORDER',
    # Utilities
    'get_cluster_colors',
    'get_dominant_state_per_cluster',
    'plot_state_legend_vertical',
    'compute_cluster_state_zscore',
]
