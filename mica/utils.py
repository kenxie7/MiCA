"""
MiCA Utilities
==============

State-based node coloring and legend utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional


# =============================================================================
# STATE COLORS
# =============================================================================

STATE_COLORS = {
    'pdam': '#73b2b2',
    'img': '#7f81c7',
    'hm': '#4a4a9d',
    'irm': '#b38dd0',
    'idam': '#904849',
    'exdam': '#d3676d',
}


STATE_ORDER = ['pdam', 'exdam', 'idam', 'irm', 'img', 'hm']


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_cluster_state_zscore(cluster_profiles):
    """
    Compute z-score normalized cluster state profiles.
    Z-score is computed ACROSS CLUSTERS for each state.
    """
    mean_per_state = cluster_profiles.mean(axis=0)
    std_per_state = cluster_profiles.std(axis=0)
    std_per_state = std_per_state.replace(0, 1)
    
    return (cluster_profiles - mean_per_state) / std_per_state


def get_dominant_state_per_cluster(cluster_profiles, method='raw'):
    """
    For each cluster, find the dominant state and its score.
    
    Parameters
    ----------
    cluster_profiles : pd.DataFrame
        Shape (n_clusters, n_states).
    method : str
        'raw' - use max raw value (recommended, biologically intuitive)
        'zscore' - use max z-scored value
        'spearman' - use max Spearman correlation with pure state vectors
    
    Returns
    -------
    pd.DataFrame with columns: ['dominant_state', 'score', 'raw_value', 'z_score']
    """
    state_order = list(cluster_profiles.columns)
    z_scored = compute_cluster_state_zscore(cluster_profiles)
    
    results = []
    
    for cluster in cluster_profiles.index:
        raw_row = cluster_profiles.loc[cluster]
        z_row = z_scored.loc[cluster]
        
        if method == 'raw':
            dominant = raw_row.idxmax()
            raw_val = raw_row.max()
            z_val = z_row[dominant]
            score = raw_val
            
        elif method == 'zscore':
            dominant = z_row.idxmax()
            z_val = z_row.max()
            raw_val = raw_row[dominant]
            score = z_val
            
        elif method == 'spearman':
            correlations = {}
            for state in state_order:
                pure_vec = np.array([1.0 if s == state else 0.0 for s in state_order])
                corr, _ = spearmanr(raw_row.values, pure_vec)
                correlations[state] = corr
            dominant = max(correlations, key=correlations.get)
            score = correlations[dominant]
            raw_val = raw_row[dominant]
            z_val = z_row[dominant]
        
        results.append({
            'cluster': cluster,
            'dominant_state': dominant,
            'score': score,
            'raw_value': raw_val,
            'z_score': z_val
        })
    
    return pd.DataFrame(results).set_index('cluster')


def get_cluster_colors(cluster_profiles, method='raw', min_alpha=0.5,
                       state_colors=None, per_state_scaling=True):
    """
    Get RGBA colors for each cluster based on dominant state.
    
    Parameters
    ----------
    cluster_profiles : pd.DataFrame
    method : str
        'raw' - color by max raw value (recommended)
        'zscore' - color by max z-score
        'spearman' - color by max Spearman correlation
    min_alpha : float
        Minimum opacity (default 0.5)
    state_colors : dict or None
        Custom colors. Uses default if None.
    per_state_scaling : bool
        If True, scale alpha within each state (max=1.0 per state).
        If False, scale alpha globally.
    
    Returns
    -------
    dict : {cluster_name: {'rgba': tuple, 'state': str, 'score': float, ...}}
    """
    if state_colors is None:
        state_colors = STATE_COLORS
    
    # Get dominant state for each cluster
    dominant_df = get_dominant_state_per_cluster(cluster_profiles, method=method)
    z_scored = compute_cluster_state_zscore(cluster_profiles)
    
    if per_state_scaling:
        # Group clusters by their dominant state
        state_to_clusters = {}
        for cluster in dominant_df.index:
            state = dominant_df.loc[cluster, 'dominant_state']
            z_val = z_scored.loc[cluster, state]
            if state not in state_to_clusters:
                state_to_clusters[state] = []
            state_to_clusters[state].append((cluster, z_val))
        
        # Compute per-state scaled alphas
        cluster_colors = {}
        for state, clusters in state_to_clusters.items():
            z_vals = [z for c, z in clusters]
            max_z = max(z_vals)
            min_z = min(z_vals)
            
            for cluster, z_val in clusters:
                if max_z == min_z:
                    alpha = 1.0
                else:
                    alpha = min_alpha + (1.0 - min_alpha) * (z_val - min_z) / (max_z - min_z)
                
                alpha = np.clip(alpha, min_alpha, 1.0)
                rgb = mcolors.hex2color(state_colors[state])
                
                cluster_colors[cluster] = {
                    'rgba': (*rgb, alpha),
                    'hex': state_colors[state],
                    'state': state,
                    'z_score': z_val,
                    'raw_value': dominant_df.loc[cluster, 'raw_value'],
                    'score': dominant_df.loc[cluster, 'score'],
                    'alpha': alpha
                }
    else:
        # Global scaling (original behavior)
        cluster_colors = {}
        for cluster in dominant_df.index:
            row = dominant_df.loc[cluster]
            state = row['dominant_state']
            score = row['score']
            raw_val = row['raw_value']
            z_val = row['z_score']
            
            # Global alpha scaling
            if method == 'raw':
                normalized = min((raw_val - 0.1) / 1.0, 1.0)
                normalized = max(normalized, 0)
            elif method == 'zscore':
                normalized = min(score / 2.5, 1.0)
            else:
                normalized = 0.5
            
            alpha = min_alpha + (1 - min_alpha) * normalized
            rgb = mcolors.hex2color(state_colors[state])
            
            cluster_colors[cluster] = {
                'rgba': (*rgb, alpha),
                'hex': state_colors[state],
                'state': state,
                'score': score,
                'raw_value': raw_val,
                'z_score': z_val,
                'alpha': alpha
            }
    
    return cluster_colors


# =============================================================================
# VERTICAL STATE LEGEND
# =============================================================================

def create_state_legend_data(cluster_colors):
    """
    Organize cluster colors into state-grouped structure for legend.
    
    Parameters
    ----------
    cluster_colors : dict
        Output from get_cluster_colors()
    
    Returns
    -------
    dict : {state: [(cluster, z_score, alpha), ...]} sorted by alpha desc
    """
    state_to_clusters = {}
    
    for cluster, info in cluster_colors.items():
        state = info['state']
        if state not in state_to_clusters:
            state_to_clusters[state] = []
        state_to_clusters[state].append((cluster, info['z_score'], info['alpha']))
    
    # Sort each state's clusters by alpha (descending)
    for state in state_to_clusters:
        state_to_clusters[state].sort(key=lambda x: x[2], reverse=True)
    
    return state_to_clusters


def plot_state_legend_vertical(cluster_colors, 
                                ax: Optional[plt.Axes] = None,
                                state_order: List[str] = None,
                                state_colors: Dict[str, str] = None,
                                box_width: float = 1.0,
                                box_height: float = 1.0,
                                gap: float = 0.25,
                                show_cluster_names: bool = True,
                                name_fontsize: int = 7,
                                state_fontsize: int = 11,
                                figsize: Tuple[float, float] = (3, 7)) -> plt.Axes:
    """
    Create vertical stacked legend with clusters grouped by state.
    
    Parameters
    ----------
    cluster_colors : dict
        Output from get_cluster_colors()
    ax : plt.Axes, optional
    state_order : list
        Order of states top to bottom
    state_colors : dict
    box_width, box_height : float
    gap : float
        Gap between state groups
    show_cluster_names : bool
        Show cluster names in boxes
    figsize : tuple
    
    Returns
    -------
    plt.Axes
    """
    if state_order is None:
        state_order = STATE_ORDER
    if state_colors is None:
        state_colors = STATE_COLORS
    
    # Get organized data from cluster_colors
    state_to_clusters = create_state_legend_data(cluster_colors)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    y = 0
    state_label_positions = []
    
    for state in state_order:
        if state not in state_to_clusters:
            continue
        
        clusters = state_to_clusters[state]
        state_start_y = y
        
        for cluster, z_score, alpha in clusters:
            rgb = mcolors.hex2color(state_colors[state])
            rect = Rectangle((0, y), box_width, box_height,
                            facecolor=(*rgb, alpha), 
                            edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            
            if show_cluster_names:
                short_name = cluster.replace('Homeostatic', 'Hom').replace('_Act', '_A')
                text_color = 'white' if alpha > 0.6 else 'black'
                ax.text(box_width/2, y + box_height/2, short_name,
                       fontsize=name_fontsize, ha='center', va='center',
                       fontweight='bold', color=text_color)
            
            y += box_height
        
        # Store position for state label
        state_mid_y = (state_start_y + y) / 2
        state_label_positions.append((state, state_mid_y))
        
        y += gap
    
    # Add state labels on right
    for state, mid_y in state_label_positions:
        ax.text(box_width + 0.15, mid_y, state,
               fontsize=state_fontsize, ha='left', va='center',
               fontweight='bold', color=state_colors[state])
    
    ax.set_xlim(-0.1, box_width + 1.5)
    ax.set_ylim(-0.2, y + 0.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.invert_yaxis()  # Top to bottom
    
    return ax


def add_state_legend_to_figure(fig, cluster_colors, 
                                position: Tuple[float, float, float, float] = (0.95, 0.15, 0.12, 0.7),
                                **kwargs):
    """
    Add vertical state legend to an existing figure.
    
    Parameters
    ----------
    fig : plt.Figure
    cluster_colors : dict
        Output from get_cluster_colors()
    position : tuple
        (left, bottom, width, height) in figure coordinates
    **kwargs : passed to plot_state_legend_vertical
    
    Returns
    -------
    plt.Axes
    """
    ax_legend = fig.add_axes(position)
    plot_state_legend_vertical(cluster_colors, ax=ax_legend, **kwargs)
    return ax_legend
