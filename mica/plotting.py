"""
MiCA Plotting Module
====================

Visualization functions for KO edge projections.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from typing import Optional, Dict, List, Tuple
from adjustText import adjust_text
import warnings
warnings.filterwarnings('ignore')

from .utils import (
    plot_state_legend_vertical,
    get_cluster_colors,
    STATE_COLORS,
    STATE_ORDER
)


# =============================================================================
# FONT SETTINGS
# =============================================================================

FONT_TITLE = 10
FONT_AXIS_LABEL = 8
FONT_TICK = 7


# =============================================================================
# COLOR PALETTES
# =============================================================================

def _make_sequential(c1, c2, c3, c4):
    """4-stop smooth gradient."""
    return mcolors.LinearSegmentedColormap.from_list('seq', [c1, c2, c3, c4])

_subtle = {
    'DeepTeal': ['#1a3a3a', '#4a8a8a', '#b8e0e0'],
    'DuskPlum': ['#2d2345', '#7a6a9a', '#d4c6e6'],
    'ForestMoss': ['#1e3d2f', '#5a9a6a', '#c2ddc9'],
    'SlateStone': ['#2a3444', '#7a8a94', '#d4dde4'],
    'WarmEmber': ['#3d2828', '#9a6a5a', '#e8d0c4'],
}

CMAP_DEFAULT = mcolors.LinearSegmentedColormap.from_list('custom', _subtle['DeepTeal']).reversed()


# =============================================================================
# DEFAULT COLORS
# =============================================================================

#DEFAULT_CLUSTER_COLORS = {
#    'Homeostatic': '#2E86AB',
#    'Homeostatic_Act': '#A23B72', 
#    'Homeostatic_Act2': '#9B59B6',
#    'Hom_Inf': '#F18F01',
#    'DAMs_Inf': '#C73E1D',
#    'DAMs1': '#E74C3C', 
#    'DAMs2': '#8E44AD',
#    'DAMs': '#1B1B3A'
#}

# Default color schemes
DEFAULT_CLUSTER_COLORS = {
    "Homeostatic": "#3a7b80",
    "Homeostatic_Act": "#5a9ba0",
    "Homeostatic_Act2": "#a8c9a5",
    "Hom_Inf": "#8e9b6c",
    "DAMs": "#8e6898",
    "DAMs1": "#d39cc7",
    "DAMs2": "#6c559e",
    "DAMs_Inf": "#d3676d",
    "DAMs_Inf_Monocytic": "#f3c17b",
    "DAMs_MHC_Inf": "#904849",
}


DEFAULT_DIRECTION_COLORS = {
    'toward_disease': '#C73E1D',
    'toward_homeostatic': '#2E86AB'
}


# =============================================================================
# PROJECTION PLOTTER CLASS
# =============================================================================

class ProjectionPlotter:
    """
    Plotting class for edge projection results.
    
    Parameters
    ----------
    projector : EdgeProjector
        EdgeProjector instance with built graph
    results : pd.DataFrame
        Projection results from projector.project_all()
    cluster_colors : dict, optional
        Custom colors for clusters
    direction_colors : dict, optional
        Custom colors for directions
        
    Examples
    --------
    >>> plotter = ProjectionPlotter(proj, results)
    >>> plotter.plot_projection(node_color_method='raw')
    >>> plotter.plot_state_profiles(auto_scale=True)
    """
    
    def __init__(self, projector, results: pd.DataFrame,
                 cluster_colors: Dict[str, str] = None,
                 direction_colors: Dict[str, str] = None):
        self.proj = projector
        self.results = results
        self.sc = projector.sc
        self.centroids = projector.centroids
        self.graph = projector.graph
        self.clusters = projector.graph['clusters']
        
        self.cluster_colors = cluster_colors or DEFAULT_CLUSTER_COLORS
        self.direction_colors = direction_colors or DEFAULT_DIRECTION_COLORS
    
    # =========================================================================
    # MAIN PROJECTION PLOT
    # =========================================================================
    
    def plot_projection(self, 
                        ax: Optional[plt.Axes] = None,
                        show_cells: bool = True,
                        show_edges: bool = True,
                        show_centroids: bool = True,
                        show_kos: bool = True,
                        show_labels: bool = True,
                        label_top_n: int = 10,
                        color_by: str = 'direction',
                        size_by: str = 'confidence',
                        ko_alpha: float = 0.6,
                        cell_alpha: float = 0.1,
                        figsize: Tuple[float, float] = (12, 10),
                        node_color_method: str = None,
                        node_min_alpha: float = 0.3,
                        show_state_legend: bool = True,
                        cmap_choice = None,
                        label_list: list = None,
                        title: str = None) -> plt.Axes:
        """
        Main projection plot showing KOs on UMAP with graph overlay.
        
        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to plot on. Creates new figure if None.
        show_cells : bool
            Show background cells
        show_edges : bool
            Show graph edges
        show_centroids : bool
            Show cluster centroids
        show_kos : bool
            Show projected KOs
        show_labels : bool
            Show KO labels
        label_top_n : int
            Number of top confidence KOs to label
        color_by : str
            'direction' or 'confidence' or 'edge'
        size_by : str
            'confidence' or 'fixed' or column name
        ko_alpha : float
            KO point transparency
        cell_alpha : float
            Background cell transparency
        figsize : tuple
            Figure size if creating new
        node_color_method : str
            'raw', 'zscore', or 'spearman' for state-based coloring
        node_min_alpha : float
            Minimum alpha for node coloring
        show_state_legend : bool
            Show vertical state legend
        cmap_choice : colormap
            Colormap for confidence coloring
        label_list : list
            Additional KOs to label
        title : str
            Plot title
        """
        if cmap_choice is None:
            cmap_choice = CMAP_DEFAULT
            
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Background cells
        if show_cells:
            for cluster in self.clusters:
                mask = self.sc['labels_merged'] == cluster
                color = self.cluster_colors.get(cluster, 'gray')
                ax.scatter(self.sc.loc[mask, 'x'], self.sc.loc[mask, 'y'],
                          c=color, s=3, alpha=cell_alpha, rasterized=True)
        
        # Graph edges
        if show_edges:
            for c1, c2 in self.graph['edges']:
                x1, y1 = self.centroids.loc[c1, ['x', 'y']]
                x2, y2 = self.centroids.loc[c2, ['x', 'y']]
                ax.plot([x1, x2], [y1, y2], 'k-', lw=1, alpha=0.5, zorder=2)
        
        # Centroids
        if show_centroids:
            if node_color_method in ['raw', 'zscore', 'spearman']:
                cluster_colors = get_cluster_colors(
                    self.proj.profile_zscore,
                    method=node_color_method,
                    min_alpha=node_min_alpha,
                    per_state_scaling=True
                )

                for cluster in self.proj.clusters:
                    x = self.proj.centroids_umap.loc[cluster, 'x']
                    y = self.proj.centroids_umap.loc[cluster, 'y']
                    info = cluster_colors[cluster]
                    ax.scatter(x, y, c=[info['rgba']], s=300, zorder=5,
                              edgecolors='black', linewidths=1)
                
                # Add vertical state legend as inset
                if show_state_legend:
                    fig = ax.get_figure()
                    bbox = ax.get_position()
                    legend_ax = fig.add_axes([bbox.x1 + 0.02, bbox.y0 + 0.15, 0.08, 0.4])
                    plot_state_legend_vertical(
                        cluster_colors, 
                        ax=legend_ax,
                        show_cluster_names=True,
                        name_fontsize=4,
                        state_fontsize=9
                    )
            else:
                for cluster in self.clusters:
                    x, y = self.centroids.loc[cluster, ['x', 'y']]
                    color = self.cluster_colors.get(cluster, 'gray')
                    ax.scatter(x, y, c=color, s=300, zorder=5,
                              edgecolors='black', linewidths=1)
                    ax.annotate(cluster.replace('_', '\n'), (x, y), fontsize=FONT_TICK,
                               fontweight='bold', ha='center', va='bottom',
                               xytext=(0, 18), textcoords='offset points')

        # KO points
        if show_kos:
            # Determine colors
            if color_by == 'direction':
                colors = [self.direction_colors.get(d, 'gray') 
                         for d in self.results['direction']]
            elif color_by == 'confidence':
                colors = self.results['confidence']
            elif color_by == 'edge':
                unique_edges = self.results['edge'].unique()
                edge_cmap = plt.cm.get_cmap('tab20', len(unique_edges))
                edge_color_map = {e: edge_cmap(i) for i, e in enumerate(unique_edges)}
                colors = [edge_color_map[e] for e in self.results['edge']]
            else:
                colors = 'steelblue'
            
            # Determine sizes
            if size_by == 'confidence':
                sizes = 15 + self.results['confidence'] * 100
            elif size_by == 'fixed':
                sizes = 50
            elif size_by in self.results.columns:
                sizes = 30 + (self.results[size_by] - self.results[size_by].min()) / \
                        (self.results[size_by].max() - self.results[size_by].min() + 1e-8) * 100
            else:
                sizes = 50
            
            scatter = ax.scatter(self.results['umap_x'], self.results['umap_y'],
                                c=colors, s=sizes, alpha=ko_alpha, zorder=3,
                                edgecolors='white', linewidths=0.5,
                                cmap=cmap_choice if color_by == 'confidence' else None)
            
            if color_by == 'confidence':
                plt.colorbar(scatter, ax=ax, label='Confidence', shrink=0.6)
                scatter.set_clim(vmin=0, vmax=1) 

        # Labels
        if show_labels:
            texts = []
            labels_to_show = pd.DataFrame()

            if label_top_n > 0:
                labels_to_show = self.results.nlargest(label_top_n, 'confidence')

            if label_list is not None and len(label_list) > 0:
                custom_labels = self.results[self.results.ko.isin(label_list)]
                labels_to_show = pd.concat([labels_to_show, custom_labels]).drop_duplicates(subset='ko')

            for _, r in labels_to_show.iterrows():
                txt = ax.text(r['umap_x'], r['umap_y'], r['ko'], fontsize=FONT_TICK,
                              bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9))
                texts.append(txt)

            if texts:
                adjust_text(texts, 
                            x=self.results['umap_x'].values,
                            y=self.results['umap_y'].values,
                            ax=ax,
                            arrowprops=dict(arrowstyle='-', color='black', lw=0.5),
                            expand_text=(1.2, 1.2),
                            force_text=(0.5, 0.5),
                            force_points=(0.3, 0.3))
        
        # Legend for direction
        if color_by == 'direction':
            handles = [
                Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=self.direction_colors['toward_disease'],
                      markersize=10, label='Toward Disease'),
                Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=self.direction_colors['toward_homeostatic'],
                      markersize=10, label='Toward Homeostatic'),
            ]
            ax.legend(handles=handles, loc='lower left', title='Direction')
        
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel('UMAP 1', fontsize=FONT_AXIS_LABEL)
        ax.set_ylabel('UMAP 2', fontsize=FONT_AXIS_LABEL)
        ax.tick_params(axis='both', labelsize=FONT_TICK)
        ax.set_title(title or 'KO Edge Projection', fontsize=FONT_TITLE)
        
        return ax
    
    # =========================================================================
    # STATE PROFILES
    # =========================================================================
    
    def plot_state_profiles(self,
                            figsize: Tuple[float, float] = (14, 8),
                            node_size: int = 400,
                            edge_alpha: float = 0.4,
                            cmap_range: Tuple[float, float] = (-2, 2),
                            auto_scale: bool = False,
                            negative_color: str = '#888888',
                            normalize: str = None,
                            penalty: float = 1.0,
                            save_path: str = None) -> plt.Figure:
        """
        Six-panel plot showing per-state z-scores across clusters.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        node_size : int
            Size of cluster nodes
        edge_alpha : float
            Edge transparency
        cmap_range : tuple
            (vmin, vmax) for colormap, used when auto_scale=False
        auto_scale : bool
            If True, each panel uses its own symmetric range based on data.
        negative_color : str
            Color for negative z-scores
        normalize : str or None
            None - absolute z-scores
            'relative' - z * (z/sum_positive)^penalty for positive values
            'relative_both' - applies to both positive and negative
        penalty : float
            Exponent for dominance penalty
        save_path : str
            Path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        z_profiles = self.proj.profile_zscore.copy()

        if normalize == 'relative':
            zp = z_profiles.copy()
            zp[zp <= 0] = 0
            row_sum = zp.sum(axis=1).replace(0, 1)
            ratio = zp.div(row_sum, axis=0).clip(lower=0)
            penalized = z_profiles * (ratio ** penalty)
            z_profiles = z_profiles.where(z_profiles <= 0, penalized)
            
        elif normalize == 'relative_both':
            # Positive: scale by proportion of total enrichment
            zp = z_profiles.copy()
            zp[zp <= 0] = 0
            row_sum_pos = zp.sum(axis=1).replace(0, 1)
            ratio_pos = zp.div(row_sum_pos, axis=0)
            penalized_pos = z_profiles * (ratio_pos ** penalty)

            # Negative: scale by proportion of total depletion
            zn = z_profiles.copy()
            zn[zn >= 0] = 0
            row_sum_neg = zn.abs().sum(axis=1).replace(0, 1)
            ratio_neg = zn.abs().div(row_sum_neg, axis=0)
            penalized_neg = z_profiles * (ratio_neg ** penalty)

            # Combine
            z_profiles = penalized_pos.where(z_profiles > 0, penalized_neg)

        centroids = self.proj.centroids_umap
        edges = self.graph['edges']

        for idx, state in enumerate(STATE_ORDER):
            ax = axes[idx]
            z_vals = z_profiles[state]

            state_rgb = mcolors.hex2color(STATE_COLORS[state])
            neg_rgb = mcolors.hex2color(negative_color)

            cmap = mcolors.LinearSegmentedColormap.from_list(
                f'{state}_cmap', 
                [neg_rgb, 'white', state_rgb],
                N=256
            )

            # Set scale per panel or global
            if auto_scale:
                abs_max = max(abs(z_vals.min()), abs(z_vals.max()), 0.1)
                vmin, vmax = -abs_max, abs_max
            else:
                vmin, vmax = cmap_range

            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

            for c1, c2 in edges:
                x1, y1 = centroids.loc[c1, ['x', 'y']]
                x2, y2 = centroids.loc[c2, ['x', 'y']]
                ax.plot([x1, x2], [y1, y2], 'k-', lw=1.5, alpha=edge_alpha, zorder=1)

            for cluster in z_profiles.index:
                x, y = centroids.loc[cluster, ['x', 'y']]
                z = z_vals[cluster]
                color = cmap(norm(z))

                ax.scatter(x, y, c=[color], s=node_size, 
                          edgecolors='black', linewidths=1.5, zorder=5)

                short_name = cluster.replace('Homeostatic', 'Hom').replace('_Act', '_A')
                ax.annotate(f'{short_name}', (x, y+0.5), 
                           fontsize=6, ha='center', va='center', fontweight='normal', zorder=6)

            ax.set_title(state, fontsize=FONT_TITLE, color=STATE_COLORS[state])
            ax.margins(0.1)
            ax.set_aspect('equal')
            ax.axis('off')

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
            cbar.set_label('z-score', fontsize=FONT_AXIS_LABEL)
            cbar.ax.tick_params(labelsize=FONT_TICK)
        
        title = 'State Profiles Across Clusters'
        if auto_scale:
            title += ' [auto-scaled]'
        plt.suptitle(title, fontsize=FONT_TITLE, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

        return fig
    
    # =========================================================================
    # EDGE DISTRIBUTION
    # =========================================================================
    
    def plot_edge_distribution(self,
                               ax: Optional[plt.Axes] = None,
                               sort_by: str = 'count',
                               show_confidence: bool = False,
                               figsize: Tuple[float, float] = (10, 6),
                               title: str = None) -> plt.Axes:
        """
        Bar plot of edge assignment distribution.
        
        Parameters
        ----------
        ax : plt.Axes, optional
        sort_by : str
            'count' or 'confidence' or 'alphabetical'
        show_confidence : bool
            Show mean confidence per edge instead of count
        figsize : tuple
        title : str
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        if show_confidence:
            data = self.results.groupby('edge')['confidence'].mean().sort_values(ascending=False)
            xlabel = 'Mean Confidence'
        else:
            data = self.results['edge'].value_counts()
            if sort_by == 'alphabetical':
                data = data.sort_index()
            xlabel = 'Count'
        
        # Colors based on edge type
        colors = []
        for edge in data.index:
            if 'DAMs' in edge and 'Home' not in edge and 'Hom' not in edge:
                colors.append(self.direction_colors['toward_disease'])
            else:
                colors.append(self.direction_colors['toward_homeostatic'])
        
        bars = ax.barh(range(len(data)), data.values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([e.replace('--', ' ↔ ') for e in data.index], fontsize=FONT_TICK)
        ax.set_xlabel(xlabel, fontsize=FONT_AXIS_LABEL)
        ax.set_title(title or 'Edge Distribution', fontsize=FONT_TITLE)
        ax.invert_yaxis()
        ax.tick_params(axis='x', labelsize=FONT_TICK)
        
        # Value labels
        for bar, val in zip(bars, data.values):
            label = f'{val:.2f}' if show_confidence else str(int(val))
            ax.text(val + 0.02 * data.max(), bar.get_y() + bar.get_height()/2,
                   label, va='center', fontsize=FONT_TICK)
        
        return ax
    
    # =========================================================================
    # CONFIDENCE DISTRIBUTION
    # =========================================================================
    
    def plot_confidence_distribution(self,
                                     ax: Optional[plt.Axes] = None,
                                     by_direction: bool = True,
                                     bins: int = 15,
                                     figsize: Tuple[float, float] = (8, 5),
                                     density: bool = False,
                                     title: str = None) -> plt.Axes:
        """
        Histogram of confidence scores.
        
        Parameters
        ----------
        ax : plt.Axes, optional
        by_direction : bool
            Color by direction
        bins : int
            Number of histogram bins
        figsize : tuple
        density : bool
            Normalize histogram
        title : str
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        if by_direction:
            for direction in ['toward_disease', 'toward_homeostatic']:
                subset = self.results[self.results['direction'] == direction]
                ax.hist(subset['confidence'], bins=bins, alpha=0.6, density=density,
                       color=self.direction_colors[direction],
                       label=direction.replace('_', ' ').title(),
                       edgecolor='black')
        else:
            ax.hist(self.results['confidence'], bins=bins, alpha=0.7,
                   color='steelblue', edgecolor='black')
        
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel('Confidence', fontsize=FONT_AXIS_LABEL)
        ax.set_ylabel('Count', fontsize=FONT_AXIS_LABEL)
        ax.set_title(title or 'Confidence Distribution', fontsize=FONT_TITLE)
        ax.tick_params(axis='both', labelsize=FONT_TICK)
        ax.legend(fontsize=FONT_TICK)
        
        # Stats annotation
        stats = f"Mean: {self.results['confidence'].mean():.2f}\n"
        stats += f"Std: {self.results['confidence'].std():.2f}\n"
        stats += f">=0.5: {(self.results['confidence'] >= 0.5).sum()}/{len(self.results)}"
        ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=FONT_TICK,
               va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return ax
    
    # =========================================================================
    # SINGLE KO PLOT
    # =========================================================================
    
    def plot_ko_on_edge(self,
                        ko_name: str,
                        ax: Optional[plt.Axes] = None,
                        show_all_edges: bool = True,
                        figsize: Tuple[float, float] = (10, 8),
                        title: str = None) -> plt.Axes:
        """
        Plot a single KO's projection with detailed information.
        
        Parameters
        ----------
        ko_name : str
            Name of KO to plot
        ax : plt.Axes, optional
        show_all_edges : bool
            Show all graph edges
        figsize : tuple
        title : str
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        ko_data = self.results[self.results['ko'] == ko_name].iloc[0]
        
        # Background cells (lighter)
        for cluster in self.clusters:
            mask = self.sc['labels_merged'] == cluster
            ax.scatter(self.sc.loc[mask, 'x'], self.sc.loc[mask, 'y'],
                      c=self.cluster_colors.get(cluster, 'gray'),
                      s=2, alpha=0.05, rasterized=True)
        
        # All edges
        for c1, c2 in self.graph['edges']:
            x1, y1 = self.centroids.loc[c1, ['x', 'y']]
            x2, y2 = self.centroids.loc[c2, ['x', 'y']]
            is_assigned = f"{c1}--{c2}" == ko_data['edge'] or f"{c2}--{c1}" == ko_data['edge']
            
            if is_assigned:
                ax.plot([x1, x2], [y1, y2], linestyle='-', color='darkred', lw=3, alpha=0.75, zorder=3)
            elif show_all_edges:
                ax.plot([x1, x2], [y1, y2], 'k-', lw=1, alpha=0.3, zorder=2)
        
        # Centroids
        for cluster in self.clusters:
            x, y = self.centroids.loc[cluster, ['x', 'y']]
            color = self.cluster_colors.get(cluster, 'gray')
            is_endpoint = cluster in [ko_data['c1'], ko_data['c2']]
            size = 600 if is_endpoint else 300
            edge_width = 4 if is_endpoint else 2
            ax.scatter(x, y, c=color, s=size, zorder=5, alpha=.75,
                      edgecolors='black' if not is_endpoint else 'darkred',
                      linewidths=edge_width)
            ax.annotate(cluster.replace('_', '\n'), (x, y), fontsize=FONT_AXIS_LABEL,
                       fontweight='bold', ha='center', va='bottom',
                       xytext=(0, 20), textcoords='offset points')
        
        # KO position
        color = self.direction_colors[ko_data['direction']]
        ax.scatter(ko_data['umap_x'], ko_data['umap_y'], c=color, s=300,
                  zorder=10, edgecolors='black', linewidths=1.5, marker='*')
        ax.annotate(ko_name, (ko_data['umap_x'], ko_data['umap_y']),
                   fontsize=FONT_TICK,
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Info box
        info = f"KO: {ko_name}\n"
        info += f"Edge: {ko_data['edge']}\n"
        info += f"Position: {ko_data['t']:.2f}\n"
        info += f"Confidence: {ko_data['confidence']:.2f}\n"
        info += f"Direction: {ko_data['direction'].replace('_', ' ')}"
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=FONT_TICK,
               va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel('UMAP 1', fontsize=FONT_AXIS_LABEL)
        ax.set_ylabel('UMAP 2', fontsize=FONT_AXIS_LABEL)
        ax.tick_params(axis='both', labelsize=FONT_TICK)
        ax.set_title(title or f'{ko_name} Projection', fontsize=FONT_TITLE)
        
        return ax
