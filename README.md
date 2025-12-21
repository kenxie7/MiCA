<p align="center">
  <img src="imgs/mica.png" alt="MiCA" align="left" width="125px">
</p>

<p>MiCA is a python package for miCRISPR-seq analysis. We developed a probabilistic assignment of knockout perturbations onto cell state transition graphs in defined microglial states to show the effect of perturbation on the trajectory, revealing important regulators of distinct putative microglial states in the context of Alzheimer's Disease.
</p> 

---
<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API</a> •
  <a href="#visualization">Visualization</a> •
  <a href="#todo">Todos</a>
</p>

## Overview

**MiCA** projects CRISPR knockout effects onto a graph of cell state transitions developed for miCRISPR-seq data. Given perturbation enrichment scores across cell states, the method:

1. Constructs a **Gabriel graph** connecting cell type clusters in state space
2. **Probabilistically assigns** each perturbation to a graph edge via cosine similarity
3. Returns **position** (where along the transition) and **confidence** scores

This reveals not just *which* states are affected, but *where along the trajectory* each gene acts — transforming pooled screen data into interpretable phenotypic coordinates.

<p align="center">
  <img src="imgs/trajectory_example.png" alt="Example trajectory projection" width="700">
</p>

---

## Installation

```bash
pip install mica
# or from source
git clone https://github.com/username/mica.git
cd mica
pip install -e .
```

**Requirements:** numpy, pandas, scipy, matplotlib, adjustText, igraph

---

## Quick Start

```python
from mica import EdgeProjector, ProjectionPlotter, load_data

# Load data
sc_data, ko_data = load_data('single_cell.csv', 'ko_enrichments.csv')

# Initialize and build graph
proj = EdgeProjector(sc_data, ko_data)
proj.build_graph()  # Default: Gabriel graph in state space

# Project all knockouts
results = proj.project_all()

# Visualize
plotter = ProjectionPlotter(proj, results)
plotter.plot_projection(node_color_method='raw')
plotter.plot_state_profiles(auto_scale=True)
plotter.plot_confidence_distribution()
```

**Output:**
| ko | edge | t | confidence |
|----|------|---|------------|
| Trem2 | iHM--pDAM | 0.32 | 0.84 |
| Mef2c | HM--iHM | 0.67 | 0.71 |
| ... | ... | ... | ... |

---

## API Reference

### `EdgeProjector`

```python
proj = EdgeProjector(sc_data, ko_data, states=None)
```

| Parameter | Description |
|-----------|-------------|
| `sc_data` | Single-cell DataFrame with: `labels_merged`, `x`, `y`, `crispr_guides2`, state scores |
| `ko_data` | KO enrichment DataFrame with: `crispr_guides2`, state scores |
| `states` | List of state columns (default: `['pdam', 'exdam', 'idam', 'irm', 'img', 'hm']`) |

### `build_graph()`

```python
proj.build_graph(method='gabriel', space='state', k=3)
```

| Parameter | Default | Options |
|-----------|---------|---------|
| `method` | `'gabriel'` | `'gabriel'`, `'mst'`, `'knn'` |
| `space` | `'state'` | `'state'`, `'umap'` |
| `k` | `3` | k for k-NN (ignored for other methods) |

### `project_all()`

```python
results = proj.project_all(tau=0.4, tau_cluster=0.3)
```

Returns DataFrame with: `ko`, `edge`, `c1`, `c2`, `t`, `confidence`, `direction`, `umap_x`, `umap_y`

---

## Visualization

### `ProjectionPlotter`

```python
plotter = ProjectionPlotter(proj, results)
```


#### Main projection plot

```python
plotter.plot_projection(
    node_color_method='raw',   # Color nodes by dominant state
    show_state_legend=True,    # Vertical state legend
    label_top_n=10,            # Label top confident KOs
    color_by='direction'       # 'direction', 'confidence', or 'edge'
)
```

<p align="center">
  <img src="imgs/projection_plot.png" alt="Projection plot" width="600">
</p>

#### State profiles

```python
plotter.plot_state_profiles(
    normalize='relative',  # Dominance-scaled
    auto_scale=True        # Per-panel color scaling
)
```

#### Other plots

```python
plotter.plot_edge_distribution()       # Bar plot of edge assignments
plotter.plot_confidence_distribution() # Confidence histogram
plotter.plot_ko_on_edge('Trem2')       # Single KO detail
```

---

## Todo
- Anndata/DataFrame input support
- Calculation of module scores and pathway scores
- Calculation of various states

```


## Citation

This work is currently under submission

```

---
