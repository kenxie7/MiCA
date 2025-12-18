<p align="center">
  <img src="imgs/mica.png" alt="MiCA" align="left" width="125px">
</p>

<p>MiCA is a python package for miCRISPR-seq analysis. We developed a probabilistic assignment of knockout perturbations onto cell state transition graphs in defined microglial states to show the effect of perturbation on the trajectory, revealing important regulators of distinct putative microglial states in the context of Alzheimer's Disease.
</p> 


<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API</a> •
  <a href="#visualization">Visualization</a> •
  <a href="#method">Method</a>
</p>

---

## Overview

**Trajectory Projection** maps CRISPR knockout effects onto a graph of cell state transitions. Given perturbation enrichment scores across cell states, the method:

1. Constructs a **Gabriel graph** connecting cell type clusters in state space
2. **Probabilistically assigns** each perturbation to a graph edge
3. Returns **position** (where along the transition) and **confidence** scores

This reveals not just *which* states are affected, but *where along the trajectory* each gene acts — transforming pooled screen data into interpretable phenotypic coordinates.

<p align="center">
  <img src="assets/trajectory_example.png" alt="Example trajectory projection" width="700">
</p>

---

## Installation

```bash
git clone https://github.com/username/trajectory-projection.git
cd trajectory-projection
pip install -r requirements.txt
```

**Requirements:** numpy, pandas, scipy, matplotlib, adjustText

---

## Quick Start

```python
from edge_projection import EdgeProjector
from edge_projection_plots import ProjectionPlotter

# Load your data
sc_data = pd.read_csv('single_cell_data.csv')  # cells × states
ko_data = pd.read_csv('ko_enrichments.csv')    # knockouts × states

# Initialize projector
proj = EdgeProjector(sc_data, ko_data)

# Build Gabriel graph in state space
proj.build_graph(method='gabriel', space='state')

# Project all knockouts
results = proj.project_all(model='cosine')

# Visualize
plotter = ProjectionPlotter(proj, results)
plotter.plot_projection(node_color_method='raw')
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
| `sc_data` | Single-cell DataFrame with columns: `labels_merged`, `x`, `y`, and state scores |
| `ko_data` | KO enrichment DataFrame with columns: `crispr_guides2` and state scores |
| `states` | List of state columns (default: `['pdam', 'exdam', 'idam', 'irm', 'img', 'hm']`) |

### `build_graph()`

```python
proj.build_graph(method='gabriel', space='state', k=3)
```

| Parameter | Options | Description |
|-----------|---------|-------------|
| `method` | `'gabriel'`, `'rng'`, `'mst'`, `'knn'`, `'delaunay'` | Graph construction method |
| `space` | `'state'`, `'umap'` | Coordinate space for distances |
| `k` | int | Neighbors for k-NN (ignored for other methods) |

**Recommended:** `method='gabriel', space='state'` — parameter-free, biologically interpretable.

### `project_all()`

```python
results = proj.project_all(model='cosine', tau=0.4, tau_cluster=0.3)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `'cosine'` | Projection model (`'cosine'`, `'simple'`, `'support_spec'`) |
| `tau` | `0.4` | Edge temperature |
| `tau_cluster` | `0.3` | Cluster temperature |

**Returns:** DataFrame with columns: `ko`, `edge`, `c1`, `c2`, `t`, `confidence`, `direction`, `umap_x`, `umap_y`

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
  <img src="assets/projection_plot.png" alt="Projection plot" width="600">
</p>

#### State profiles

```python
plotter.plot_state_profiles(
    normalize='relative',  # Dominance-scaled
    auto_scale=True        # Per-panel color scaling
)
```

<p align="center">
  <img src="assets/state_profiles.png" alt="State profiles" width="700">
</p>

#### Other plots

```python
plotter.plot_edge_distribution()      # Bar plot of edge assignments
plotter.plot_confidence_distribution() # Confidence histogram
plotter.plot_ko_on_edge('Trem2')       # Single KO detail
plotter.plot_summary()                 # 4-panel summary figure
```

---

## Method

The algorithm projects perturbations onto a biologically-informed graph of cell states:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Algorithm: Perturbation Trajectory Projection                           │
├──────────────────────────────────────────────────────────────────────────┤
│ Input: KO enrichment vectors {zₖ}, cluster centroids {cⱼ} in state space│
│ Output: Edge assignment e*, position t ∈ (0,1), confidence              │
│                                                                          │
│ 1. Z-score normalize centroids against control cells                    │
│ 2. Construct Gabriel graph on centroids in state space                  │
│ 3. For each knockout k:                                                 │
│    ├─ Compute cluster activations via cosine similarity                 │
│    ├─ Convert to cluster probabilities via softmax                      │
│    ├─ Score edges by geometric mean + specificity                       │
│    ├─ Assign to maximum-scoring edge e*                                 │
│    ├─ Compute position t via sigmoid on activation difference           │
│    └─ Compute confidence from normalized edge posterior entropy         │
│                                                                          │
│ Return edge assignments, positions, and confidence scores               │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why Gabriel graph?** Unlike k-NN (requires choosing k) or MST (too sparse), the Gabriel graph is parameter-free and connects clusters only when no intermediate state "blocks" the direct transition.

**Why cosine similarity?** Measures pattern alignment independent of perturbation magnitude — robust across strong and weak effects.

---

## Citation

This work is currently under submission

```

---
