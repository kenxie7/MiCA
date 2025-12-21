"""
MiCA Projector Module
=====================

Core projection algorithm for mapping perturbations onto cell state graphs.
"""

import numpy as np
import pandas as pd
from scipy.special import softmax, expit
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Literal
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ProjectionResult:
    """Result of projecting a single KO onto the graph."""
    ko: str
    edge: str
    c1: str
    c2: str
    t: float
    edge_prob: float
    confidence: float
    umap_x: float
    umap_y: float
    direction: str
    scores: Dict[str, float]
    posterior: Dict[str, float]


class EdgeProjector:
    """
    Project KO enrichments onto a graph of cell type clusters.
    
    Parameters
    ----------
    sc_data : pd.DataFrame
        Single-cell data with columns: labels_merged, x, y, crispr_guides2, and state scores
    ko_data : pd.DataFrame
        KO-level enrichment data with columns: crispr_guides2 and state scores
    states : list, optional
        State column names (default: ['pdam', 'exdam', 'idam', 'irm', 'img', 'hm'])
    
    Examples
    --------
    >>> proj = EdgeProjector(sc_data, ko_data)
    >>> proj.build_graph(method='gabriel')
    >>> results = proj.project_all()
    """
    
    def __init__(self, sc_data: pd.DataFrame, ko_data: pd.DataFrame,
                 states: List[str] = None):
        self.sc = sc_data
        self.sc_ctrl = sc_data[sc_data.crispr_guides2 == 'CTRL']
        self.sc_ko = sc_data[sc_data.crispr_guides2 != 'CTRL']
        self.ko_df = ko_data
        self.states = states or ['pdam', 'exdam', 'idam', 'irm', 'img', 'hm']
        
        # Compute cluster profiles and centroids
        self.clusters = list(self.sc['labels_merged'].unique())
        self.profile_raw = self.sc.groupby('labels_merged')[self.states].mean()
        self.profile_zscore = (self.profile_raw - self.profile_raw.mean()) / self.profile_raw.std()
        self.profile_zscore_orig = (self.profile_raw - self.profile_raw.mean()) / self.profile_raw.std()

        # Z-score normalization against control
        self.profile_r2 = self.sc_ko.groupby('labels_merged')[self.states].mean()
        self.profile_r2c = self.sc_ctrl.groupby('labels_merged')[self.states].mean()
        self.profile_zscore = (self.profile_r2 - self.profile_r2c.mean()) / self.profile_r2c.std()

        # UMAP centroids (2D)
        self.centroids_umap = self.sc.groupby('labels_merged')[['x', 'y']].mean()
        
        # State centroids (6D)
        self.centroids_state = self.profile_raw.copy()
        
        # Default centroids for backward compatibility
        self.centroids = self.centroids_umap
        
        # Graph (initialized by build_graph)
        self.graph = None
    
    # =========================================================================
    # GRAPH CONSTRUCTION METHODS
    # =========================================================================
    
    def _build_knn(self, points: np.ndarray, clusters: List[str], k: int) -> List[Tuple[str, str]]:
        """k-nearest neighbors graph."""
        n = len(clusters)
        dist_matrix = squareform(pdist(points))
        adj = np.zeros((n, n))
        for i in range(n):
            neighbors = np.argsort(dist_matrix[i])[1:k+1]
            for j in neighbors:
                adj[i, j] = adj[j, i] = 1
        return [(clusters[i], clusters[j]) for i in range(n) for j in range(i+1, n) if adj[i, j] > 0]
    
    def _build_mst(self, points: np.ndarray, clusters: List[str]) -> List[Tuple[str, str]]:
        """Minimum spanning tree."""
        import igraph as ig
        n = len(clusters)
        dist_matrix = squareform(pdist(points))
        g = ig.Graph.Full(n)
        weights = [dist_matrix[e.tuple[0], e.tuple[1]] for e in g.es]
        mst = g.spanning_tree(weights=weights)
        return [(clusters[e.tuple[0]], clusters[e.tuple[1]]) for e in mst.es]
    
    def _build_gabriel(self, points: np.ndarray, clusters: List[str]) -> List[Tuple[str, str]]:
        """
        Gabriel graph - edge (i,j) exists iff no other point lies inside
        the hypersphere with diameter ij.
        
        Works in any dimension. Parameter-free and biologically interpretable.
        """
        n = len(clusters)
        dist_matrix = squareform(pdist(points))
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                midpoint = (points[i] + points[j]) / 2
                radius_sq = (dist_matrix[i, j] / 2) ** 2
                is_gabriel = True
                for k in range(n):
                    if k in [i, j]:
                        continue
                    if np.sum((points[k] - midpoint) ** 2) < radius_sq:
                        is_gabriel = False
                        break
                if is_gabriel:
                    edges.append((clusters[i], clusters[j]))
        return edges
        
    def build_graph(self, k: int = 3, 
                    method: Literal['knn', 'mst', 'gabriel'] = 'gabriel',
                    space: Literal['umap', 'state'] = 'state') -> Dict:
        """
        Build graph from cluster centroids.
        
        Parameters
        ----------
        k : int
            Number of neighbors for k-NN graph (ignored for other methods)
        method : str
            'gabriel': Gabriel graph (default, parameter-free)
            'mst': Minimum spanning tree
            'knn': k-nearest neighbors
        space : str
            'state': 6D state score space (default)
            'umap': 2D UMAP coordinates
            
        Returns
        -------
        dict
            Graph info with keys: edges, clusters, method, space, coords
        """
        clusters = list(self.centroids_umap.index)
        
        # Get coordinates
        if space == 'umap':
            points = self.centroids_umap[['x', 'y']].values
        elif space == 'state':
            points = self.centroids_state[self.states].values
        else:
            raise ValueError(f"Unknown space: {space}")
        
        # Build graph
        if method == 'knn':
            edges = self._build_knn(points, clusters, k)
        elif method == 'mst':
            edges = self._build_mst(points, clusters)
        elif method == 'gabriel':
            edges = self._build_gabriel(points, clusters)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'gabriel', 'mst', or 'knn'.")
        
        # Compute edge distances
        dist_matrix = squareform(pdist(points))
        cluster_idx = {c: i for i, c in enumerate(clusters)}
        
        self.graph = {
            'edges': edges,
            'clusters': clusters,
            'method': method,
            'space': space,
            'n_edges': len(edges),
            'coords': {c: self.centroids_umap.loc[c, ['x', 'y']].values for c in clusters},
            'distances': {f"{c1}--{c2}": dist_matrix[cluster_idx[c1], cluster_idx[c2]] 
                         for c1, c2 in edges}
        }
        
        print(f"Built {method.upper()} graph in {space.upper()} space: {len(edges)} edges")
        return self.graph
    
    # =========================================================================
    # PROJECTION
    # =========================================================================
    
    def _compute_direction(self, enrichment: Dict[str, float]) -> str:
        """Classify KO direction based on state enrichment contrast."""
        disease = enrichment.get('pdam', 0) + enrichment.get('exdam', 0) + enrichment.get('idam', 0)
        homeostatic = enrichment.get('hm', 0) + enrichment.get('img', 0)
        return 'toward_disease' if disease > homeostatic else 'toward_homeostatic'
    
    def _project_cosine(self, enrichment: Dict[str, float], 
                        tau: float = 0.4, tau_cluster: float = 0.3) -> ProjectionResult:
        """Cosine similarity model (recommended)."""
        edges = self.graph['edges']
        clusters = self.graph['clusters']
        
        z = np.array([enrichment[s] for s in self.states])
        z_norm = z / (np.linalg.norm(z) + 1e-8)
        
        activations = {}
        for c in clusters:
            p = self.profile_zscore.loc[c, self.states].values
            p_norm = p / (np.linalg.norm(p) + 1e-8)
            activations[c] = np.dot(z_norm, p_norm)
        
        P_cluster = softmax(np.array([activations[c] for c in clusters]) / tau_cluster)
        P_dict = dict(zip(clusters, P_cluster))
        
        edge_scores = []
        edge_details = []
        for c1, c2 in edges:
            p1, p2 = P_dict[c1], P_dict[c2]
            joint = np.sqrt(p1 * p2)
            others = [P_dict[c] for c in clusters if c not in [c1, c2]]
            spec = (p1 + p2) / 2 - np.mean(others)
            edge_scores.append(np.log(joint + 1e-10) + spec)
            edge_details.append({'c1': c1, 'c2': c2, 'a1': activations[c1], 'a2': activations[c2]})
        
        edge_scores = np.array(edge_scores) / tau
        edge_scores = edge_scores - edge_scores.max()
        posterior = np.exp(edge_scores) / np.exp(edge_scores).sum()
        
        idx = np.argmax(posterior)
        info = edge_details[idx]
        c1, c2 = info['c1'], info['c2']
        t = expit(3.0 * (info['a2'] - info['a1']))
        
        coord1, coord2 = self.graph['coords'][c1], self.graph['coords'][c2]
        umap_pos = np.array(coord1) + t * (np.array(coord2) - np.array(coord1))
        
        return ProjectionResult(
            ko='', edge=f"{c1}--{c2}", c1=c1, c2=c2, t=t,
            edge_prob=posterior[idx],
            confidence=1 - entropy(posterior) / np.log(len(edges)),
            umap_x=umap_pos[0], umap_y=umap_pos[1],
            direction=self._compute_direction(enrichment),
            scores=activations,
            posterior=dict(zip([f"{e[0]}--{e[1]}" for e in edges], posterior))
        )
    
    # =========================================================================
    # MAIN API
    # =========================================================================
    
    def project(self, enrichment: Dict[str, float], 
                tau: float = 0.4, tau_cluster: float = 0.3) -> ProjectionResult:
        """
        Project a single KO enrichment onto the graph.
        
        Parameters
        ----------
        enrichment : dict
            State enrichment scores {state_name: score}
        tau : float
            Edge temperature (default 0.4)
        tau_cluster : float
            Cluster temperature (default 0.3)
            
        Returns
        -------
        ProjectionResult
        """
        if self.graph is None:
            raise ValueError("Must call build_graph() first")
        return self._project_cosine(enrichment, tau=tau, tau_cluster=tau_cluster)
    
    def project_all(self, tau: float = 0.4, tau_cluster: float = 0.3) -> pd.DataFrame:
        """
        Project all KOs in ko_data.
        
        Parameters
        ----------
        tau : float
            Edge temperature (default 0.4)
        tau_cluster : float
            Cluster temperature (default 0.3)
        
        Returns
        -------
        pd.DataFrame
            Projection results with columns: ko, edge, c1, c2, t, 
            confidence, direction, umap_x, umap_y, z_<state>
        """
        results = []
        for _, row in self.ko_df.iterrows():
            ko = row['crispr_guides2']
            enrichment = {s: row[s] for s in self.states}
            result = self.project(enrichment, tau=tau, tau_cluster=tau_cluster)
            result.ko = ko
            results.append({
                'ko': ko, 'edge': result.edge, 'c1': result.c1, 'c2': result.c2,
                't': result.t, 'edge_prob': result.edge_prob, 'confidence': result.confidence,
                'umap_x': result.umap_x, 'umap_y': result.umap_y, 'direction': result.direction,
                **{f'z_{s}': row[s] for s in self.states}
            })
        return pd.DataFrame(results)
    
    def summary(self, results: pd.DataFrame) -> None:
        """Print summary statistics."""
        print(f"=== Projection Summary ===")
        print(f"Graph: {self.graph['method'].upper()} in {self.graph['space'].upper()} space")
        print(f"Edges: {len(self.graph['edges'])}")
        print(f"KOs: {len(results)}")
        print(f"\nConfidence: mean={results['confidence'].mean():.3f}, std={results['confidence'].std():.3f}")
        print(f"Position t: mean={results['t'].mean():.2f}, std={results['t'].std():.2f}")
        print(f"\nEdge distribution:")
        print(results['edge'].value_counts().head(8))


def load_data(sc_path: str, ko_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load single-cell and KO data from CSV files.
    
    Parameters
    ----------
    sc_path : str
        Path to single-cell CSV
    ko_path : str
        Path to KO enrichment CSV
        
    Returns
    -------
    tuple
        (sc_data, ko_data) DataFrames
    """
    sc = pd.read_csv(sc_path, index_col=0)
    ko = pd.read_csv(ko_path)
    return sc, ko
