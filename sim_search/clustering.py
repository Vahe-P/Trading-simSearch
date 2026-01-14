"""
Future Path Clustering.

Identifies distinct market scenarios (e.g., Bull/Bear) by clustering
the future paths of nearest neighbors.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from loguru import logger

def cluster_paths(paths: np.ndarray, max_k: int = 3, min_k: int = 2) -> Dict[str, Any]:
    """
    Cluster a set of future paths into distinct scenarios.
    
    Parameters
    ----------
    paths : np.ndarray
        Array of shape (n_samples, horizon_len) containing future returns/prices.
    max_k : int
        Maximum number of clusters to try.
    min_k : int
        Minimum number of clusters.
        
    Returns
    -------
    Dict containing:
        - labels: cluster label for each path
        - centers: center/centroid of each cluster (distinct scenarios)
        - probabilities: weight/probability of each cluster (count / distinct total)
        - n_clusters: optimal k selected (by silhouette score)
        - score: silhouette score of the breakdown
    """
    n_samples = len(paths)
    
    # Not enough samples to cluster or just 1 sample
    if n_samples < min_k + 1:
        return {
            'labels': np.zeros(n_samples, dtype=int),
            'centers': np.mean(paths, axis=0, keepdims=True),
            'probabilities': [1.0],
            'n_clusters': 1,
            'score': 0.0
        }
        
    best_score = -1
    best_model = None
    best_k = 1
    
    # Flatten paths if needed (KMeans expects 2D: samples x features)
    # paths is already (n, horizon), so it's good.
    
    # Try different K values
    for k in range(min_k, min(max_k + 1, n_samples)):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(paths)
            
            # evaluate
            score = silhouette_score(paths, labels)
            
            if score > best_score:
                best_score = score
                best_model = kmeans
                best_k = k
        except Exception as e:
            logger.warning(f"Clustering failed for k={k}: {e}")
            continue
            
    # Use best model
    if best_model is None or best_score < 0.15: 
        # If score is too low, it means no distinct clusters found. 
        # Fallback to single cluster (average)
        return {
            'labels': np.zeros(n_samples, dtype=int),
            'centers': np.mean(paths, axis=0, keepdims=True),
            'probabilities': [1.0],
            'n_clusters': 1,
            'score': 0.0
        }
    
    # Calculate probabilities
    labels = best_model.labels_
    unique, counts = np.unique(labels, return_counts=True)
    probs = counts / n_samples
    
    # Sort clusters by center's final return (Bearish to Bullish) for consistent coloring/ordering
    # center shape: (k, horizon)
    centers = best_model.cluster_centers_
    final_point = centers[:, -1]
    sorted_indices = np.argsort(final_point)
    
    # Remap labels and centers to sorted order
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
    new_labels = np.array([mapping[l] for l in labels])
    new_centers = centers[sorted_indices]
    new_probs = probs[sorted_indices]
    
    return {
        'labels': new_labels,
        'centers': new_centers,
        'probabilities': new_probs,
        'n_clusters': best_k,
        'score': best_score
    }
