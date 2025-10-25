import pandas as pd

from sklearn.cluster import AgglomerativeClustering

__all__ = ["agglomerative_cluster"]

def agglomerative_cluster(distance: pd.DataFrame, n_clusters: int, linkage: str = "average") -> tuple[pd.Series, pd.DataFrame]:
    """
    Run hierarchical agglomerative clustering on a precomputed distance matrix.

    Parameters
    ----------
    distance : pd.DataFrame
        Square, symmetric distance matrix between metrics. Index = columns = metric names.
    n_clusters : int
        Desired number of clusters.
    linkage : str, default "average"
        Linkage strategy. For precomputed distances, use one of {"average","complete","single"}.
        Note: 'ward' does NOT support precomputed distances.

    Returns
    -------
    labels : pd.Series
        Cluster labels (0..n_clusters-1), indexed by metric, preserving the original order.
    distance : pd.DataFrame
        The same distance matrix (use as-is for MDS).
    """
    if linkage == "ward":
        raise ValueError("Ward linkage requires feature vectors, not a precomputed distance matrix. "
                         "Use linkage='average' (or 'complete'/'single'), or embed to features first.")

    # scikit-learn API compatibility: metric='precomputed' (new) vs affinity='precomputed' (old)
    D = distance.values
    try:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric="precomputed")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity="precomputed")

    labels_arr = model.fit_predict(D)
    labels = pd.Series(labels_arr, index=distance.index, name="cluster")
    return labels, distance
