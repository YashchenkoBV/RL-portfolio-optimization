import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

__all__ = ["mds_embed", "plot_clusters_mds", "embed_and_plot"]

def mds_embed(distance: pd.DataFrame, random_state: int = 0) -> pd.DataFrame:
    """
    2D MDS on a precomputed distance matrix.
    Returns a DataFrame with columns ['x','y'] indexed by metric name.
    """
    D = distance.values
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=4,
        max_iter=300,
        normalized_stress="auto" if hasattr(MDS, "normalized_stress") else "deprecated",
    )
    X = mds.fit_transform(D)  # shape: (n_metrics, 2)
    coords = pd.DataFrame(X, index=distance.index, columns=["x", "y"])
    return coords

def plot_clusters_mds(coords: pd.DataFrame, labels: pd.Series, title: str | None = None):
    """
    Scatter plot with per-point text labels (metric names) and cluster ids.
    Returns (fig, ax). Does not show() or savefig().
    """
    # align labels to coords index (preserve input order)
    lab = labels.reindex(coords.index)

    fig, ax = plt.subplots(figsize=(7, 6))
    # Use cluster ids to split plotting; avoid specifying explicit colors
    for cid in sorted(lab.dropna().unique()):
        sel = lab == cid
        ax.scatter(coords.loc[sel, "x"], coords.loc[sel, "y"], label=f"cluster {int(cid)}", s=40)

    # annotate every point with its metric name
    for name, (x, y) in coords.iterrows():
        ax.annotate(name, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)

    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    if title:
        ax.set_title(title)
    if lab.notna().any():
        ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    return fig, ax

def embed_and_plot(distance: pd.DataFrame, labels: pd.Series, title: str | None = None, random_state: int = 0):
    """
    Convenience wrapper: run MDS then plot. Returns (coords, fig, ax).
    """
    coords = mds_embed(distance, random_state=random_state)
    fig, ax = plot_clusters_mds(coords, labels, title=title)
    return coords, fig, ax
