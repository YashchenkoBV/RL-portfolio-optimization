import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

__all__ = ["mds_embed", "plot_clusters_mds", "embed_and_plot"]

# --- updated mds_embed with dimensionality option ---
def mds_embed(
    distance: pd.DataFrame,
    n_components: int = 2,
    random_state: int = 0
) -> pd.DataFrame:
    """
    MDS on a precomputed distance matrix.
    Returns a DataFrame with columns ['x','y'] or ['x','y','z'] indexed by metric name.
    """
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3.")
    D = distance.values
    # Try with normalized_stress (newer sklearn); fall back if param unsupported
    try:
        mds = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=4,
            max_iter=300,
            normalized_stress="auto",
        )
    except TypeError:
        mds = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=4,
            max_iter=300,
        )
    X = mds.fit_transform(D)
    cols = ["x", "y"] if n_components == 2 else ["x", "y", "z"]
    coords = pd.DataFrame(X, index=distance.index, columns=cols)
    return coords


# --- updated plot function to handle 2D or 3D ---
def plot_clusters_mds(coords: pd.DataFrame, labels: pd.Series, title: str | None = None):
    """
    Scatter plot with per-point text labels (metric names) and cluster ids.
    Supports 2D (columns ['x','y']) or 3D (['x','y','z']).
    Returns (fig, ax). Does not show() or savefig().
    """
    lab = labels.reindex(coords.index)

    if {"x", "y", "z"}.issubset(coords.columns):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        for cid in sorted(lab.dropna().unique()):
            sel = lab == cid
            ax.scatter(coords.loc[sel, "x"], coords.loc[sel, "y"], coords.loc[sel, "z"], label=f"cluster {int(cid)}", s=40)
        for name, row in coords.iterrows():
            ax.text(row["x"], row["y"], row["z"], name, fontsize=8)
        ax.set_xlabel("Dim-1")
        ax.set_ylabel("Dim-2")
        ax.set_zlabel("Dim-3")
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        for cid in sorted(lab.dropna().unique()):
            sel = lab == cid
            ax.scatter(coords.loc[sel, "x"], coords.loc[sel, "y"], label=f"cluster {int(cid)}", s=40)
        for name, (x, y) in coords.iterrows():
            ax.annotate(name, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Dim-1")
        ax.set_ylabel("Dim-2")

    if title:
        ax.set_title(title)
    if lab.notna().any():
        ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    return fig, ax


# --- updated wrapper to pass through dimensionality ---
def embed_and_plot(
    distance: pd.DataFrame,
    labels: pd.Series,
    title: str | None = None,
    n_components: int = 2,
    random_state: int = 0,
):
    """
    Convenience wrapper: run MDS (2D/3D) then plot. Returns (coords, fig, ax).
    """
    coords = mds_embed(distance, n_components=n_components, random_state=random_state)
    fig, ax = plot_clusters_mds(coords, labels, title=title)
    return coords, fig, ax


# --- new PCA embedding on distances (principal coordinates via double-centering) ---
def pca_embed(distance: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    PCA-style embedding from a precomputed distance matrix using double-centering.
    Equivalent to principal coordinates analysis:
        B = -1/2 * J D^2 J,  eig(B) = V diag(w) V^T,  X = V_k diag(sqrt(w_k))
    Returns DataFrame with ['x','y'] or ['x','y','z'] indexed by metric name.
    """
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3.")

    D = distance.values
    D2 = D ** 2
    n = D2.shape[0]
    I = np.eye(n)
    one = np.ones((n, 1))
    J = I - (one @ one.T) / n
    B = -0.5 * (J @ D2 @ J)

    # symmetric -> eigh; sort descending
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]

    k = n_components
    w_k = np.clip(w[:k], a_min=0.0, a_max=None)
    X = V[:, :k] * np.sqrt(w_k)

    cols = ["x", "y"] if k == 2 else ["x", "y", "z"]
    return pd.DataFrame(X, index=distance.index, columns=cols)
