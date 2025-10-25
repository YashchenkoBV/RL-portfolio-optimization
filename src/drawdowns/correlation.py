import numpy as np
import pandas as pd

__all__ = [
    "compute_window_spearman",
    "fisher_z_average",
    "correlation_distance",
    "compute_corr_and_distance",
]

def _metric_order(panel: pd.DataFrame) -> pd.Index:
    # Preserve original metric order as it appears in panel's (metric, ticker) columns
    return panel.columns.get_level_values(0).unique()

def compute_window_spearman(panel: pd.DataFrame) -> dict:
    """
    For each window (row in panel), compute a cross-sectional Spearman correlation
    matrix between metrics (columns = metrics, rows = metrics). Returns a dict:
        {window_end_timestamp: corr_df}
    """
    metrics = _metric_order(panel)
    out = {}
    for t, row in panel.iterrows():
        # row is a Series with MultiIndex (metric, ticker)
        # -> X: rows=tickers, cols=metrics (no sorting)
        X = row.unstack(0)[metrics]  # shape: (n_tickers, n_metrics)

        # Spearman = Pearson on ranks (average ranks for ties, pairwise deletion for NaNs)
        R = X.rank(axis=0, method="average", na_option="keep")
        C = R.corr(method="pearson", min_periods=3)
        # Ensure metric order and square shape
        C = C.reindex(index=metrics, columns=metrics)
        out[t] = C
    return out

def fisher_z_average(corrs: dict) -> pd.DataFrame:
    """
    Fisher-z aggregate across windows.
    For each pair (a,b): z_t = arctanh(r_t);  z̄ = mean_t z_t;  r̄ = tanh(z̄).
    Ignores NaNs (windows where r_t is undefined). Diagonal set to 1.
    """
    if not corrs:
        return pd.DataFrame()

    # stack each window's corr into long form (a,b) -> r_t
    z_list = []
    for t, C in corrs.items():
        C = C.copy()
        # drop diagonal to avoid arctanh(1)
        np.fill_diagonal(C.values, np.nan)
        r = C.stack(dropna=False)  # MultiIndex (a,b)
        r = r.clip(-0.999999, 0.999999)  # guard against ±1
        z = np.arctanh(r)
        z.name = t
        z_list.append(z)

    Z = pd.concat(z_list, axis=1)                 # rows=(a,b), cols=windows
    zbar = Z.mean(axis=1, skipna=True)            # Fisher-z mean
    rbar = np.tanh(zbar)                          # back-transform to correlation

    # rebuild square matrix in original metric order
    metrics = corrs[next(iter(corrs))].index
    R = rbar.unstack().reindex(index=metrics, columns=metrics)
    np.fill_diagonal(R.values, 1.0)
    return R

def correlation_distance(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation distance: d_ab = sqrt( 2*(1 - rho_ab) ), diagonal = 0.
    """
    R = corr_matrix.copy()
    D = np.sqrt(2.0 * (1.0 - R))
    np.fill_diagonal(D.values, 0.0)
    D.index = R.index
    D.columns = R.columns
    return D

def compute_corr_and_distance(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience: from rolling panel -> Fisher-z averaged correlation and distance matrices.
    """
    per_window = compute_window_spearman(panel)
    R = fisher_z_average(per_window)
    D = correlation_distance(R)
    return R, D
