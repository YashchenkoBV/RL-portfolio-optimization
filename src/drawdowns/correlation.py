import numpy as np
import pandas as pd

_SCALAR_METRICS: dict[str, pd.Series] = {}

__all__ = [
    "compute_window_spearman",
    "fisher_z_average",
    "correlation_distance",
    "compute_corr_and_distance",
]


def _metric_order(panel: pd.DataFrame) -> pd.Index:
    # Preserve original metric order as it appears in panel's (metric, ticker) columns
    return panel.columns.get_level_values(0).unique()


def add_scalar_metric_to_corr(
        R: pd.DataFrame,
        panel_oriented: pd.DataFrame,
        scalar: pd.Series,
        name: str,
        agg: str = "mean",
        min_tickers: int = 3,
) -> pd.DataFrame:
    """
    Augment an existing metric correlation matrix R with one scalar-per-ticker
    metric (e.g. CED, MCED, DaR).

    R : metric × metric Spearman correlation matrix (metrics = rows/cols).
    panel_oriented : rolling panel, index = window_end, columns = (metric, ticker).
    scalar : Series[ticker] with one scalar per asset (already oriented if needed).
    name : new metric name to add as row/column in R.

    Math:
      - Correlation between the scalar metric and each rolling metric m is
        Spearman rank correlation between:
            v_m(ticker) = (mean or median over windows of panel_oriented[:, (m, ticker)])
            s(ticker)   = scalar[ticker]
      - Correlation between two scalar metrics (e.g. CED and MCED) is Spearman
        rank correlation between their scalar Series across tickers.
    """
    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'.")

    if not isinstance(panel_oriented.columns, pd.MultiIndex):
        raise ValueError("panel_oriented must have MultiIndex columns (metric, ticker).")

    # Normalize scalar index to string tickers and store it in the registry
    s_scalar = scalar.astype(float).copy()
    s_scalar.index = s_scalar.index.astype(str)
    _SCALAR_METRICS[name] = s_scalar

    # Metrics present in the rolling panel (genuine rolling metrics)
    panel_metrics = panel_oriented.columns.get_level_values(0).unique()
    # Metrics currently represented in the correlation matrix
    corr_metrics = R.index

    # Rolling metrics for which we can use panel_oriented
    base_metrics = corr_metrics.intersection(panel_metrics)
    # Existing scalar metrics (already added previously)
    scalar_metrics_existing = [m for m in corr_metrics if m not in panel_metrics]

    rho = {}

    # 1) Correlation vs rolling metrics (via panel_oriented)
    for m in base_metrics:
        X = panel_oriented.loc[:, (m, slice(None))]     # windows × (m, ticker)
        if isinstance(X.columns, pd.MultiIndex):
            X.columns = X.columns.get_level_values(1)   # columns -> tickers

        v = X.mean(axis=0, skipna=True) if agg == "mean" else X.median(axis=0, skipna=True)
        v.index = v.index.astype(str)

        s = s_scalar.reindex(v.index)
        mask = v.notna() & s.notna()
        if int(mask.sum()) < min_tickers:
            rho[m] = np.nan
            continue

        r1 = v[mask].rank(method="average").to_numpy()
        r2 = s[mask].rank(method="average").to_numpy()
        r1c = r1 - r1.mean()
        r2c = r2 - r2.mean()
        denom = float(np.sqrt((r1c ** 2).sum() * (r2c ** 2).sum()))
        rho[m] = float((r1c * r2c).sum() / denom) if denom > 0.0 else np.nan

    # 2) Correlation vs existing scalar metrics (directly from scalar Series)
    for m in scalar_metrics_existing:
        s_other = _SCALAR_METRICS.get(m)
        if s_other is None:
            continue
        s_other = s_other.astype(float)

        # Align on common tickers
        idx = s_scalar.index.intersection(s_other.index)
        if idx.size < min_tickers:
            rho[m] = np.nan
            continue

        u = s_scalar.loc[idx]
        v = s_other.loc[idx]
        mask = u.notna() & v.notna()
        if int(mask.sum()) < min_tickers:
            rho[m] = np.nan
            continue

        r1 = u[mask].rank(method="average").to_numpy()
        r2 = v[mask].rank(method="average").to_numpy()
        r1c = r1 - r1.mean()
        r2c = r2 - r2.mean()
        denom = float(np.sqrt((r1c ** 2).sum() * (r2c ** 2).sum()))
        rho[m] = float((r1c * r2c).sum() / denom) if denom > 0.0 else np.nan

    # 3) Build full correlation vector for the new metric (one entry per existing metric)
    s_vec = pd.Series(np.nan, index=corr_metrics, dtype=float)
    for m in corr_metrics:
        if m in rho:
            s_vec[m] = rho[m]

    R2 = R.copy()
    # Add new column
    R2[name] = s_vec
    # Add new row (symmetric)
    R2.loc[name] = s_vec.reindex(R2.columns)
    # Set self-correlation to 1
    R2.loc[name, name] = 1.0

    # Enforce symmetry and unit diagonal
    R2 = 0.5 * (R2 + R2.T)
    np.fill_diagonal(R2.values, 1.0)
    return R2




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

    z_list = []
    for t, C in corrs.items():
        C = C.copy()
        # drop diagonal to avoid arctanh(1)
        np.fill_diagonal(C.values, np.nan)

        # Stack to long form (a,b) -> r_t
        try:
            r = C.stack(future_stack=True)  # pandas ≥ 2.1 (no dropna param allowed)
        except TypeError:
            r = C.stack()  # older pandas (default dropna behavior)

        r = r.clip(-0.999999, 0.999999)  # guard against ±1
        z = np.arctanh(r)
        z.name = t
        z_list.append(z)

    Z = pd.concat(z_list, axis=1)  # rows=(a,b), cols=windows
    zbar = Z.mean(axis=1, skipna=True)  # Fisher-z mean
    rbar = np.tanh(zbar)  # back-transform to correlation

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
