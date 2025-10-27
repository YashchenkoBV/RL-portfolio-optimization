# pipeline.py
# End-to-end: load -> rolling metrics -> Spearman/Fisher-z -> HAC -> MDS -> show plot.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from data_processing import load_prices, validate_prices
from rolling_window import rolling_apply
from calculate_metrics import (
    compute_return, compute_volatility, compute_sharpe, compute_sortino,
    compute_var, compute_es, compute_mdd, compute_ulcer_index,
    compute_avg_drawdown, compute_max_dd_duration, compute_avg_dd_duration,
    compute_calmar, compute_csi, compute_cii,
    get_benchmark_prices, compute_beta, compute_r2, compute_tracking_error,
    compute_information_ratio, compute_alpha, compute_treynor,
    compute_up_capture, compute_down_capture,
)
from correlation import compute_corr_and_distance
from clustering import agglomerative_cluster
from visualization import embed_and_plot, pca_embed, plot_clusters_mds

# --------- globals (edit as needed) ---------
DATA_PATH = "data/company_prices.csv"
WINDOW = 63  # primary lens (~1 quarter); change to 42/126 for sensitivity
STEP = 1
MIN_VALID_FRAC = 0.9
N_CLUSTERS = 5
LINKAGE = "average"  # {"average","complete","single"} for precomputed distances
RANDOM_STATE = 0
# --------------------------------------------

BAD_METRICS = {
    "Vol", "VaR95", "ES95", "MDD_pct", "UlcerIndex", "AvgDD", "MaxDD_Dur", "AvgDD_Dur", "CII", "TE",
}


def window_metrics(wdf: pd.DataFrame, bench_prices: pd.Series | None = None) -> pd.Series:
    # core, per-asset metrics
    ret = compute_return(wdf, log=True).rename("Return")
    vol = compute_volatility(wdf, log=True, annualize=True).rename("Vol")
    shrp = compute_sharpe(wdf, risk_free=0.0, log=True, annualize=True).rename("Sharpe")
    sort = compute_sortino(wdf, target=0.0, log=True, annualize=True).rename("Sortino")
    var95 = compute_var(wdf, alpha=0.95, log=True).rename("VaR95")
    es95 = compute_es(wdf, alpha=0.95, log=True).rename("ES95")

    mdd = compute_mdd(wdf).rename("MDD_pct")
    ui = compute_ulcer_index(wdf).rename("UlcerIndex")
    add = compute_avg_drawdown(wdf).rename("AvgDD")
    mdd_d = compute_max_dd_duration(wdf).rename("MaxDD_Dur")
    add_d = compute_avg_dd_duration(wdf).rename("AvgDD_Dur")
    cal = compute_calmar(wdf).rename("Calmar")
    csi = compute_csi(wdf).rename("CSI")
    cii = compute_cii(wdf).rename("CII")

    # benchmark-relative metrics (SPY by default if bench_prices is None)
    beta = compute_beta(wdf, bench_prices=bench_prices).rename("Beta")
    r2 = compute_r2(wdf, bench_prices=bench_prices).rename("R2")
    te = compute_tracking_error(wdf, bench_prices=bench_prices, annualize=True).rename("TE")
    ir = compute_information_ratio(wdf, bench_prices=bench_prices, annualize=True).rename("IR")
    alpha = compute_alpha(wdf, bench_prices=bench_prices, risk_free=0.0, annualize=True).rename("Alpha")
    trey = compute_treynor(wdf, bench_prices=bench_prices, risk_free=0.0, annualize=True).rename("Treynor")
    # ucr = compute_up_capture(wdf, bench_prices=bench_prices).rename("UCR")
    # dcr = compute_down_capture(wdf, bench_prices=bench_prices).rename("DCR")

    # preserve order via explicit concatenation
    out = pd.concat(
        {
            "Return": ret, "Vol": vol, "Sharpe": shrp, "Sortino": sort,
            "VaR95": var95, "ES95": es95, "MDD_pct": mdd, "UlcerIndex": ui,
            "AvgDD": add, "MaxDD_Dur": mdd_d, "AvgDD_Dur": add_d,
            "Calmar": cal, "CSI": csi, "CII": cii,
            # benchmarks appended in fixed order
            "Beta": beta, "R2": r2, "TE": te, "IR": ir,
            "Alpha": alpha, "Treynor": trey,
        },
        axis=0
    )
    out.index.names = ["metric", "ticker"]
    return out


def build_panel(prices: pd.DataFrame, bench_prices: pd.Series) -> pd.DataFrame:
    panel = rolling_apply(
        prices,
        func=lambda w: window_metrics(w, bench_prices=bench_prices),
        window=WINDOW,
        step=STEP,
        min_valid_frac=MIN_VALID_FRAC,
        drop_constant=True,
    )
    return panel


def orient_panel(panel: pd.DataFrame) -> pd.DataFrame:
    mask = panel.columns.get_level_values(0).isin(BAD_METRICS)
    oriented = panel.copy()
    oriented.loc[:, mask] = -oriented.loc[:, mask]
    return oriented


def sanitize_distance(D: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Make the distance matrix usable for clustering/embedding by
    iteratively removing metrics that introduce NaNs, keeping the
    largest all-finite submatrix.
    Returns (clean_D, removed_metric_names).
    """
    D = D.copy()
    # enforce symmetry and zero diagonal
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D.values, 0.0)

    removed: list[str] = []
    # drop nothing if already finite
    while np.isnan(D.values).any() and D.shape[0] > 2:
        nan_counts = pd.Series(np.isnan(D.values).sum(axis=1), index=D.index)
        worst = nan_counts.sort_values(ascending=False).index[0]
        removed.append(worst)
        D = D.drop(index=worst, columns=worst)
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D.values, 0.0)

    D = D.clip(lower=0.0)
    return D, removed



def main():
    parser = argparse.ArgumentParser(description="Rolling metrics → corr/dist → HAC → embed → plot")
    parser.add_argument("--embed", choices=["mds", "pca"], default="mds", help="Embedding method for distances")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2, help="Embedding dimensionality")
    args = parser.parse_args()

    prices, _meta = load_prices(DATA_PATH)
    validate_prices(prices)

    # fetch benchmark once (aligned later inside metric functions)
    bench_prices = get_benchmark_prices(prices)

    panel = build_panel(prices, bench_prices)  # rows: window_end; cols: (metric, ticker)
    panel = orient_panel(panel)  # higher = better orientation

    R, D = compute_corr_and_distance(panel)  # Fisher-z avg Spearman -> R, then distance D
    # sanitize distance matrix (iterative pruning)
    D, removed = sanitize_distance(D)
    if removed:
        print("[info] dropped metrics due to NaNs:", list(removed))
    if D.shape[0] < 2:
        raise RuntimeError("Distance matrix degenerated after sanitization; check data/metrics coverage.")

    # guard cluster count
    n_clusters = min(N_CLUSTERS, D.shape[0])
    labels, _ = agglomerative_cluster(D, n_clusters=n_clusters, linkage=LINKAGE)


    if args.embed == "mds":
        title = f"Metrics: HAC + MDS ({args.dim}D)"
        coords, fig, ax = embed_and_plot(D, labels, title=title, n_components=args.dim, random_state=RANDOM_STATE)
    else:
        title = f"Metrics: HAC + PCA ({args.dim}D)"
        coords = pca_embed(D, n_components=args.dim)
        fig, ax = plot_clusters_mds(coords, labels, title=title)

    plt.show()


if __name__ == "__main__":
    main()
