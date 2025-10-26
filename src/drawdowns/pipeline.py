# pipeline.py
# End-to-end: load -> rolling metrics -> Spearman/Fisher-z -> HAC -> MDS -> show plot.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_processing import load_prices, validate_prices
from rolling_window import rolling_apply
from calculate_metrics import (
    compute_return, compute_volatility, compute_sharpe, compute_sortino,
    compute_var, compute_es, compute_mdd, compute_ulcer_index,
    compute_avg_drawdown, compute_max_dd_duration, compute_avg_dd_duration,
    compute_calmar, compute_csi, compute_cii,
)
from correlation import compute_corr_and_distance
from clustering import agglomerative_cluster
from visualization import embed_and_plot

# --------- globals (edit as needed) ---------
DATA_PATH      = "data/company_prices.csv"
WINDOW         = 63            # primary lens (~1 quarter); change to 42/126 for sensitivity
STEP           = 1
MIN_VALID_FRAC = 0.9
N_CLUSTERS     = 4
LINKAGE        = "average"     # {"average","complete","single"} for precomputed distances
RANDOM_STATE   = 0
# --------------------------------------------

BAD_METRICS = {
    "Vol", "VaR95", "ES95", "MDD_pct", "UlcerIndex", "AvgDD", "MaxDD_Dur", "AvgDD_Dur", "CII"
}

def window_metrics(wdf: pd.DataFrame) -> pd.Series:
    ret   = compute_return(wdf, log=True).rename("Return")
    vol   = compute_volatility(wdf, log=True, annualize=True).rename("Vol")
    shrp  = compute_sharpe(wdf, risk_free=0.0, log=True, annualize=True).rename("Sharpe")
    sort  = compute_sortino(wdf, target=0.0, log=True, annualize=True).rename("Sortino")
    var95 = compute_var(wdf, alpha=0.95, log=True).rename("VaR95")
    es95  = compute_es(wdf,  alpha=0.95, log=True).rename("ES95")

    mdd   = compute_mdd(wdf).rename("MDD_pct")
    ui    = compute_ulcer_index(wdf).rename("UlcerIndex")
    add   = compute_avg_drawdown(wdf).rename("AvgDD")
    mdd_d = compute_max_dd_duration(wdf).rename("MaxDD_Dur")
    add_d = compute_avg_dd_duration(wdf).rename("AvgDD_Dur")
    cal   = compute_calmar(wdf).rename("Calmar")
    csi   = compute_csi(wdf).rename("CSI")
    cii   = compute_cii(wdf).rename("CII")
    # Preserve metric order by concatenating in the desired sequence
    out = pd.concat(
        {
            "Return": ret, "Vol": vol, "Sharpe": shrp, "Sortino": sort,
            "VaR95": var95, "ES95": es95, "MDD_pct": mdd, "UlcerIndex": ui,
            "AvgDD": add, "MaxDD_Dur": mdd_d, "AvgDD_Dur": add_d,
            "Calmar": cal, "CSI": csi, "CII": cii,
        },
        axis=0
    )
    out.index.names = ["metric", "ticker"]
    return out

def build_panel(prices: pd.DataFrame) -> pd.DataFrame:
    panel = rolling_apply(
        prices,
        func=window_metrics,
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

def main():
    prices, _meta = load_prices(DATA_PATH)
    validate_prices(prices)
    print('!')
    panel = build_panel(prices)                 # rows: window_end; cols: (metric, ticker)
    panel = orient_panel(panel)                 # higher = better orientation
    print('!!')

    R, D = compute_corr_and_distance(panel)     # Fisher-z avg Spearman -> R, then distance D
    print('!!!')
    labels, _ = agglomerative_cluster(D, n_clusters=N_CLUSTERS, linkage=LINKAGE)
    coords, fig, ax = embed_and_plot(D, labels, title="Metrics: HAC + MDS", random_state=RANDOM_STATE)

    plt.show()

if __name__ == "__main__":
    main()
