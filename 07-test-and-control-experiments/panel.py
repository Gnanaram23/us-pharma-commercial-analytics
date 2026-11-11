"""Build a state x quarter panel of GLP-1 use in Medicaid from SDUD extracts.

Rates are prescriptions per 10,000 Medicaid prescriptions of any drug in the
same state and quarter. Using all prescriptions as the denominator corrects for
state size and for the last quarter being incomplete when it is first
published, without needing separate enrolment data.

Groups:
  AOM  anti-obesity medicines: Wegovy, Zepbound, Saxenda
  DM   GLP-1 medicines labelled for type 2 diabetes: Ozempic, Mounjaro, Rybelsus,
       Trulicity, Victoza / liraglutide, Bydureon / Byetta / exenatide

SDUD suppresses counts under 11, and suppressed rows are dropped here, so small
states and early quarters are slightly undercounted.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

AOM = ("WEGOVY", "ZEPBOUND", "SAXENDA")


def load(raw: Path) -> pd.DataFrame:
    g = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(raw / "sdud_glp1_*.csv.gz")))], ignore_index=True)
    t = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(raw / "sdud_totals_*.csv.gz")))], ignore_index=True)
    g = g[g["suppression_used"].astype(str).str.lower() != "true"].copy()
    name = g["product_name"].str.upper().str.strip()
    g["group"] = np.where(name.str.startswith(AOM), "AOM", "DM")
    g["brand"] = name.str.split().str[0].str.replace("LIRAGLUTID", "LIRAGLUTIDE")
    num = g.groupby(["state", "year", "quarter", "group"]).agg(
        rx=("number_of_prescriptions", "sum"), spend=("total_amount_reimbursed", "sum")).unstack("group", fill_value=0)
    num.columns = [f"{a}_{b}" for a, b in num.columns]
    den = t.groupby(["state", "year", "quarter"])["prescriptions"].sum().rename("all_rx")
    p = num.join(den, how="right").fillna(0).reset_index()
    p = p[p["all_rx"] > 0]
    p["period"] = pd.PeriodIndex.from_fields(year=p["year"], quarter=p["quarter"], freq="Q")
    p["aom"] = p["rx_AOM"] / p["all_rx"] * 1e4
    p["dm"] = p["rx_DM"] / p["all_rx"] * 1e4
    return p.sort_values(["state", "period"]).reset_index(drop=True)


def wide(p: pd.DataFrame, col: str) -> pd.DataFrame:
    """States in columns, quarters in rows."""
    return p.pivot(index="period", columns="state", values=col).sort_index()


def synth_weights(y_treated: np.ndarray, Y_donors: np.ndarray) -> np.ndarray:
    """Synthetic control: non-negative donor weights that sum to 1 and best
    reproduce the treated unit's pre-period path (Abadie, Diamond & Hainmueller)."""
    from scipy.optimize import minimize

    k = Y_donors.shape[1]
    res = minimize(lambda w: np.sum((y_treated - Y_donors @ w) ** 2), np.full(k, 1 / k), method="SLSQP",
                   bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x
