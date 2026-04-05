"""
final.py — MCDA Analysis: UK Banks 2024

Usage:
  python scripts/final.py
  python scripts/final.py --norm rank --weights entropy
  python scripts/final.py --norm minmax --weights ahp --top 20
  python scripts/final.py --norm minmax --weights pa --output results.csv

Arguments:
  --norm     {minmax, rank}                        Normalisation method      (default: minmax)
  --weights  {equal, pa, ahp, entropy, critic}     Weighting method          (default: pa)
  --top      N                                     Show top N banks          (default: 10)
  --output   PATH                                  Save full ranking to PATH (optional CSV)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

from mcda_functions import (
    minmax_norm, rank_norm,
    entropy_weights, critic_weights, ahp_weights,
    wsm_scores, wpm_scores,
)

# ── Constants ─────────────────────────────────────────────────────────────────
CRITERIA = [
    "INTEREST_EXPENSE", "OPERATING_EXPENSE",
    "INTEREST_INCOME", "NON_INTEREST_INCOME", "ROAE",
]
DIRECTION = {
    "INTEREST_EXPENSE":    "non-beneficial",
    "OPERATING_EXPENSE":   "non-beneficial",
    "INTEREST_INCOME":     "beneficial",
    "NON_INTEREST_INCOME": "beneficial",
    "ROAE":                "beneficial",
}
LABELS = {
    "INTEREST_EXPENSE":    "Interest Expense",
    "OPERATING_EXPENSE":   "Operating Expense",
    "INTEREST_INCOME":     "Interest Income",
    "NON_INTEREST_INCOME": "Non-interest Income",
    "ROAE":                "ROAE (%)",
}

# Point Allocation weights (stakeholder-justified)
PA_WEIGHTS = {
    "INTEREST_EXPENSE":    0.15,
    "OPERATING_EXPENSE":   0.10,
    "INTEREST_INCOME":     0.25,
    "NON_INTEREST_INCOME": 0.20,
    "ROAE":                0.30,
}

# AHP pairwise comparison matrix (Saaty scale)
AHP_CRITERIA = [
    "ROAE", "INTEREST_INCOME", "NON_INTEREST_INCOME",
    "INTEREST_EXPENSE", "OPERATING_EXPENSE",
]
AHP_MATRIX = np.array([
    [1,    2,    2,    3,    3  ],   # ROAE
    [1/2,  1,    2,    2,    3  ],   # Interest Income
    [1/2,  1/2,  1,    2,    2  ],   # Non-interest Income
    [1/3,  1/2,  1/2,  1,    2  ],   # Interest Expense
    [1/3,  1/3,  1/2,  1/2,  1  ],   # Operating Expense
])


# ── Pipeline steps ────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = "data/MCDA_SPGlobal_SNL_UK Banks_2024.xlsx"
    df = pd.read_excel(path)
    print(f"Loaded: {len(df)} banks × {len(CRITERIA)} criteria")
    missing = df[CRITERIA].isnull().sum()
    if missing.any():
        print("WARNING — missing values:")
        print(missing[missing > 0].to_string())
    return df


def normalise(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "minmax":
        return minmax_norm(df, CRITERIA, DIRECTION)
    elif method == "rank":
        return rank_norm(df, CRITERIA, DIRECTION)
    raise ValueError(f"Unknown normalisation method: {method!r}")


def get_weights(norm_df: pd.DataFrame, method: str) -> dict:
    if method == "equal":
        return {c: 1 / len(CRITERIA) for c in CRITERIA}
    elif method == "pa":
        return PA_WEIGHTS
    elif method == "ahp":
        w, lmax, ci, cr = ahp_weights(AHP_MATRIX, AHP_CRITERIA)
        status = "consistent (CR < 0.10)" if cr < 0.10 else "INCONSISTENT (CR >= 0.10)"
        print(f"AHP: λ_max={lmax:.4f}  CI={ci:.4f}  CR={cr:.4f}  [{status}]")
        return w
    elif method == "entropy":
        w, _ = entropy_weights(norm_df, CRITERIA)
        return w
    elif method == "critic":
        w, _ = critic_weights(norm_df, CRITERIA)
        return w
    raise ValueError(f"Unknown weighting method: {method!r}")


def rank_agreement(wsm_ranks: np.ndarray, wpm_ranks: np.ndarray) -> None:
    rho, _ = stats.spearmanr(wsm_ranks, wpm_ranks)
    abs_diff = np.abs(wsm_ranks - wpm_ranks)
    print(f"WSM vs WPM agreement:")
    print(f"  Spearman ρ          : {rho:.4f}")
    print(f"  Mean |rank diff|    : {abs_diff.mean():.2f}")
    print(f"  Max  |rank diff|    : {abs_diff.max()}")
    print(f"  Banks with |diff|>10: {(abs_diff > 10).sum()}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MCDA ranking of UK banks (WSM + WPM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--norm", choices=["minmax", "rank"], default="minmax",
        help="Normalisation method",
    )
    parser.add_argument(
        "--weights", choices=["equal", "pa", "ahp", "entropy", "critic"], default="pa",
        help="Weighting method",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of top banks to display",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save full ranking as CSV to this path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load
    df = load_data()

    # 2. Normalise
    norm_df = normalise(df, args.norm)

    # 3. Weights
    print(f"\nNorm={args.norm!r}  Weights={args.weights!r}")
    weights = get_weights(norm_df, args.weights)
    print("Weights:")
    for c in CRITERIA:
        print(f"  {LABELS[c]:<30} {weights[c]:.4f}")
    print(f"  {'Sum':<30} {sum(weights.values()):.4f}")

    # 4. Aggregate
    wsm = wsm_scores(norm_df, weights, CRITERIA)
    wpm = wpm_scores(norm_df, weights, CRITERIA)

    wsm_rank = pd.Series(wsm).rank(ascending=False).astype(int).values
    wpm_rank = pd.Series(wpm).rank(ascending=False).astype(int).values

    ranking = pd.DataFrame({
        "Bank":      df["BANK_NAME"].values,
        "WSM_score": wsm.round(6),
        "WSM_rank":  wsm_rank,
        "WPM_score": wpm.round(6),
        "WPM_rank":  wpm_rank,
        "Rank_diff": wpm_rank - wsm_rank,
    })

    # 5. Print top-N
    top = ranking.nsmallest(args.top, "WSM_rank").reset_index(drop=True)
    top.index = top.index + 1
    print(f"\nTop {args.top} banks — WSM ranking")
    print(top[["Bank", "WSM_rank", "WSM_score", "WPM_rank", "WPM_score", "Rank_diff"]].to_string())

    # 6. Agreement stats
    print()
    rank_agreement(wsm_rank, wpm_rank)

    # 7. Optional CSV export
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ranking.sort_values("WSM_rank").to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
