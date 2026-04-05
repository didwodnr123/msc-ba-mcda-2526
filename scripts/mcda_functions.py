"""
mcda_functions.py
Pure utility functions for MCDA analysis (normalisation, weighting, aggregation).
"""
import numpy as np
import pandas as pd


# ── Normalisation ──────────────────────────────────────────────────────────────

def minmax_norm(df_raw, criteria, direction):
    """Min-Max normalisation. Maps each criterion to [0, 1].
    Non-beneficial criteria are inverted so higher = better always.
    Handles negative values by construction.
    """
    norm = pd.DataFrame(index=df_raw.index)
    for col in criteria:
        lo, hi = df_raw[col].min(), df_raw[col].max()
        if direction[col] == 'beneficial':
            norm[col] = (df_raw[col] - lo) / (hi - lo)
        else:
            norm[col] = (hi - df_raw[col]) / (hi - lo)
    return norm


def rank_norm(df_raw, criteria, direction):
    """Rank-based normalisation. Scores in [1/n, 1].
    Beneficial: ascending rank / n   (largest raw → score 1)
    Non-beneficial: descending rank / n  (smallest raw → score 1)
    Ordinal only; handles negatives without issue.
    """
    n = len(df_raw)
    norm = pd.DataFrame(index=df_raw.index)
    for col in criteria:
        s = pd.Series(df_raw[col].values.astype(float))
        if direction[col] == 'non-beneficial':
            norm[col] = s.rank(ascending=False, method='average').values / n
        else:
            norm[col] = s.rank(ascending=True, method='average').values / n
    return norm


# ── Weighting ──────────────────────────────────────────────────────────────────

def entropy_weights(norm_df, criteria):
    """Shannon entropy weights (data-driven).
    High variability across banks → high weight.
    """
    eps = 1e-10
    m = len(norm_df)

    p = pd.DataFrame(index=norm_df.index)
    for col in criteria:
        col_sum = norm_df[col].sum()
        p[col] = norm_df[col] / col_sum if col_sum != 0 else 1 / m

    E = {}
    for col in criteria:
        pj = p[col].values.clip(eps, None)
        E[col] = -1 / np.log(m) * np.sum(pj * np.log(pj + eps))

    d = {col: 1 - E[col] for col in criteria}
    d_total = sum(d.values())
    return {col: d[col] / d_total for col in criteria}, E


def critic_weights(norm_df, criteria):
    """CRITIC weights (data-driven).
    Combines standard deviation and inter-criterion conflict (1 - correlation).
    """
    norm_df_ = pd.DataFrame(norm_df, columns=criteria)
    sigma = norm_df_.std()
    corr_mat = norm_df_.corr()

    C = {}
    for col in criteria:
        sum_conflict = sum(1 - corr_mat[col][other] for other in criteria if other != col)
        C[col] = sigma[col] * sum_conflict

    C_total = sum(C.values())
    return {col: C[col] / C_total for col in criteria}, C


def ahp_weights(A, criteria_order):
    """AHP weights from a pairwise comparison matrix (Saaty scale).
    Returns weights dict, lambda_max, CI, CR.
    """
    RI_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}
    n = len(criteria_order)

    col_sums = A.sum(axis=0)
    A_norm = A / col_sums
    priority = A_norm.mean(axis=1)

    Aw = A @ priority
    lambda_max = (Aw / priority).mean()
    CI = (lambda_max - n) / (n - 1)
    CR = CI / RI_table[n]

    weights = dict(zip(criteria_order, priority))
    return weights, lambda_max, CI, CR


# ── Aggregation ────────────────────────────────────────────────────────────────

def wsm_scores(norm_df, weights, criteria):
    """Weighted Sum Model: S_i = sum_j(w_j * r_ij)"""
    w = np.array([weights[c] for c in criteria])
    return norm_df[criteria].values @ w


def wpm_scores(norm_df, weights, criteria, eps=1e-6):
    """Weighted Product Model: S_i = prod_j(r_ij ^ w_j).
    Scores floored at eps to avoid zero-product collapse.
    """
    w = np.array([weights[c] for c in criteria])
    R = norm_df[criteria].values.clip(eps, None)
    return np.prod(R ** w, axis=1)
