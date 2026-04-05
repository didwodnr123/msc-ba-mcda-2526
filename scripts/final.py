"""
final.py  —  MCDA Analysis: UK Banks 2024
Run from project root or from scripts/ (paths are resolved automatically).

  cd scripts && python final.py
  # or
  python scripts/final.py
"""
import os
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── Resolve project root so all relative paths (data/, outputs/) work ─────────
ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'scripts'))

from mcda_functions import (
    minmax_norm, rank_norm,
    entropy_weights, critic_weights, ahp_weights,
    wsm_scores, wpm_scores,
)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)

os.makedirs('outputs/sec3', exist_ok=True)
os.makedirs('outputs/sec4', exist_ok=True)
os.makedirs('outputs/sec5', exist_ok=True)
os.makedirs('outputs/sec6', exist_ok=True)
os.makedirs('outputs/sec7', exist_ok=True)
os.makedirs('outputs/sec8', exist_ok=True)

CRITERIA = [
    'INTEREST_EXPENSE', 'OPERATING_EXPENSE',
    'INTEREST_INCOME', 'NON_INTEREST_INCOME', 'ROAE',
]
DIRECTION = {
    'INTEREST_EXPENSE':    'non-beneficial',
    'OPERATING_EXPENSE':   'non-beneficial',
    'INTEREST_INCOME':     'beneficial',
    'NON_INTEREST_INCOME': 'beneficial',
    'ROAE':                'beneficial',
}
LABELS = {
    'INTEREST_EXPENSE':    'Interest Expense',
    'OPERATING_EXPENSE':   'Operating Expense',
    'INTEREST_INCOME':     'Interest Income',
    'NON_INTEREST_INCOME': 'Non-interest Income',
    'ROAE':                'ROAE (%)',
}
print("Setup complete.")


# ── Section 3: Data Loading & EDA ─────────────────────────────────────────────

# 3.1 Load dataset
df = pd.read_excel('data/MCDA_SPGlobal_SNL_UK Banks_2024.xlsx')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# 3.2 Overview
overview_cols = ['BANK_NAME', 'BANK_ID', 'GEOGRAPHY'] + CRITERIA
print(df[overview_cols].head(10).to_string())

# 3.3 Missing values
missing = df[CRITERIA].isnull().sum()
print("\nMissing values per criterion:")
print(missing.to_frame('missing'))
print(f"\nTotal banks: {len(df)}")

# 3.4 Descriptive statistics
desc = df[CRITERIA].describe().T
desc['skewness'] = df[CRITERIA].skew()
desc['kurtosis'] = df[CRITERIA].kurt()
desc.columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis']
print("\nDescriptive Statistics:")
print(desc.round(2))

# 3.5 Distribution histograms
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(CRITERIA):
    ax = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=30, edgecolor='white', linewidth=0.5, color='steelblue', alpha=0.85)
    ax.set_title(LABELS[col], fontsize=12, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    sk = data.skew()
    ax.text(0.97, 0.95, f'Skewness: {sk:.2f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
axes[5].set_visible(False)
fig.suptitle('Figure 3.1: Distribution of Criteria Values (91 UK Banks, 2024)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/sec3/distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/sec3/distributions.png")

# 3.6 Box plots (with outlier labels)
fig, axes = plt.subplots(1, 5, figsize=(20, 6))
for i, col in enumerate(CRITERIA):
    ax = axes[i]
    data = df[col].dropna()
    ax.boxplot(data, patch_artist=True,
               boxprops=dict(facecolor='lightsteelblue'),
               medianprops=dict(color='darkred', linewidth=2),
               flierprops=dict(marker='o', markersize=4, alpha=0.6))
    ax.set_title(LABELS[col], fontsize=10, fontweight='bold')
    ax.set_xticklabels([])
    q75 = data.quantile(0.75)
    iqr = data.quantile(0.75) - data.quantile(0.25)
    outliers = df.loc[df[col] > q75 + 1.5 * iqr, ['BANK_NAME', col]].nlargest(2, col)
    for _, row in outliers.iterrows():
        name = row['BANK_NAME'].split(' ')[0]
        ax.annotate(name, xy=(1, row[col]), xytext=(1.15, row[col]),
                    fontsize=7, color='darkred',
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8))
fig.suptitle('Figure 3.2: Box Plots of Criteria Values — Outlier Banks Annotated',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/sec3/boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/sec3/boxplots.png")

# 3.7 Pearson correlation heatmap
corr = df[CRITERIA].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=-1, vmax=1, center=0,
            linewidths=0.5, linecolor='white',
            xticklabels=[LABELS[c] for c in CRITERIA],
            yticklabels=[LABELS[c] for c in CRITERIA],
            ax=ax, square=True)
ax.set_title('Figure 3.3: Pearson Correlation Matrix of Criteria', fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('outputs/sec3/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/sec3/correlation_heatmap.png")
print("\nCorrelation matrix:")
print(corr.round(3))

# 3.8 Pareto frontier analysis
data_pareto = df[CRITERIA].copy()
for col in CRITERIA:
    if DIRECTION[col] == 'non-beneficial':
        data_pareto[col] = -data_pareto[col]
M = data_pareto.values
n_banks = len(M)
dominated = np.zeros(n_banks, dtype=bool)
for i in range(n_banks):
    for j in range(n_banks):
        if i == j:
            continue
        if np.all(M[j] >= M[i]) and np.any(M[j] > M[i]):
            dominated[i] = True
            break
print(f"\nPareto-efficient banks : {np.sum(~dominated)} / {n_banks}")
print(f"Pareto-dominated banks : {np.sum(dominated)} / {n_banks}")
if np.sum(dominated) > 0:
    print("\nDominated banks:")
    print(df.loc[dominated, ['BANK_NAME'] + CRITERIA])


# ── Section 4: Performance Representation ─────────────────────────────────────

# 4.2 Min-Max normalisation
norm_mm = minmax_norm(df, CRITERIA, DIRECTION)
print("\nMin-Max normalised data shape:", norm_mm.shape)
print(norm_mm.describe().round(3))

# 4.3 Verify range
print("\nMin-Max normalised range [should be 0-1 for all criteria]:")
print(norm_mm.agg(['min', 'max']).round(4))

# 4.4 Rank-based normalisation
norm_rank = rank_norm(df, CRITERIA, DIRECTION)
print("\nRank-based normalised range [all criteria should be in (1/n, 1]]:")
print(norm_rank.agg(['min', 'max']).round(4))
print(f"All values positive: {(norm_rank > 0).all().all()}")
print(f"All values ≤ 1:      {(norm_rank <= 1).all().all()}")
neg_ni_idx = df[df['NON_INTEREST_INCOME'] < 0].index
print("\nBanks with negative NON_INTEREST_INCOME — rank-based scores (no out-of-bounds):")
print(
    df.loc[neg_ni_idx, ['BANK_NAME', 'NON_INTEREST_INCOME']]
    .assign(rank_score=norm_rank.loc[neg_ni_idx, 'NON_INTEREST_INCOME'].values)
    .reset_index(drop=True)
)

# 4.5 Side-by-side box plots: Min-Max vs Rank-based
fig, axes = plt.subplots(2, 5, figsize=(15, 10), sharey=False)
for i, col in enumerate(CRITERIA):
    ax_mm = axes[0][i]
    ax_mm.boxplot(norm_mm[col], patch_artist=True,
                  boxprops=dict(facecolor='lightsteelblue'),
                  medianprops=dict(color='darkred', linewidth=2))
    ax_mm.set_title(LABELS[col], fontsize=9, fontweight='bold')
    if i == 0:
        ax_mm.set_ylabel('Min-Max', fontsize=10)
    ax_rk = axes[1][i]
    ax_rk.boxplot(norm_rank[col], patch_artist=True,
                  boxprops=dict(facecolor='lightyellow'),
                  medianprops=dict(color='darkblue', linewidth=2))
    if i == 0:
        ax_rk.set_ylabel('Rank-based', fontsize=10)
fig.suptitle('Figure 4.1: Normalised Score Distributions — Min-Max (top) vs Rank-based (bottom)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/sec4/norm_comparison_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/sec4/norm_comparison_boxplots.png")

# 4.6 Scatter: Min-Max vs Rank-based per criterion
fig, axes = plt.subplots(1, 5, figsize=(10, 4))
for i, col in enumerate(CRITERIA):
    ax = axes[i]
    ax.scatter(norm_mm[col], norm_rank[col], alpha=0.5, s=25, color='steelblue')
    ax.set_xlabel('Min-Max', fontsize=9)
    ax.set_ylabel('Rank-based', fontsize=9)
    ax.set_title(LABELS[col], fontsize=9, fontweight='bold')
    rho, _ = stats.spearmanr(norm_mm[col], norm_rank[col])
    ax.text(0.05, 0.93, f'ρ = {rho:.3f}', transform=ax.transAxes,
            fontsize=9, bbox=dict(facecolor='lightyellow', alpha=0.8))
fig.suptitle('Figure 4.3: Scatter Plots — Min-Max vs Rank-based Normalised Scores per Criterion',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/sec4/norm_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/sec4/norm_scatter.png")

# 4.7 Rank comparison under equal weights: Min-Max vs Rank-based
score_mm   = norm_mm[CRITERIA].mul([1/5] * 5).sum(axis=1)
score_rank = norm_rank[CRITERIA].mul([1/5] * 5).sum(axis=1)
rank_mm  = score_mm.rank(ascending=False).astype(int)
rank_rk  = score_rank.rank(ascending=False).astype(int)
rank_df = pd.DataFrame({
    'Bank':       df['BANK_NAME'].values,
    'Rank (MM)':  rank_mm.values,
    'Score (MM)': score_mm.round(4).values,
    'Rank (RB)':  rank_rk.values,
    'Score (RB)': score_rank.round(4).values,
})
rank_df['Rank Shift'] = (rank_df['Rank (RB)'] - rank_df['Rank (MM)']).abs()
rho_overall, _ = stats.spearmanr(rank_mm, rank_rk)
print(f"\nSpearman rank correlation (Min-Max vs Rank-based, equal weights): {rho_overall:.4f}")
print(f"Mean absolute rank shift:   {rank_df['Rank Shift'].mean():.2f}")
print(f"Max absolute rank shift:    {rank_df['Rank Shift'].max()}")
print("Banks with rank shift > 5:")
large_shift = rank_df[rank_df['Rank Shift'] > 5].sort_values('Rank Shift', ascending=False)
print(large_shift[['Bank', 'Rank (MM)', 'Rank (RB)', 'Rank Shift']].reset_index(drop=True))

# Top-10 and Bottom-10 comparison
print("\nTop 10 banks under Min-Max normalisation (equal weights):")
print(rank_df.nsmallest(10, 'Rank (MM)')[['Bank', 'Rank (MM)', 'Rank (RB)']].reset_index(drop=True))
print("\nTop 10 banks under Rank-based normalisation (equal weights):")
print(rank_df.nsmallest(10, 'Rank (RB)')[['Bank', 'Rank (RB)', 'Rank (MM)']].reset_index(drop=True))
print("\nBottom 10 banks under Min-Max normalisation (equal weights):")
print(rank_df.nlargest(10, 'Rank (MM)')[['Bank', 'Rank (MM)', 'Rank (RB)']].reset_index(drop=True))


# ── Section 5: Weighting ───────────────────────────────────────────────────────

# 5.3 Equal weights
equal_w = {c: 1 / len(CRITERIA) for c in CRITERIA}
print("\nEqual Weights:")
for c, w in equal_w.items():
    print(f"  {LABELS[c]:<30} {w:.4f}")

# 5.4 Point Allocation weights
pa_w = {
    'INTEREST_EXPENSE':    0.15,
    'OPERATING_EXPENSE':   0.10,
    'INTEREST_INCOME':     0.25,
    'NON_INTEREST_INCOME': 0.20,
    'ROAE':                0.30,
}
assert abs(sum(pa_w.values()) - 1.0) < 1e-9, "Weights must sum to 1"
print("\nPoint Allocation Stakeholder-Justified Weights:")
for c, w in pa_w.items():
    print(f"  {LABELS[c]:<30} {w:.4f}")

# 5.5 AHP weights via pairwise comparison matrix (Saaty scale)
AHP_CRITERIA = ['ROAE', 'INTEREST_INCOME', 'NON_INTEREST_INCOME', 'INTEREST_EXPENSE', 'OPERATING_EXPENSE']
A = np.array([
    [1,    2,    2,    3,    3  ],   # ROAE
    [1/2,  1,    2,    2,    3  ],   # Interest Income
    [1/2,  1/2,  1,    2,    2  ],   # Non-interest Income
    [1/3,  1/2,  1/2,  1,    2  ],   # Interest Expense
    [1/3,  1/3,  1/2,  1/2,  1  ],   # Operating Expense
])
ahp_eigen_w, lambda_max, CI, CR = ahp_weights(A, AHP_CRITERIA)
print("\nAHP Weights (from pairwise comparison matrix):")
for c, w in ahp_eigen_w.items():
    print(f"  {LABELS[c]:<30} {w:.4f}")
print(f"\n  lambda_max = {lambda_max:.4f}")
print(f"  CI         = {CI:.4f}")
print(f"  CR         = {CR:.4f}  {'✓ Consistent (CR < 0.10)' if CR < 0.10 else '✗ Inconsistent (CR >= 0.10)'}")

# 5.6 Entropy weights
entropy_w, E = entropy_weights(norm_mm, CRITERIA)
print("\nEntropy Weights:")
for c in CRITERIA:
    d = 1 - E[c]
    print(f"  {LABELS[c]:<30} E={E[c]:.4f}  d={d:.4f}  w={entropy_w[c]:.4f}")
print(f"Sum of entropy weights: {sum(entropy_w.values()):.6f}")

# 5.7 CRITIC weights
critic_w, C = critic_weights(norm_mm, CRITERIA)
print("\nCRITIC Weights (based on Min-Max normalised data):")
for c in CRITERIA:
    print(f"  {LABELS[c]:<30} C={C[c]:.4e}  w={critic_w[c]:.4f}")
print(f"Sum of CRITIC weights: {sum(critic_w.values()):.6f}")

# 5.8 Weight comparison table
weight_df = pd.DataFrame({
    'Equal':            equal_w,
    'Point Allocation': pa_w,
    'AHP':              ahp_eigen_w,
    'Entropy':          entropy_w,
    'CRITIC':           critic_w,
})
weight_df.index = [LABELS[c] for c in CRITERIA]
weight_df.loc['SUM'] = weight_df.sum()
print("\nWeight Comparison Table:")
print(weight_df.round(4))

# 5.9 Weight comparison bar chart
weight_plot = pd.DataFrame({
    'Equal':            equal_w,
    'Point Allocation': pa_w,
    'AHP':              ahp_eigen_w,
    'Entropy':          entropy_w,
    'CRITIC':           critic_w,
}, index=CRITERIA)
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(CRITERIA))
width = 0.15
colors = ['#4878CF', '#6ACC65', '#E8963A', '#D65F5F', '#B47CC7']
for i, (col, color) in enumerate(zip(weight_plot.columns, colors)):
    ax.bar(x + i * width, weight_plot[col], width, label=col, color=color, alpha=0.85, edgecolor='white')
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels([LABELS[c] for c in CRITERIA], rotation=20, ha='right', fontsize=10)
ax.set_ylabel('Weight')
ax.set_ylim(0, 0.65)
ax.axhline(0.2, color='grey', linestyle='--', linewidth=0.8, label='Equal weight (0.20)')
ax.legend(loc='upper right', fontsize=9)
ax.set_title('Figure 5.1: Criterion Weights under Five Weighting Approaches', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/sec5/weight_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/sec5/weight_comparison.png")


# ── Section 6: WSM & WPM Rankings ─────────────────────────────────────────────

WEIGHTS = {
    'Equal':            equal_w,
    'Point Allocation': pa_w,
    'AHP':              ahp_eigen_w,
    'Entropy':          entropy_w,
    'CRITIC':           critic_w,
}

results = {}
for wname, w in WEIGHTS.items():
    wsm = wsm_scores(norm_mm, w, CRITERIA)
    wpm = wpm_scores(norm_mm, w, CRITERIA)
    results[wname] = {
        'WSM_score': wsm,
        'WPM_score': wpm,
        'WSM_rank':  pd.Series(wsm).rank(ascending=False).astype(int).values,
        'WPM_rank':  pd.Series(wpm).rank(ascending=False).astype(int).values,
    }

# Primary result frame (Point Allocation)
primary = results['Point Allocation']
ranking_df = pd.DataFrame({
    'Bank':      df['BANK_NAME'].values,
    'WSM_score': primary['WSM_score'].round(6),
    'WSM_rank':  primary['WSM_rank'],
    'WPM_score': primary['WPM_score'].round(6),
    'WPM_rank':  primary['WPM_rank'],
})
ranking_df['Rank_diff'] = ranking_df['WPM_rank'] - ranking_df['WSM_rank']
print('\nPrimary ranking table (Point Allocation weights, Min-Max normalisation):')
print(f'  Banks computed: {len(ranking_df)}')
print(f'  WSM score range: [{primary["WSM_score"].min():.4f}, {primary["WSM_score"].max():.4f}]')
print(f'  WPM score range: [{primary["WPM_score"].min():.6f}, {primary["WPM_score"].max():.6f}]')

# 6.4 Full ranked table — sorted by WSM rank
full_table = ranking_df.sort_values('WSM_rank').reset_index(drop=True)
full_table.index = full_table.index + 1
print('\nFull ranking (Primary: Point Allocation weights, Min-Max normalisation)')
print('Sorted by WSM rank')
print(full_table[['Bank', 'WSM_rank', 'WSM_score', 'WPM_rank', 'WPM_score', 'Rank_diff']])
full_table.to_csv('outputs/sec6/full_ranking_ahp_minmax.csv', index=False)
print('\nSaved: outputs/sec6/full_ranking_ahp_minmax.csv')

# 6.5 Top-20 and Bottom-20 under WSM
top20 = ranking_df.nsmallest(20, 'WSM_rank')[['Bank', 'WSM_rank', 'WSM_score', 'WPM_rank', 'WPM_score', 'Rank_diff']]
bot20 = ranking_df.nlargest(20, 'WSM_rank')[['Bank', 'WSM_rank', 'WSM_score', 'WPM_rank', 'WPM_score', 'Rank_diff']]
print('\n── Top 20 Banks (WSM rank, Point Allocation weights) ──')
print(top20.reset_index(drop=True))
print('\n── Bottom 20 Banks (WSM rank, Point Allocation weights) ──')
print(bot20.sort_values('WSM_rank').reset_index(drop=True))

# 6.5b Top 10 WSM and Top 10 WPM
top10_wsm = ranking_df.nsmallest(10, 'WSM_rank')[['Bank', 'WSM_rank', 'WSM_score']].reset_index(drop=True)
top10_wsm.index = top10_wsm.index + 1
top10_wpm = ranking_df.nsmallest(10, 'WPM_rank')[['Bank', 'WPM_rank', 'WPM_score']].reset_index(drop=True)
top10_wpm.index = top10_wpm.index + 1
print('\nTable 6.1 Top 10 WSM Rankings (Point Allocation weights, Min-Max normalisation)')
print(top10_wsm)
print('\nTable 6.2 Top 10 WPM Rankings (Point Allocation weights, Min-Max normalisation)')
print(top10_wpm)

# 6.6 Figure 6.1: Top-20 WSM vs WPM bar chart
top20_sorted = top20.sort_values('WSM_rank')
banks_short = [b.split(' ')[0] + ' ' + b.split(' ')[1] if len(b.split()) > 1 else b
               for b in top20_sorted['Bank']]
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(top20_sorted))
w = 0.38
ax.bar(x - w/2, top20_sorted['WSM_score'], w,
       label='WSM Score', color='#4878CF', alpha=0.85, edgecolor='white')
ax.bar(x + w/2, top20_sorted['WPM_score'], w,
       label='WPM Score', color='#6ACC65', alpha=0.85, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(banks_short, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Score')
ax.set_title('Figure 6.1: WSM vs WPM Scores — Top 20 Banks (Point Allocation weights, Min-Max normalisation)',
             fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('outputs/sec6/top20_wsm_wpm_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: outputs/sec6/top20_wsm_wpm_bar.png')

# 6.7 Figure 6.2: WSM rank vs WPM rank scatter (all 91 banks)
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(ranking_df['WSM_rank'], ranking_df['WPM_rank'],
           s=40, alpha=0.7, color='steelblue', edgecolors='white', linewidths=0.4)
ax.plot([1, 91], [1, 91], 'r--', linewidth=1, label='WSM = WPM (no rank change)')
for _, row in ranking_df[ranking_df['WSM_rank'] <= 10].iterrows():
    name = row['Bank'].split()[0]
    ax.annotate(name, xy=(row['WSM_rank'], row['WPM_rank']),
                xytext=(row['WSM_rank'] + 1.5, row['WPM_rank'] - 1.5),
                fontsize=7.5, color='darkblue',
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=0.6))
for _, row in ranking_df[ranking_df['WPM_rank'] <= 10].iterrows():
    if row['WSM_rank'] > 10:
        name = row['Bank'].split()[0]
        ax.annotate(name, xy=(row['WSM_rank'], row['WPM_rank']),
                    xytext=(row['WSM_rank'] + 1.5, row['WPM_rank'] + 2),
                    fontsize=7.5, color='darkgreen',
                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=0.6))
ax.set_xlabel('WSM Rank', fontsize=11)
ax.set_ylabel('WPM Rank', fontsize=11)
ax.set_title('Figure 6.2: WSM vs WPM Rank — All 91 Banks (Point Allocation weights)', fontweight='bold')
ax.invert_xaxis()
ax.invert_yaxis()
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/sec6/wsm_vs_wpm_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: outputs/sec6/wsm_vs_wpm_scatter.png')

# 6.8 WSM vs WPM rank agreement statistics
rho_primary, _ = stats.spearmanr(ranking_df['WSM_rank'], ranking_df['WPM_rank'])
abs_diff = ranking_df['Rank_diff'].abs()
print('\nWSM vs WPM rank agreement (Point Allocation weights, Min-Max normalisation)')
print(f'  Spearman rank correlation (ρ): {rho_primary:.4f}')
print(f'  Mean absolute rank difference:  {abs_diff.mean():.2f}')
print(f'  Median absolute rank difference:{abs_diff.median():.1f}')
print(f'  Max absolute rank difference:   {abs_diff.max()}')
print(f'  Banks with |rank shift| > 10:   {(abs_diff > 10).sum()}')
print(f'  Banks with |rank shift| > 20:   {(abs_diff > 20).sum()}')
print('\nTop 10 largest WSM–WPM rank disagreements:')
disagreements = ranking_df.assign(abs_diff=abs_diff).nlargest(10, 'abs_diff')
print(disagreements[['Bank', 'WSM_rank', 'WPM_rank', 'Rank_diff', 'abs_diff']].reset_index(drop=True))
pa_top10 = ranking_df.nsmallest(10, 'WSM_rank')[['Bank', 'WSM_rank', 'WPM_rank', 'Rank_diff']].copy()
pa_top10 = pa_top10.assign(abs_diff=pa_top10['Rank_diff'].abs()).reset_index(drop=True)
pa_top10.index = range(1, len(pa_top10) + 1)
print('\nWSM Top-10 banks (Point Allocation):')
print(pa_top10)

# 6.8b WSM vs WPM rank disagreements — CRITIC weights
crit = results['CRITIC']
crit_df = pd.DataFrame({
    'Bank':     df['BANK_NAME'].values,
    'WSM_rank': crit['WSM_rank'],
    'WPM_rank': crit['WPM_rank'],
})
crit_df['Rank_diff'] = crit_df['WPM_rank'] - crit_df['WSM_rank']
crit_abs = crit_df['Rank_diff'].abs()
rho_crit, _ = stats.spearmanr(crit_df['WSM_rank'], crit_df['WPM_rank'])
print('\nWSM vs WPM rank agreement (CRITIC weights, Min-Max normalisation)')
print(f'  Spearman rank correlation (ρ): {rho_crit:.4f}')
print(f'  Mean absolute rank difference:  {crit_abs.mean():.2f}')
print(f'  Median absolute rank difference:{crit_abs.median():.1f}')
print(f'  Max absolute rank difference:   {crit_abs.max()}')
print(f'  Banks with |rank shift| > 10:   {(crit_abs > 10).sum()}')
print(f'  Banks with |rank shift| > 20:   {(crit_abs > 20).sum()}')
print('\nTop 10 largest WSM–WPM rank disagreements (CRITIC weights):')
crit_disagree = crit_df.assign(abs_diff=crit_abs).nlargest(10, 'abs_diff')
tbl = crit_disagree[['Bank', 'WSM_rank', 'WPM_rank', 'Rank_diff', 'abs_diff']].reset_index(drop=True)
tbl.index = range(1, len(tbl) + 1)
print(tbl)
crit_top10 = crit_df.nsmallest(10, 'WSM_rank')[['Bank', 'WSM_rank', 'WPM_rank', 'Rank_diff']].copy()
crit_top10 = crit_top10.assign(abs_diff=crit_top10['Rank_diff'].abs()).reset_index(drop=True)
crit_top10.index = range(1, len(crit_top10) + 1)
print('\nWSM Top-10 banks (CRITIC weights):')
print(crit_top10)

# 6.9 Figure 6.3: Rank heatmap — WSM and WPM across all weight schemes
wsm_rank_matrix = pd.DataFrame(
    {wname: results[wname]['WSM_rank'] for wname in WEIGHTS},
    index=df['BANK_NAME'].values
)
wpm_rank_matrix = pd.DataFrame(
    {wname: results[wname]['WPM_rank'] for wname in WEIGHTS},
    index=df['BANK_NAME'].values
)
top20_idx = ranking_df.nsmallest(20, 'WSM_rank')['Bank'].values
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
for ax, rmat, title in zip(
        axes,
        [wsm_rank_matrix.loc[top20_idx], wpm_rank_matrix.loc[top20_idx]],
        ['WSM Ranks', 'WPM Ranks']):
    sns.heatmap(rmat, annot=True, fmt='d', cmap='RdYlGn_r',
                vmin=1, vmax=91, linewidths=0.4, linecolor='white',
                ax=ax, cbar_kws={'label': 'Rank (1=best)'})
    ax.set_title(f'Figure 6.3: {title} — Top 20 AHP-WSM Banks\nacross All Weighting Schemes',
                 fontweight='bold', fontsize=10)
    ax.set_xlabel('Weighting Scheme')
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=8)
plt.tight_layout()
plt.savefig('outputs/sec6/rank_heatmap_top20.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: outputs/sec6/rank_heatmap_top20.png')


# ── Section 7: Results — Scores and Rankings ──────────────────────────────────

wsm_score_matrix = pd.DataFrame(
    {wname: results[wname]['WSM_score'].round(4) for wname in WEIGHTS},
    index=df['BANK_NAME'].values
)
wsm_rank_matrix_all = pd.DataFrame(
    {wname: results[wname]['WSM_rank'] for wname in WEIGHTS},
    index=df['BANK_NAME'].values
)
wpm_score_matrix = pd.DataFrame(
    {wname: results[wname]['WPM_score'].round(6) for wname in WEIGHTS},
    index=df['BANK_NAME'].values
)
wpm_rank_matrix_all = pd.DataFrame(
    {wname: results[wname]['WPM_rank'] for wname in WEIGHTS},
    index=df['BANK_NAME'].values
)
pa_wsm_order = wsm_rank_matrix_all['Point Allocation'].nsmallest(20).index

# 7.2 WSM scores — Top 20
print("\nWSM scores — Top 20 banks (sorted by Point Allocation rank) across all weight schemes:")
print(wsm_score_matrix.loc[pa_wsm_order].assign(
    **{'PA Rank': wsm_rank_matrix_all.loc[pa_wsm_order, 'Point Allocation']}
))

# 7.3 WPM scores — Top 20
print("\nWPM scores — Top 20 banks (sorted by Point Allocation WSM rank) across all weight schemes:")
print(wpm_score_matrix.loc[pa_wsm_order].assign(
    **{'PA WSM Rank': wsm_rank_matrix_all.loc[pa_wsm_order, 'Point Allocation'],
       'PA WPM Rank': wpm_rank_matrix_all.loc[pa_wsm_order, 'Point Allocation']}
))

# 7.4 Score distributions
wsm_primary_scores = results['Point Allocation']['WSM_score']
wpm_primary_scores = results['Point Allocation']['WPM_score']
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(wsm_primary_scores, bins=20, edgecolor='black', color='steelblue', alpha=0.8)
axes[0].axvline(wsm_primary_scores.mean(), color='red', linestyle='--',
                label=f'Mean: {wsm_primary_scores.mean():.3f}')
axes[0].axvline(float(np.median(wsm_primary_scores)), color='orange', linestyle=':',
                label=f'Median: {float(np.median(wsm_primary_scores)):.3f}')
axes[0].set_title('Figure 7.1a: WSM Score Distribution\n(Point Allocation weights, Min-Max normalisation)', fontsize=10)
axes[0].set_xlabel('WSM Score')
axes[0].set_ylabel('Number of banks')
axes[0].legend(fontsize=9)
axes[1].hist(wpm_primary_scores, bins=20, edgecolor='black', color='salmon', alpha=0.8)
axes[1].axvline(wpm_primary_scores.mean(), color='red', linestyle='--',
                label=f'Mean: {wpm_primary_scores.mean():.4f}')
axes[1].axvline(float(np.median(wpm_primary_scores)), color='orange', linestyle=':',
                label=f'Median: {float(np.median(wpm_primary_scores)):.4f}')
axes[1].set_title('Figure 7.1b: WPM Score Distribution\n(Point Allocation weights, Min-Max normalisation)', fontsize=10)
axes[1].set_xlabel('WPM Score')
axes[1].set_ylabel('Number of banks')
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/sec7/fig7_1_score_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: outputs/sec7/fig7_1_score_distributions.png')

# 7.5 Score gap analysis
wsm_sorted_scores = pd.Series(wsm_primary_scores).sort_values(ascending=False).reset_index(drop=True)
wpm_sorted_scores = pd.Series(wpm_primary_scores).sort_values(ascending=False).reset_index(drop=True)
gaps_wsm = wsm_sorted_scores.diff(-1).dropna()
gaps_wpm = wpm_sorted_scores.diff(-1).dropna()
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(range(1, len(gaps_wsm) + 1), gaps_wsm.values, color='steelblue', alpha=0.7)
axes[0].axhline(gaps_wsm.mean(), color='red', linestyle='--', label=f'Mean gap: {gaps_wsm.mean():.4f}')
axes[0].set_title('Figure 7.2a: WSM — Score gap between consecutive ranks', fontsize=10)
axes[0].set_xlabel('Rank position')
axes[0].set_ylabel('Score gap to next rank')
axes[0].legend(fontsize=9)
axes[1].bar(range(1, len(gaps_wpm) + 1), gaps_wpm.values, color='salmon', alpha=0.7)
axes[1].axhline(gaps_wpm.mean(), color='red', linestyle='--', label=f'Mean gap: {gaps_wpm.mean():.6f}')
axes[1].set_title('Figure 7.2b: WPM — Score gap between consecutive ranks', fontsize=10)
axes[1].set_xlabel('Rank position')
axes[1].set_ylabel('Score gap to next rank')
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/sec7/fig7_2_score_gaps.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"WSM score gap: top={gaps_wsm.iloc[0]:.4f},  mean={gaps_wsm.mean():.4f},  max={gaps_wsm.max():.4f} at rank {int(gaps_wsm.idxmax())+1}")
print(f"WPM score gap: top={gaps_wpm.iloc[0]:.6f},  mean={gaps_wpm.mean():.6f},  max={gaps_wpm.max():.6f} at rank {int(gaps_wpm.idxmax())+1}")

# 7.6 Banks most favoured/penalised by weighting philosophy
equal_wsm_ranks = pd.Series(results['Equal']['WSM_rank'], index=df['BANK_NAME'].values)
for wname in ['Point Allocation', 'AHP', 'Entropy']:
    alt_ranks = pd.Series(results[wname]['WSM_rank'], index=df['BANK_NAME'].values)
    gain = (equal_wsm_ranks - alt_ranks).nlargest(5)
    loss = (equal_wsm_ranks - alt_ranks).nsmallest(5)
    print(f"\n--- {wname} vs Equal (WSM) ---")
    print(f"  Top gainers: {', '.join([f'{b} (+{int(v)})' for b, v in gain.items()])}")
    print(f"  Top losers:  {', '.join([f'{b} ({int(v)})' for b, v in loss.items()])}")

# 7.7 Rank stability across all weight schemes (WSM)
wsm_rank_range_all = wsm_rank_matrix_all.max(axis=1) - wsm_rank_matrix_all.min(axis=1)
print("\n15 most stable banks — WSM rank range across all 5 weight schemes:")
stable_tbl = wsm_rank_range_all.nsmallest(15).rename('Rank Range').to_frame()
stable_tbl = stable_tbl.join(wsm_rank_matrix_all)
print(stable_tbl)
print("\n15 most volatile banks — WSM rank range across all 5 weight schemes:")
volatile_tbl = wsm_rank_range_all.nlargest(15).rename('Rank Range').to_frame()
volatile_tbl = volatile_tbl.join(wsm_rank_matrix_all)
print(volatile_tbl)


# ── Section 8: Sensitivity and Robustness Analysis ────────────────────────────

# 8.2 Representation sensitivity: Min-Max vs Rank-based
results_rank = {}
for wname, w in WEIGHTS.items():
    wsm = wsm_scores(norm_rank, w, CRITERIA)
    wpm = wpm_scores(norm_rank, w, CRITERIA)
    results_rank[wname] = {
        'WSM_score': wsm,
        'WPM_score': wpm,
        'WSM_rank':  pd.Series(wsm).rank(ascending=False).astype(int).values,
        'WPM_rank':  pd.Series(wpm).rank(ascending=False).astype(int).values,
    }

wsm_mm_ranks   = pd.Series(results['Point Allocation']['WSM_rank'],      index=df['BANK_NAME'].values)
wsm_rank_ranks = pd.Series(results_rank['Point Allocation']['WSM_rank'], index=df['BANK_NAME'].values)
wpm_mm_ranks   = pd.Series(results['Point Allocation']['WPM_rank'],      index=df['BANK_NAME'].values)
wpm_rank_ranks = pd.Series(results_rank['Point Allocation']['WPM_rank'], index=df['BANK_NAME'].values)

repr_df = pd.DataFrame({
    'WSM_MinMax':    wsm_mm_ranks,
    'WSM_Rankbased': wsm_rank_ranks,
    'WSM_shift':     wsm_rank_ranks - wsm_mm_ranks,
    'WPM_MinMax':    wpm_mm_ranks,
    'WPM_Rankbased': wpm_rank_ranks,
    'WPM_shift':     wpm_rank_ranks - wpm_mm_ranks,
}, index=df['BANK_NAME'].values)

rho_wsm_repr, _ = stats.spearmanr(wsm_mm_ranks, wsm_rank_ranks)
rho_wpm_repr, _ = stats.spearmanr(wpm_mm_ranks, wpm_rank_ranks)
print("\nRepresentation sensitivity (Min-Max vs Rank-based, Point Allocation weights):")
print(f"  WSM Spearman rho: {rho_wsm_repr:.4f}  |  mean |rank shift|: {repr_df['WSM_shift'].abs().mean():.2f}")
print(f"  WPM Spearman rho: {rho_wpm_repr:.4f}  |  mean |rank shift|: {repr_df['WPM_shift'].abs().mean():.2f}")
print("\nTop 15 banks most affected by normalisation choice — WSM (Point Allocation):")
print(
    repr_df.assign(abs_wsm=repr_df['WSM_shift'].abs())
    .nlargest(15, 'abs_wsm')[['WSM_MinMax', 'WSM_Rankbased', 'WSM_shift']]
    .reset_index().rename(columns={'index': 'Bank'})
)

# 8.3 Figure 8.1: Min-Max vs Rank-based rank scatter (WSM and WPM)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, mm_r, rb_r, label, color in zip(
        axes,
        [wsm_mm_ranks, wpm_mm_ranks],
        [wsm_rank_ranks, wpm_rank_ranks],
        ['WSM', 'WPM'],
        ['steelblue', 'salmon']):
    ax.scatter(mm_r, rb_r, alpha=0.55, s=35, color=color, edgecolors='white', linewidth=0.3)
    ax.plot([1, 91], [1, 91], 'k--', lw=0.8, label='Perfect agreement')
    ax.set_xlabel(f'{label} rank — Min-Max normalisation')
    ax.set_ylabel(f'{label} rank — Rank-based normalisation')
    ax.set_title(f'Figure 8.1: {label} — Min-Max vs Rank-based\n(Point Allocation weights)', fontsize=10)
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/sec8/fig8_1_repr_sensitivity_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: outputs/sec8/fig8_1_repr_sensitivity_scatter.png')

# 8.4 Instability decomposition
wsm_rank_matrix_s8 = pd.DataFrame(
    {wname: results[wname]['WSM_rank'] for wname in WEIGHTS},
    index=df['BANK_NAME'].values
)
wpm_rank_matrix_s8 = pd.DataFrame(
    {wname: results[wname]['WPM_rank'] for wname in WEIGHTS},
    index=df['BANK_NAME'].values
)
weight_instab_wsm = wsm_rank_matrix_s8.max(axis=1) - wsm_rank_matrix_s8.min(axis=1)
weight_instab_wpm = wpm_rank_matrix_s8.max(axis=1) - wpm_rank_matrix_s8.min(axis=1)
method_instab     = (wsm_mm_ranks - wpm_mm_ranks).abs()
repr_instab_wsm   = repr_df['WSM_shift'].abs()
repr_instab_wpm   = repr_df['WPM_shift'].abs()
instab_summary = pd.DataFrame({
    'Weight-driven (WSM rank range)':  weight_instab_wsm,
    'Weight-driven (WPM rank range)':  weight_instab_wpm,
    'Method-driven (|WSM-WPM|, PA)':   method_instab,
    'Repr-driven (|MM-Rank|, WSM/PA)': repr_instab_wsm,
    'Repr-driven (|MM-Rank|, WPM/PA)': repr_instab_wpm,
})
print("\nInstability Decomposition — statistics across 91 banks:")
print(pd.DataFrame({
    'Mean':          instab_summary.mean().round(2),
    'Median':        instab_summary.median().round(1),
    'Max':           instab_summary.max().astype(int),
    '# banks > 10':  (instab_summary > 10).sum(),
    '# banks > 20':  (instab_summary > 20).sum(),
}))

# 8.5 Figure 8.2: Instability source comparison (histograms)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
panels = [
    ('Weight-driven (WSM rank range)',    'steelblue',      'Weight-driven (WSM)\nrank range across 5 schemes'),
    ('Weight-driven (WPM rank range)',    'royalblue',      'Weight-driven (WPM)\nrank range across 5 schemes'),
    ('Method-driven (|WSM-WPM|, PA)',     'darkorange',     'Method-driven\n|WSM rank - WPM rank|, PA'),
    ('Repr-driven (|MM-Rank|, WSM/PA)',   'seagreen',       'Repr-driven (WSM)\n|Min-Max rank - Rank-based rank|'),
    ('Repr-driven (|MM-Rank|, WPM/PA)',   'mediumseagreen', 'Repr-driven (WPM)\n|Min-Max rank - Rank-based rank|'),
]
for ax, (col, color, title) in zip(axes, panels):
    vals = instab_summary[col].values
    ax.hist(vals, bins=20, edgecolor='black', color=color, alpha=0.8)
    ax.axvline(vals.mean(), color='red', linestyle='--', label=f'Mean: {vals.mean():.1f}')
    ax.set_title(f'Figure 8.2: {title}', fontsize=9)
    ax.set_xlabel('Absolute rank shift')
    ax.set_ylabel('Number of banks')
    ax.legend(fontsize=8)
axes[5].set_visible(False)
plt.suptitle('Figure 8.2: Instability Decomposition — Weight, Method, and Representation Sources',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/sec8/fig8_2_instability_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: outputs/sec8/fig8_2_instability_decomposition.png')

# 8.6 Rank reversals and top-20 stability
median_rank = 46
top_to_bottom = (wsm_mm_ranks <= median_rank) & (wsm_rank_ranks > median_rank)
bottom_to_top = (wsm_mm_ranks > median_rank)  & (wsm_rank_ranks <= median_rank)
reversals_df = repr_df[top_to_bottom | bottom_to_top][
    ['WSM_MinMax', 'WSM_Rankbased', 'WSM_shift']].sort_values('WSM_MinMax')
print(f"\nRank reversals crossing median (rank 46) — Min-Max vs Rank-based, WSM, PA weights:")
print(f"  Top-half -> Bottom-half: {top_to_bottom.sum()} banks")
print(f"  Bottom-half -> Top-half: {bottom_to_top.sum()} banks")
if len(reversals_df) > 0:
    print(reversals_df.reset_index().rename(columns={'index': 'Bank'}))

top20_primary = set(wsm_mm_ranks.nsmallest(20).index)
print("\nTop-20 stability — how many primary top-20 banks remain in top-20 under alternative specs:")
for label, rank_s in [
    ('WPM, PA, Min-Max',      wpm_mm_ranks),
    ('WSM, Equal, Min-Max',   pd.Series(results['Equal']['WSM_rank'],   index=df['BANK_NAME'].values)),
    ('WSM, AHP, Min-Max',     pd.Series(results['AHP']['WSM_rank'],     index=df['BANK_NAME'].values)),
    ('WSM, Entropy, Min-Max', pd.Series(results['Entropy']['WSM_rank'], index=df['BANK_NAME'].values)),
    ('WSM, PA, Rank-based',   wsm_rank_ranks),
]:
    overlap = len(top20_primary & set(rank_s.nsmallest(20).index))
    print(f"  {label:42s}: {overlap}/20 retained")

print("\nAnalysis complete. All outputs saved to outputs/")
