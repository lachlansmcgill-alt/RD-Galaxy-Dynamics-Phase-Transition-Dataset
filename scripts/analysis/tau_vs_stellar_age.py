"""
Test 3: τ vs Stellar Age (g-W1 Color Proxy)
============================================
g-W1 color is an excellent stellar age indicator:
- Blue (g-W1 < 2.5): Young populations (< 3 Gyr)
- Intermediate (2.5-3.5): Mixed ages (3-8 Gyr)
- Red (g-W1 > 3.5): Old populations (> 8 Gyr)

RD Prediction: High τ → Young (blue, low g-W1)
Expected: ρ(τ, g-W1) < -0.3 (anti-correlation)
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

# Unit conversion
CONVERSION_FACTOR = 978.4  # kpc/(km/s) to Myr

print("="*70)
print("RD TEST 3: tau vs Stellar Age (g-W1 Color)")
print("="*70)

# Load tau data
coherence = pd.read_csv('data/results/sparc_spirals/extended_coherence_params.csv')
coherence['tau_Myr'] = coherence['tau_unified'] * CONVERSION_FACTOR
coherence['Galaxy'] = coherence['Galaxy'].str.strip()

print(f"\nLoaded {len(coherence)} galaxies with tau")

# Load WISE data with g-W1 color
wise = pd.read_excel('data/raw/WISE/WISE_Stellar_Masses.xlsx')
wise.columns = wise.columns.str.strip()
wise['ID'] = wise['ID'].str.strip()

# Convert g-W1 color to numeric
wise['g-W1 color'] = pd.to_numeric(wise['g-W1 color'], errors='coerce')

print(f"Loaded {len(wise)} galaxies with WISE photometry")
print(f"Columns: {wise.columns.tolist()}")

# Merge on galaxy name
df = pd.merge(coherence, wise, left_on='Galaxy', right_on='ID', how='inner')
print(f"\nMatched {len(df)} galaxies")

# Remove invalid values
df = df.dropna(subset=['tau_Myr', 'g-W1 color'])
df = df[df['tau_Myr'] > 0]  # Only positive tau for now
print(f"Valid galaxies: {len(df)}")

# Get arrays
tau = df['tau_Myr'].values
g_W1 = df['g-W1 color'].values

print(f"\ntau range: {tau.min():.1f} - {tau.max():.1f} Myr")
print(f"g-W1 range: {g_W1.min():.2f} - {g_W1.max():.2f}")

# Correlation tests
rho_spearman, p_spearman = spearmanr(tau, g_W1)
rho_pearson, p_pearson = pearsonr(tau, g_W1)

print("\n" + "="*70)
print("CORRELATION RESULTS")
print("="*70)
print(f"Sample size: N = {len(df)}")
print(f"\nSpearman rho(tau, g-W1):  {rho_spearman:+.3f}")
print(f"P-value:                  {p_spearman:.2e}")
print(f"\nPearson r(tau, g-W1):     {rho_pearson:+.3f}")
print(f"P-value:                  {p_pearson:.2e}")

# Significance
if p_spearman < 0.001:
    sigma = abs(rho_spearman) / (1 / np.sqrt(len(df)))
    print(f"Significance:             {sigma:.1f}σ")
    if p_spearman < 1e-10:
        print(f"Result:                   HIGHLY SIGNIFICANT ***")
    elif p_spearman < 1e-5:
        print(f"Result:                   VERY SIGNIFICANT **")
    else:
        print(f"Result:                   SIGNIFICANT *")
elif p_spearman < 0.05:
    print(f"Result:                   MARGINAL")
else:
    print(f"Result:                   NOT SIGNIFICANT")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

if rho_spearman < -0.2 and p_spearman < 0.001:
    print(f"✓ PREDICTION CONFIRMED (ANTI-CORRELATION)!")
    print(f"  High tau → Low g-W1 (blue) → Young populations")
    print(f"  Low tau → High g-W1 (red) → Old populations")
    print(f"\n  This validates tau as age-dependent parameter!")
    print(f"  Consistent with:")
    print(f"    - Gas fraction (rho=+0.612, 11.7σ)")
    print(f"    - Hubble Type (rho=+0.668, 6.8σ)")
elif rho_spearman < -0.1 and p_spearman < 0.05:
    print(f"✓ WEAK ANTI-CORRELATION")
    print(f"  Trend supports prediction but weak")
elif rho_spearman > 0.2 and p_spearman < 0.001:
    print(f"✗ OPPOSITE OF PREDICTION!")
    print(f"  High tau → Red → Old (contradicts other tests!)")
else:
    print(f"○ NO CLEAR CORRELATION")
    print(f"  g-W1 may not be sensitive enough")

# Age categories
print("\n" + "="*70)
print("BREAKDOWN BY AGE CATEGORY (g-W1)")
print("="*70)

young = df[df['g-W1 color'] < 2.5]
intermediate = df[(df['g-W1 color'] >= 2.5) & (df['g-W1 color'] < 3.5)]
old = df[df['g-W1 color'] >= 3.5]

print(f"\n{'Category':<15} {'g-W1 Range':<12} {'N':>4} {'<tau> (Myr)':>13} {'sigma':>9}")
print("-" * 60)
print(f"{'Young':<15} {'< 2.5':<12} {len(young):>4} {young['tau_Myr'].mean():>13.1f} {young['tau_Myr'].std():>9.1f}")
print(f"{'Intermediate':<15} {'2.5-3.5':<12} {len(intermediate):>4} {intermediate['tau_Myr'].mean():>13.1f} {intermediate['tau_Myr'].std():>9.1f}")
print(f"{'Old':<15} {'> 3.5':<12} {len(old):>4} {old['tau_Myr'].mean():>13.1f} {old['tau_Myr'].std():>9.1f}")

# Statistical test between young and old
from scipy.stats import mannwhitneyu
if len(young) > 0 and len(old) > 0:
    u_stat, p_utest = mannwhitneyu(young['tau_Myr'], old['tau_Myr'], alternative='greater')
    print(f"\nMann-Whitney U test (Young > Old):")
    print(f"  U statistic: {u_stat:.1f}")
    print(f"  P-value: {p_utest:.2e}")
    if p_utest < 0.001:
        print(f"  Result: Young galaxies have SIGNIFICANTLY higher tau ***")
    elif p_utest < 0.05:
        print(f"  Result: Young galaxies have higher tau *")
    else:
        print(f"  Result: No significant difference")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Scatter plot
ax = axes[0, 0]
scatter = ax.scatter(g_W1, tau, alpha=0.6, s=50, c=g_W1, cmap='RdYlBu_r',
                     edgecolors='k', linewidth=0.5)
ax.set_xlabel('g-W1 Color (Age Proxy)', fontsize=12)
ax.set_ylabel('tau (Myr)', fontsize=12)
ax.set_title(f'tau vs g-W1 Color (rho = {rho_spearman:+.3f}, p = {p_spearman:.2e})',
             fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)

# Add trend line
z = np.polyfit(g_W1, tau, 1)
p_fit = np.poly1d(z)
g_W1_line = np.linspace(g_W1.min(), g_W1.max(), 100)
ax.plot(g_W1_line, p_fit(g_W1_line), 'r--', alpha=0.8, linewidth=2,
        label=f'Linear fit: tau = {z[0]:.1f}*(g-W1) + {z[1]:.1f}')
ax.legend()

# Add age labels
ax.axvline(2.5, color='gray', linestyle=':', alpha=0.5)
ax.axvline(3.5, color='gray', linestyle=':', alpha=0.5)
ax.text(2.0, ax.get_ylim()[1]*0.9, 'Young', fontsize=10, ha='center')
ax.text(3.0, ax.get_ylim()[1]*0.9, 'Intermediate', fontsize=10, ha='center')
ax.text(4.0, ax.get_ylim()[1]*0.9, 'Old', fontsize=10, ha='center')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('g-W1 (Old→)', fontsize=10)

# 2. Box plot by age category
ax = axes[0, 1]
if len(young) > 0 and len(intermediate) > 0 and len(old) > 0:
    bp = ax.boxplot([young['tau_Myr'], intermediate['tau_Myr'], old['tau_Myr']],
                     tick_labels=['Young\n(<2.5)', 'Intermediate\n(2.5-3.5)', 'Old\n(>3.5)'],
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
ax.set_ylabel('tau (Myr)', fontsize=12)
ax.set_title('tau Distribution by Age Category', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 3. Histogram by age category
ax = axes[1, 0]
if len(young) > 0 and len(intermediate) > 0 and len(old) > 0:
    ax.hist([young['tau_Myr'], intermediate['tau_Myr'], old['tau_Myr']],
            bins=20, alpha=0.7, label=['Young', 'Intermediate', 'Old'],
            color=['blue', 'green', 'red'], edgecolor='k')
    ax.legend()
ax.set_xlabel('tau (Myr)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('tau Distribution by Age', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)

# 4. Mean tau vs g-W1 bins
ax = axes[1, 1]
g_W1_bins = np.linspace(g_W1.min(), g_W1.max(), 8)
bin_centers = (g_W1_bins[:-1] + g_W1_bins[1:]) / 2
bin_means = []
bin_stds = []
for i in range(len(g_W1_bins)-1):
    mask = (g_W1 >= g_W1_bins[i]) & (g_W1 < g_W1_bins[i+1])
    if mask.sum() > 0:
        bin_means.append(tau[mask].mean())
        bin_stds.append(tau[mask].std())
    else:
        bin_means.append(np.nan)
        bin_stds.append(np.nan)

ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=5,
            markersize=8, linewidth=2, color='steelblue')
ax.set_xlabel('g-W1 Color', fontsize=12)
ax.set_ylabel('Mean tau (Myr)', fontsize=12)
ax.set_title('Mean tau vs g-W1 (Binned)', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)
ax.axvline(2.5, color='gray', linestyle=':', alpha=0.5)
ax.axvline(3.5, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('results/stage8/figures/tau_vs_stellar_age_gW1.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: results/stage8/figures/tau_vs_stellar_age_gW1.png")

# Save results
df[['Galaxy', 'tau_Myr', 'g-W1 color', 'Hubble_Type_Num']].to_csv(
    'results/stage8/tau_stellar_age_correlation.csv', index=False)
print(f"Saved: results/stage8/tau_stellar_age_correlation.csv")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if p_spearman < 0.001 and rho_spearman < -0.3:
    print("*** STRONG EVIDENCE: tau anti-correlates with stellar age")
    print("    → High tau galaxies are young (blue, low g-W1)")
    print("    → Low tau galaxies are old (red, high g-W1)")
    print("\n    THREE INDEPENDENT VALIDATIONS:")
    print("    1. Gas fraction: rho = +0.612 (11.7σ)")
    print("    2. Hubble Type: rho = +0.668 (6.8σ)")
    print("    3. Stellar Age: rho = {:.3f} (this test)".format(rho_spearman))
    print("\n    RD PREDICTION: CONFIRMED ✓✓✓")
    print("    tau IS AN EVOLUTIONARY TIMESCALE")
elif p_spearman < 0.01 and rho_spearman < -0.15:
    print("** MODERATE EVIDENCE: tau weakly anti-correlates with age")
    print("   → Trend supports evolutionary interpretation")
    print("   → g-W1 may be imperfect age proxy")
    print("\n   RD PREDICTION: SUPPORTED ✓")
elif p_spearman < 0.05:
    print("* WEAK EVIDENCE: Marginal correlation")
    print("  → Inconclusive, need better age indicators")
    print("\n  RD PREDICTION: INCONCLUSIVE ~")
else:
    print("NO EVIDENCE: tau does not correlate with age proxy")
    print("  → Unexpected! Contradicts gas fraction and morphology")
    print("  → May need direct age measurements (D4000, SED fitting)")
    print("\n  RD PREDICTION: NOT SUPPORTED ✗")

print("\n" + "="*70)
print("VALIDATION STATUS: 3/6 TESTS COMPLETED")
print("="*70)
