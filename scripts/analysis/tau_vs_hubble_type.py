"""
Test RD Prediction: τ vs Hubble Type (Stellar Age Proxy)
==========================================================
Hubble Type is strongly correlated with stellar age:
- Early type (T=0-3, E/S0/Sa): Old stars, red colors
- Late type (T=7-10, Sc/Sd/Irr): Young stars, blue colors

RD Prediction: High τ → Late Hubble Type (young, gas-rich)
Expected: ρ(τ, T) > +0.3 (positive correlation)
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

print("="*70)
print("RD TEST: tau_unified vs Hubble Type (Age Proxy)")
print("="*70)

# Load τ_unified from coherence results
coherence = pd.read_csv('data/results/sparc_spirals/extended_coherence_params.csv')
print(f"\n✓ Loaded {len(coherence)} galaxies with tau_unified")

# Load Hubble Type from Table1.txt
# Format: fixed-width, T in columns 12-13 (1-indexed in documentation)
# Using colspecs (0-indexed): Galaxy=0-11, T=11-13, D=13-19, etc.
table1 = pd.read_fwf('data/raw/Table1.txt', 
                      skiprows=24,  # Skip header (data starts line 25)
                      colspecs=[(0,11), (11,13), (13,19), (19,24), (24,26), 
                               (26,30), (30,34), (34,41), (41,48), (48,53),
                               (53,61), (61,66), (66,74), (74,81), (81,86),
                               (86,91), (91,96), (96,99), (99,113)],
                      names=['Galaxy', 'T', 'D', 'e_D', 'f_D', 'Inc', 'e_Inc',
                             'L36', 'e_L36', 'Reff', 'SBeff', 'Rdisk', 'SBdisk',
                             'MHI', 'RHI', 'Vflat', 'e_Vflat', 'Q', 'Ref'])

print(f"✓ Loaded {len(table1)} galaxies from Table1.txt")

# Clean galaxy names
table1['Galaxy'] = table1['Galaxy'].str.strip()
coherence['Galaxy'] = coherence['Galaxy'].str.strip()

# Convert T to numeric (it's being read as string)
table1['T'] = pd.to_numeric(table1['T'], errors='coerce')
table1['Vflat'] = pd.to_numeric(table1['Vflat'], errors='coerce')

# Merge on galaxy name
merged = pd.merge(coherence, table1[['Galaxy', 'T', 'D', 'Inc', 'Vflat']], 
                  on='Galaxy', how='inner')

print(f"✓ Matched {len(merged)} galaxies with both tau and Hubble Type")

# Remove invalid values
merged = merged[(merged['tau_unified'] > 0) & 
                 (merged['T'].notna()) & 
                 (merged['T'] >= 0) &
                 (merged['T'] <= 10)]

print(f"✓ {len(merged)} galaxies with valid tau and T")

# Correlation test
tau = merged['tau_unified'].values
T = merged['T'].values

rho_spearman, p_spearman = spearmanr(tau, T)
rho_pearson, p_pearson = pearsonr(tau, T)

print("\n" + "="*70)
print("CORRELATION RESULTS")
print("="*70)
print(f"Sample size: N = {len(merged)}")
print(f"\nSpearman ρ(τ, T):  {rho_spearman:+.3f}")
print(f"P-value:           {p_spearman:.2e}")
print(f"\nPearson r(τ, T):   {rho_pearson:+.3f}")
print(f"P-value:           {p_pearson:.2e}")

# Significance
if p_spearman < 0.001:
    sigma = abs(rho_spearman) / (1 / np.sqrt(len(merged)))
    print(f"Significance:      {sigma:.1f}σ")
    print(f"Result:            {'HIGHLY SIGNIFICANT ✓✓✓' if p_spearman < 1e-5 else 'SIGNIFICANT ✓'}")
else:
    print(f"Result:            {'NOT SIGNIFICANT ✗' if p_spearman > 0.05 else 'MARGINAL'}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

if rho_spearman > 0.2:
    print(f"✓ PREDICTION CONFIRMED!")
    print(f"  High τ → Late Hubble Type (Sc, Sd, Irr)")
    print(f"  → Young stellar populations")
    print(f"  → Gas-rich, actively star-forming")
    print(f"\n  This validates τ as evolutionary clock!")
elif rho_spearman < -0.2:
    print(f"✗ OPPOSITE OF PREDICTION")
    print(f"  High τ → Early Hubble Type (E, S0, Sa)")
    print(f"  → This would contradict gas fraction result")
else:
    print(f"⚠ WEAK OR NO CORRELATION")
    print(f"  τ may not correlate with Hubble Type")
    print(f"  → Hubble Type crude age proxy?")
    print(f"  → Need better age indicators (colors, SFR)")

# Hubble Type categories
print("\n" + "="*70)
print("BREAKDOWN BY HUBBLE TYPE")
print("="*70)

type_labels = {
    0: 'E (Elliptical)',
    1: 'S0 (Lenticular)', 
    2: 'Sa',
    3: 'Sab',
    4: 'Sb',
    5: 'Sbc',
    6: 'Sc',
    7: 'Scd',
    8: 'Sd',
    9: 'Sdm',
    10: 'Sm/Irr'
}

print(f"\n{'Type':<15} {'N':>4} {'<τ> (Myr)':>12} {'σ(τ)':>8}")
print("-" * 45)

for t in sorted(merged['T'].unique()):
    subset = merged[merged['T'] == t]
    mean_tau = subset['tau_unified'].mean()
    std_tau = subset['tau_unified'].std()
    label = type_labels.get(int(t), f'T={int(t)}')
    print(f"{label:<15} {len(subset):>4} {mean_tau:>12.2f} {std_tau:>8.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Scatter plot
ax = axes[0, 0]
ax.scatter(T, tau, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
ax.set_xlabel('Hubble Type T', fontsize=12)
ax.set_ylabel('τ_unified (Myr)', fontsize=12)
ax.set_title(f'τ vs Hubble Type (ρ = {rho_spearman:+.3f}, p = {p_spearman:.2e})', 
             fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)

# Add trend line
z = np.polyfit(T, tau, 1)
p_fit = np.poly1d(z)
T_line = np.linspace(T.min(), T.max(), 100)
ax.plot(T_line, p_fit(T_line), 'r--', alpha=0.8, linewidth=2, 
        label=f'Linear fit: τ = {z[0]:.2f}T + {z[1]:.2f}')
ax.legend()

# Type labels on x-axis
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xticklabels(['E/S0', 'Sa', 'Sb', 'Sc', 'Sd', 'Irr'])

# 2. Box plot by type category
ax = axes[0, 1]
early = merged[merged['T'] <= 3]['tau_unified']
mid = merged[(merged['T'] > 3) & (merged['T'] <= 7)]['tau_unified']
late = merged[merged['T'] > 7]['tau_unified']

bp = ax.boxplot([early, mid, late], 
                 labels=['Early\n(T≤3)', 'Mid\n(4≤T≤7)', 'Late\n(T>7)'],
                 patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel('τ_unified (Myr)', fontsize=12)
ax.set_title('τ Distribution by Type Category', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

print(f"\nEarly types (T≤3): <τ> = {early.mean():.2f} ± {early.std():.2f} Myr (N={len(early)})")
print(f"Mid types (4≤T≤7):  <τ> = {mid.mean():.2f} ± {mid.std():.2f} Myr (N={len(mid)})")
print(f"Late types (T>7):   <τ> = {late.mean():.2f} ± {late.std():.2f} Myr (N={len(late)})")

# 3. Histogram of residuals
ax = axes[1, 0]
residuals = tau - p_fit(T)
ax.hist(residuals, bins=30, alpha=0.7, edgecolor='k', color='steelblue')
ax.axvline(0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Residual: τ_obs - τ_fit (Myr)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Residuals (σ = {residuals.std():.2f} Myr)', 
             fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)

# 4. τ vs Vflat colored by T
ax = axes[1, 1]
scatter = ax.scatter(merged['Vflat'], merged['tau_unified'], 
                     c=merged['T'], cmap='RdYlBu_r', s=50, 
                     edgecolors='k', linewidth=0.5, alpha=0.8)
ax.set_xlabel('Vflat (km/s)', fontsize=12)
ax.set_ylabel('τ_unified (Myr)', fontsize=12)
ax.set_title('τ vs Vflat (color = Hubble Type)', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('T (E→Irr)', fontsize=10)

plt.tight_layout()
plt.savefig('results/stage8/figures/tau_vs_hubble_type.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: results/stage8/figures/tau_vs_hubble_type.png")

# Save results
output = merged[['Galaxy', 'tau_unified', 'T', 'D', 'Inc', 'Vflat']].copy()
output.to_csv('results/stage8/tau_hubble_correlation.csv', index=False)
print(f"✓ Saved: results/stage8/tau_hubble_correlation.csv")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if p_spearman < 0.001 and rho_spearman > 0.2:
    print("✓✓✓ STRONG EVIDENCE: τ correlates with Hubble Type")
    print("    → High τ galaxies are late-type (young, gas-rich)")
    print("    → Consistent with gas fraction result (ρ=+0.612)")
    print("    → τ is validated as evolutionary time parameter")
elif p_spearman < 0.05:
    print("✓ MODERATE EVIDENCE: Weak but significant correlation")
    print("  → Hubble Type is crude age proxy")
    print("  → Need colors (U-B, g-r) for better test")
else:
    print("✗ NO EVIDENCE: τ does not correlate with Hubble Type")
    print("  → Unexpected! Check data quality")
    print("  → Hubble Type may not reflect stellar age well")

print("="*70)
