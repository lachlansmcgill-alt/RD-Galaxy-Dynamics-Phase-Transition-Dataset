"""
gas_fraction_correlation.py
----------------------------
Test correlation between τ_unified and gas fraction (M_HI/M_star)

Tests RD prediction: High τ → high gas fraction (young, gas-rich)
                      Low τ → low gas fraction (mature, depleted)

Usage:
    python scripts/analysis/gas_fraction_correlation.py
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

print("="*80)
print("GAS FRACTION CORRELATION TEST")
print("="*80)

# Load Table1.txt (SPARC HI data)
print("\n1. Loading SPARC Table1 (HI masses)...")
table1_path = ROOT / "raw_data" / "Table1.txt"

# Parse fixed-width format based on byte positions
colspecs = [
    (0, 11),    # Galaxy
    (11, 13),   # T (Hubble Type)
    (13, 19),   # D (Distance)
    (19, 24),   # e_D
    (24, 26),   # f_D
    (26, 30),   # Inc
    (30, 34),   # e_Inc
    (34, 41),   # L[3.6]
    (41, 48),   # e_L[3.6]
    (48, 53),   # Reff
    (53, 61),   # SBeff
    (61, 66),   # Rdisk
    (66, 74),   # SBdisk
    (74, 81),   # MHI (10^9 Msun)
    (81, 86),   # RHI
    (86, 91),   # Vflat
    (91, 96),   # e_Vflat
    (96, 99),   # Q
    (99, 113),  # Ref
]

names = ['Galaxy', 'T', 'D_Mpc', 'e_D', 'f_D', 'Inc', 'e_Inc', 
         'L_3.6', 'e_L_3.6', 'Reff', 'SBeff', 'Rdisk', 'SBdisk',
         'M_HI_1e9', 'R_HI', 'V_flat', 'e_Vflat', 'Q', 'Ref']

df_table1 = pd.read_fwf(table1_path, colspecs=colspecs, names=names, 
                        skiprows=23, skipinitialspace=True)

# Clean galaxy names
df_table1['Galaxy'] = df_table1['Galaxy'].str.strip()

# Convert numeric columns
numeric_cols = ['T', 'D_Mpc', 'e_D', 'Inc', 'e_Inc', 'L_3.6', 'e_L_3.6',
                'Reff', 'SBeff', 'Rdisk', 'SBdisk', 'M_HI_1e9', 'R_HI', 
                'V_flat', 'e_Vflat', 'Q']
for col in numeric_cols:
    df_table1[col] = pd.to_numeric(df_table1[col], errors='coerce')

print(f"   Loaded {len(df_table1)} galaxies")
print(f"   Galaxies with HI data: {(df_table1['M_HI_1e9'] > 0).sum()}")

# Load stellar masses
print("\n2. Loading stellar masses...")
galaxy_master = ROOT / "data/results/sparc_spirals/tau_evolutionary_analysis.csv"
df_master = pd.read_csv(galaxy_master)

# Merge to get M_star
df_hi = df_table1.merge(df_master[['Galaxy', 'logMstar']], on='Galaxy', how='inner')
print(f"   Matched {len(df_hi)} galaxies with stellar masses")

# Load extended coherence parameters (τ_unified)
print("\n3. Loading τ_unified values...")
tau_path = ROOT / "data/results/sparc_spirals/extended_coherence_params.csv"
df_tau = pd.read_csv(tau_path)

# Merge all data
df = df_hi.merge(df_tau[['Galaxy', 'tau_unified', 'eta']], on='Galaxy', how='inner')
print(f"   Final matched sample: {len(df)} galaxies")

# Compute gas fraction
print("\n4. Computing gas fractions...")

# Convert M_HI from 10^9 Msun to Msun
df['M_HI'] = df['M_HI_1e9'] * 1e9

# Convert log(M_star) to M_star
df['M_star'] = 10**df['logMstar']

# Compute gas fraction f_gas = M_HI / M_star
df['f_gas'] = df['M_HI'] / df['M_star']

# Also compute gas-to-baryon fraction f_gas_baryon = M_HI / (M_HI + M_star)
df['f_gas_baryon'] = df['M_HI'] / (df['M_HI'] + df['M_star'])

# Log gas fraction for plotting
df['log_f_gas'] = np.log10(df['f_gas'])

# Remove any infinities or NaNs
df_clean = df[np.isfinite(df['tau_unified']) & np.isfinite(df['f_gas']) & (df['f_gas'] > 0)]
print(f"   Clean sample (no NaN/inf): {len(df_clean)} galaxies")

# Convert tau to Myr for interpretation
conversion = 3.086e19 / 1000.0 / 3.154e13  # kpc/(km/s) → Myr
df_clean['tau_Myr'] = df_clean['tau_unified'] * conversion

print(f"\n   Gas fraction statistics:")
print(f"      Mean f_gas: {df_clean['f_gas'].mean():.3f}")
print(f"      Median f_gas: {df_clean['f_gas'].median():.3f}")
print(f"      Range: {df_clean['f_gas'].min():.3f} - {df_clean['f_gas'].max():.3f}")
print(f"\n   τ_unified statistics:")
print(f"      Mean: {df_clean['tau_Myr'].mean():.1f} Myr")
print(f"      Median: {df_clean['tau_Myr'].median():.1f} Myr")
print(f"      Range: {df_clean['tau_Myr'].min():.1f} - {df_clean['tau_Myr'].max():.1f} Myr")

# Correlation tests
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# Test 1: τ vs f_gas
rho_fgas, p_fgas = spearmanr(df_clean['tau_unified'], df_clean['f_gas'])
r_fgas, p_fgas_pearson = pearsonr(df_clean['tau_unified'], df_clean['f_gas'])

print(f"\nτ_unified vs f_gas (M_HI/M_star):")
print(f"   Spearman ρ = {rho_fgas:+.3f}, p = {p_fgas:.2e}")
print(f"   Pearson  r = {r_fgas:+.3f}, p = {p_fgas_pearson:.2e}")

# Test 2: τ vs log(f_gas)
rho_logfgas, p_logfgas = spearmanr(df_clean['tau_unified'], df_clean['log_f_gas'])
r_logfgas, p_logfgas_pearson = pearsonr(df_clean['tau_unified'], df_clean['log_f_gas'])

print(f"\nτ_unified vs log(f_gas):")
print(f"   Spearman ρ = {rho_logfgas:+.3f}, p = {p_logfgas:.2e}")
print(f"   Pearson  r = {r_logfgas:+.3f}, p = {p_logfgas_pearson:.2e}")

# Test 3: τ vs f_gas_baryon
rho_baryon, p_baryon = spearmanr(df_clean['tau_unified'], df_clean['f_gas_baryon'])

print(f"\nτ_unified vs f_gas_baryon (M_HI/(M_HI+M_star)):")
print(f"   Spearman ρ = {rho_baryon:+.3f}, p = {p_baryon:.2e}")

# Test 4: Compare to M_HI alone
rho_mhi, p_mhi = spearmanr(df_clean['tau_unified'], df_clean['M_HI'])

print(f"\nτ_unified vs M_HI (raw HI mass):")
print(f"   Spearman ρ = {rho_mhi:+.3f}, p = {p_mhi:.2e}")

# Interpretation
print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

sigma = abs(rho_fgas) / np.sqrt((1 - rho_fgas**2) / (len(df_clean) - 2))
print(f"\nSignificance: ~{sigma:.1f}σ detection")

if rho_fgas > 0.3 and p_fgas < 0.01:
    print("\n✓ POSITIVE CORRELATION DETECTED")
    print("  → High τ associated with high gas fraction")
    print("  → Young, gas-rich galaxies have extended coherence")
    print("  → Consistent with RD evolutionary clock prediction")
elif rho_fgas < -0.3 and p_fgas < 0.01:
    print("\n✗ NEGATIVE CORRELATION (unexpected)")
    print("  → High τ associated with LOW gas fraction")
    print("  → Contradicts simple evolutionary picture")
else:
    print("\n⚠ WEAK/NO CORRELATION")
    print("  → Gas fraction may not be primary driver")
    print("  → Or correlation obscured by other factors")

# Create visualization
print("\n" + "="*80)
print("GENERATING FIGURES")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: τ vs f_gas
ax = axes[0, 0]
scatter = ax.scatter(df_clean['tau_Myr'], df_clean['f_gas'], 
                     c=df_clean['logMstar'], cmap='viridis', 
                     alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
ax.set_xlabel('τ_unified [Myr]', fontsize=12)
ax.set_ylabel('f_gas = M_HI / M_*', fontsize=12)
ax.set_title(f'Gas Fraction vs Evolutionary State\nρ = {rho_fgas:+.3f}, p = {p_fgas:.2e}', 
             fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('log(M_*/M☉)', fontsize=10)

# Add trend line
z = np.polyfit(df_clean['tau_Myr'], df_clean['f_gas'], 1)
p = np.poly1d(z)
tau_range = np.linspace(df_clean['tau_Myr'].min(), df_clean['tau_Myr'].max(), 100)
ax.plot(tau_range, p(tau_range), 'r--', alpha=0.5, linewidth=2, label='Linear fit')
ax.legend()

# Panel 2: τ vs log(f_gas)
ax = axes[0, 1]
ax.scatter(df_clean['tau_Myr'], df_clean['log_f_gas'], 
           c=df_clean['logMstar'], cmap='viridis',
           alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
ax.set_xlabel('τ_unified [Myr]', fontsize=12)
ax.set_ylabel('log(f_gas)', fontsize=12)
ax.set_title(f'Log Gas Fraction vs τ\nρ = {rho_logfgas:+.3f}, p = {p_logfgas:.2e}', 
             fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)
ax.axhline(0, color='k', linestyle=':', alpha=0.3)

# Panel 3: f_gas vs M_star
ax = axes[1, 0]
ax.scatter(df_clean['logMstar'], df_clean['f_gas'],
           c=df_clean['tau_Myr'], cmap='coolwarm',
           alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
ax.set_xlabel('log(M_*/M☉)', fontsize=12)
ax.set_ylabel('f_gas', fontsize=12)
ax.set_title('Gas Fraction vs Stellar Mass\n(colored by τ)', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_yscale('log')

# Panel 4: Histogram comparison
ax = axes[1, 1]
# Split by gas fraction
high_gas = df_clean[df_clean['f_gas'] > df_clean['f_gas'].median()]
low_gas = df_clean[df_clean['f_gas'] <= df_clean['f_gas'].median()]

ax.hist(high_gas['tau_Myr'], bins=15, alpha=0.5, label=f'High f_gas (>{df_clean["f_gas"].median():.2f})', 
        color='blue', edgecolor='k')
ax.hist(low_gas['tau_Myr'], bins=15, alpha=0.5, label=f'Low f_gas (≤{df_clean["f_gas"].median():.2f})', 
        color='red', edgecolor='k')
ax.set_xlabel('τ_unified [Myr]', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('τ Distribution by Gas Content', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
output_fig = ROOT / "results/stage8/figures/gas_fraction_correlation.png"
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved figure: {output_fig}")

# Summary statistics by gas fraction bins
print("\n" + "="*80)
print("GAS FRACTION BINS")
print("="*80)

bins = [0, 0.1, 0.5, 1.0, df_clean['f_gas'].max()]
labels = ['Very Low (<0.1)', 'Low (0.1-0.5)', 'Medium (0.5-1.0)', 'High (>1.0)']

df_clean['f_gas_bin'] = pd.cut(df_clean['f_gas'], bins=bins, labels=labels)

for label in labels:
    subset = df_clean[df_clean['f_gas_bin'] == label]
    if len(subset) > 0:
        print(f"\n{label}: N={len(subset)}")
        print(f"   Mean τ: {subset['tau_Myr'].mean():.1f} Myr")
        print(f"   Median τ: {subset['tau_Myr'].median():.1f} Myr")
        print(f"   Mean log(M*): {subset['logMstar'].mean():.2f}")

# Save results
output_csv = ROOT / "results/stage8/gas_fraction_results.csv"
df_clean[['Galaxy', 'tau_unified', 'tau_Myr', 'M_HI', 'M_star', 'f_gas', 
          'f_gas_baryon', 'log_f_gas', 'logMstar', 'T']].to_csv(output_csv, index=False)
print(f"\n✓ Saved results: {output_csv}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nSample: {len(df_clean)} SPARC galaxies with HI, M*, and τ data")
print(f"\nCorrelation: τ_unified vs f_gas = {rho_fgas:+.3f} (p = {p_fgas:.2e})")

if p_fgas < 0.001:
    print("Result: STATISTICALLY SIGNIFICANT")
else:
    print("Result: Not significant")

print("\nRD Prediction: Positive correlation (high τ → high f_gas)")
if rho_fgas > 0.3:
    print("Status: ✓ CONFIRMED")
elif rho_fgas < -0.3:
    print("Status: ✗ CONTRADICTED")
else:
    print("Status: ⚠ INCONCLUSIVE")

print("\n" + "="*80)
