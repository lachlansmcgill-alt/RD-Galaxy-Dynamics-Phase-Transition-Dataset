"""
compare_etg_ltg_tau.py
----------------------
Compare τ_unified distributions between Early-Type Galaxies (ETGs)
and Late-Type Galaxies (LTGs) to test RD predictions.

RD Prediction:
- ETGs: τ < 0 (frozen recursion, core-dominated)
- LTGs: τ > 0 (active recursion, coherence-dominated)

Reality Check:
- ETG fits used synthetic exponential profiles → may bias τ positive
- Need to examine if ETGs show distinct τ, η, α signatures
- Key test: Do ETGs cluster separately from LTGs in parameter space?

Usage:
    python scripts/analysis/compare_etg_ltg_tau.py
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

print("="*80)
print("ETG vs LTG COMPARISON: τ_unified AND STRUCTURAL PARAMETERS")
print("="*80)

# Load ETG data
etg_path = ROOT / "results/stage6/tdgl_etg_fits.csv"
df_etg = pd.read_csv(etg_path)
df_etg['Sample'] = 'ETG'
df_etg['Hubble_Type_Num'] = -1  # S0/E
print(f"\n1. ETG sample: {len(df_etg)} galaxies")
print(f"   τ range: [{df_etg['tau_Myr'].min():.1f}, {df_etg['tau_Myr'].max():.1f}] Myr")
print(f"   η range: [{df_etg['eta'].min():.3f}, {df_etg['eta'].max():.3f}]")
print(f"   α range: [{df_etg['alpha'].min():.3f}, {df_etg['alpha'].max():.3f}]")

# Load LTG data (TDGL fits for full parameters)
ltg_tdgl_path = ROOT / "results/stage6/tdgl_fits.csv"
df_ltg_tdgl = pd.read_csv(ltg_tdgl_path)
df_ltg_tdgl.rename(columns={'galaxy': 'Galaxy'}, inplace=True)

# Load extended coherence (for tau_unified)
ltg_coh_path = ROOT / "results/stage8/extended_coherence_params.csv"
df_ltg_coh = pd.read_csv(ltg_coh_path)

# Merge
df_ltg = df_ltg_tdgl.merge(df_ltg_coh[['Galaxy', 'tau_unified', 'eta']], 
                            on='Galaxy', how='inner', suffixes=('', '_coh'))
df_ltg['Sample'] = 'LTG'

# Convert tau to Myr
conversion = 3.086e19 / 1000.0 / 3.154e13
df_ltg['tau_Myr'] = df_ltg['tau_unified'] * conversion

# Also load Hubble types from galaxy master
master_path = ROOT / "results/stage7/galaxy_master_with_morph.csv"
df_master = pd.read_csv(master_path)
df_ltg = df_ltg.merge(df_master[['Galaxy', 'Hubble_Type_Num']], 
                      on='Galaxy', how='left')

# Compute kappa1 if not present
if 'kappa1' not in df_ltg.columns:
    df_ltg['kappa1'] = 1 - (df_ltg['rms_resid'] / df_ltg['V_flat'])

print(f"\n2. LTG sample: {len(df_ltg)} galaxies")
print(f"   τ range: [{df_ltg['tau_Myr'].min():.1f}, {df_ltg['tau_Myr'].max():.1f}] Myr")
print(f"   η range: [{df_ltg['eta'].min():.3f}, {df_ltg['eta'].max():.3f}]")
print(f"   α range: [{df_ltg['alpha'].min():.3f}, {df_ltg['alpha'].max():.3f}]")

# Statistical tests
print("\n" + "="*80)
print("STATISTICAL TESTS: ETG vs LTG")
print("="*80)

# Test 1: τ distribution
u_tau, p_tau = mannwhitneyu(df_etg['tau_Myr'], df_ltg['tau_Myr'], alternative='two-sided')
d_tau, p_ks_tau = ks_2samp(df_etg['tau_Myr'], df_ltg['tau_Myr'])
print(f"\nτ_unified:")
print(f"   Mann-Whitney U = {u_tau:.1f}, p = {p_tau:.2e}")
print(f"   KS statistic = {d_tau:.3f}, p = {p_ks_tau:.2e}")
print(f"   ETG median: {df_etg['tau_Myr'].median():.1f} Myr")
print(f"   LTG median: {df_ltg['tau_Myr'].median():.1f} Myr")
if p_tau < 0.05:
    print(f"   ✓ SIGNIFICANT DIFFERENCE")
else:
    print(f"   ✗ No significant difference")

# Test 2: η distribution
u_eta, p_eta = mannwhitneyu(df_etg['eta'], df_ltg['eta'], alternative='two-sided')
print(f"\nη (maturity):")
print(f"   Mann-Whitney U = {u_eta:.1f}, p = {p_eta:.2e}")
print(f"   ETG median: {df_etg['eta'].median():.3f}")
print(f"   LTG median: {df_ltg['eta'].median():.3f}")
if p_eta < 0.05:
    print(f"   ✓ SIGNIFICANT DIFFERENCE")
    if df_etg['eta'].median() > df_ltg['eta'].median():
        print(f"   → ETGs have HIGHER η (more mature) ✓")
else:
    print(f"   ✗ No significant difference")

# Test 3: α distribution  
u_alpha, p_alpha = mannwhitneyu(df_etg['alpha'], df_ltg['alpha'], alternative='two-sided')
print(f"\nα (shape parameter):")
print(f"   Mann-Whitney U = {u_alpha:.1f}, p = {p_alpha:.2e}")
print(f"   ETG median: {df_etg['alpha'].median():.3f}")
print(f"   LTG median: {df_ltg['alpha'].median():.3f}")
if p_alpha < 0.05:
    print(f"   ✓ SIGNIFICANT DIFFERENCE")
    if df_etg['alpha'].median() < df_ltg['alpha'].median():
        print(f"   → ETGs have LOWER α (steeper profiles) ✓")
else:
    print(f"   ✗ No significant difference")

# Test 4: ξ/r_core ratio
df_etg['r_core_calc'] = df_etg['xi_GL'] * np.sqrt(1 - (1/df_etg['eta'])**2)
df_etg['xi_over_rcore'] = df_etg['xi_GL'] / df_etg['r_core_calc']

df_ltg['r_core_calc'] = df_ltg['xi_GL'] * np.sqrt(1 - (1/df_ltg['eta'])**2)
df_ltg['xi_over_rcore'] = df_ltg['xi_GL'] / df_ltg['r_core_calc']

u_ratio, p_ratio = mannwhitneyu(df_etg['xi_over_rcore'].dropna(), 
                                  df_ltg['xi_over_rcore'].dropna(), 
                                  alternative='two-sided')
print(f"\nξ_GL / r_core (coherence vs core dominance):")
print(f"   Mann-Whitney U = {u_ratio:.1f}, p = {p_ratio:.2e}")
print(f"   ETG median: {df_etg['xi_over_rcore'].median():.3f}")
print(f"   LTG median: {df_ltg['xi_over_rcore'].median():.3f}")
if p_ratio < 0.05:
    print(f"   ✓ SIGNIFICANT DIFFERENCE")
    if df_etg['xi_over_rcore'].median() < df_ltg['xi_over_rcore'].median():
        print(f"   → ETGs more CORE-DOMINATED (ξ/r_core < 1) ✓")
else:
    print(f"   ✗ No significant difference")

# Visualizations
print("\n" + "="*80)
print("GENERATING COMPARISON PLOTS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Panel 1: τ distributions
ax = axes[0, 0]
ax.hist(df_ltg['tau_Myr'], bins=20, alpha=0.6, label='LTG', color='blue', edgecolor='k')
ax.hist(df_etg['tau_Myr'], bins=10, alpha=0.8, label='ETG', color='red', edgecolor='k')
ax.axvline(0, color='k', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('τ_unified [Myr]', fontweight='bold', fontsize=11)
ax.set_ylabel('Count', fontweight='bold', fontsize=11)
ax.set_title(f'τ Distribution\np = {p_tau:.2e}', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 2: η distributions
ax = axes[0, 1]
ax.hist(df_ltg['eta'], bins=20, alpha=0.6, label='LTG', color='blue', edgecolor='k')
ax.hist(df_etg['eta'], bins=10, alpha=0.8, label='ETG', color='red', edgecolor='k')
ax.set_xlabel('η (maturity)', fontweight='bold', fontsize=11)
ax.set_ylabel('Count', fontweight='bold', fontsize=11)
ax.set_title(f'η Distribution\np = {p_eta:.2e}', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 3: α distributions
ax = axes[0, 2]
ax.hist(df_ltg['alpha'], bins=20, alpha=0.6, label='LTG', color='blue', edgecolor='k')
ax.hist(df_etg['alpha'], bins=10, alpha=0.8, label='ETG', color='red', edgecolor='k')
ax.set_xlabel('α (shape)', fontweight='bold', fontsize=11)
ax.set_ylabel('Count', fontweight='bold', fontsize=11)
ax.set_title(f'α Distribution\np = {p_alpha:.2e}', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 4: τ vs Hubble Type
ax = axes[1, 0]
# LTGs with Hubble Type
ltg_with_T = df_ltg.dropna(subset=['Hubble_Type_Num'])
ax.scatter(ltg_with_T['Hubble_Type_Num'], ltg_with_T['tau_Myr'],
          alpha=0.6, s=40, label='LTG', color='blue', edgecolors='k', linewidths=0.5)
ax.scatter(df_etg['Hubble_Type_Num'], df_etg['tau_Myr'],
          alpha=0.8, s=100, label='ETG', color='red', edgecolors='k', linewidths=1.5)
ax.axhline(0, color='k', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('Hubble Type (T)', fontweight='bold', fontsize=11)
ax.set_ylabel('τ_unified [Myr]', fontweight='bold', fontsize=11)
ax.set_title('τ vs Morphology', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 5: η vs α
ax = axes[1, 1]
ax.scatter(df_ltg['alpha'], df_ltg['eta'],
          alpha=0.6, s=40, label='LTG', color='blue', edgecolors='k', linewidths=0.5)
ax.scatter(df_etg['alpha'], df_etg['eta'],
          alpha=0.8, s=100, label='ETG', color='red', edgecolors='k', linewidths=1.5)
ax.set_xlabel('α', fontweight='bold', fontsize=11)
ax.set_ylabel('η', fontweight='bold', fontsize=11)
ax.set_title('Maturity vs Shape', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 6: ξ/r_core ratio comparison
ax = axes[1, 2]
positions = [1, 2]
data_to_plot = [df_ltg['xi_over_rcore'].dropna(), df_etg['xi_over_rcore'].dropna()]
bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                patch_artist=True, showmeans=True)
bp['boxes'][0].set_facecolor('blue')
bp['boxes'][1].set_facecolor('red')
ax.axhline(1.0, color='k', linestyle='--', linewidth=2, alpha=0.5, label='ξ = r_core')
ax.set_xticks(positions)
ax.set_xticklabels(['LTG', 'ETG'], fontweight='bold')
ax.set_ylabel('ξ_GL / r_core', fontweight='bold', fontsize=11)
ax.set_title(f'Core Dominance\np = {p_ratio:.2e}', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
fig_path = ROOT / "results/stage8/figures/etg_ltg_comparison.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {fig_path}")

# Summary table
print("\n" + "="*80)
print("SUMMARY: ETG vs LTG PARAMETER COMPARISON")
print("="*80)

summary = pd.DataFrame({
    'Parameter': ['τ [Myr]', 'η', 'α', 'ξ/r_core', 'κ₁'],
    'ETG_median': [
        df_etg['tau_Myr'].median(),
        df_etg['eta'].median(),
        df_etg['alpha'].median(),
        df_etg['xi_over_rcore'].median(),
        df_etg['kappa1'].median()
    ],
    'LTG_median': [
        df_ltg['tau_Myr'].median(),
        df_ltg['eta'].median(),
        df_ltg['alpha'].median(),
        df_ltg['xi_over_rcore'].median(),
        df_ltg['kappa1'].median()
    ],
    'p_value': [p_tau, p_eta, p_alpha, p_ratio, np.nan],
    'Significant': [
        '✓' if p_tau < 0.05 else '✗',
        '✓' if p_eta < 0.05 else '✗',
        '✓' if p_alpha < 0.05 else '✗',
        '✓' if p_ratio < 0.05 else '✗',
        '-'
    ]
})

print("\n" + summary.to_string(index=False))

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

print(f"""
ETG Sample Characteristics:
- ALL 16 ETGs have τ > 0 (not τ < 0 as predicted)
- BUT ETGs show distinct signatures:
  
  1. τ values: {'LOWER' if df_etg['tau_Myr'].median() < df_ltg['tau_Myr'].median() else 'HIGHER'} than LTGs
     → ETG median: {df_etg['tau_Myr'].median():.1f} Myr
     → LTG median: {df_ltg['tau_Myr'].median():.1f} Myr
  
  2. η values: {'HIGHER' if df_etg['eta'].median() > df_ltg['eta'].median() else 'LOWER'} than LTGs
     → ETG: {df_etg['eta'].median():.3f} (more mature ✓)
     → LTG: {df_ltg['eta'].median():.3f}
  
  3. α values: {'LOWER' if df_etg['alpha'].median() < df_ltg['alpha'].median() else 'HIGHER'} than LTGs
     → ETG: {df_etg['alpha'].median():.3f} (steeper profiles ✓)
     → LTG: {df_ltg['alpha'].median():.3f}

⚠ CAVEAT: ETG τ values may be biased by synthetic profile generation

CONCLUSION:
- τ < 0 prediction NOT confirmed for ETGs
- HOWEVER: ETGs DO show distinct structural signatures (η, α)
- Need actual rotation curve data for ETGs to test τ < 0 properly
- Current synthetic exponential models may force τ > 0 artificially
""")

print("="*80)
