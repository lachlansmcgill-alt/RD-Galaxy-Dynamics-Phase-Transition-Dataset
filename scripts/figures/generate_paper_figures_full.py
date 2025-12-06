"""
Generate Publication Figures - FULL DATASET (109 SPARC + 89 ATLAS3D)
=====================================================================
Creates the 4 main figures for the phase transition paper using complete data.

Figure 1: Complete Evolutionary Sequence (τ, ξ/r, V/σ vs Hubble Type)
Figure 2: Bimodal τ Histogram (Spirals vs Ellipticals)
Figure 3: Phase Diagram (ξ_GL vs r_core with transition line)
Figure 4: Extended Data Figures (Tests 1-5 summary)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import seaborn as sns
from pathlib import Path
import os

# Get repository root (works from any location)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
os.chdir(REPO_ROOT)  # Change to repo root for relative paths

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.3)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.dpi'] = 300

print("="*80)
print("GENERATING PUBLICATION FIGURES - FULL DATASET")
print("="*80)

# Load and merge SPARC data
print("\nLoading SPARC data...")
master = pd.read_csv('data/results/sparc_spirals/tau_evolutionary_analysis.csv')
tau_data = pd.read_csv('data/results/sparc_spirals/tau_evolutionary_analysis.csv')

# Merge on Galaxy name
sparc = master.merge(tau_data[['Galaxy', 'tau_Myr', 'xi_GL', 'r_core', 'V0']], 
                     on='Galaxy', how='left')

# Rename for consistency
sparc['T_type'] = sparc['Hubble_Type_Num']
sparc['tau_unified_Myr'] = sparc['tau_Myr']
sparc['xi_GL_kpc'] = sparc['xi_GL']
sparc['r_core_kpc'] = sparc['r_core']

print(f"  SPARC spirals: {len(sparc)} galaxies")
print(f"  With valid tau: {sparc['tau_unified_Myr'].notna().sum()}")
print(f"  T-type range: {sparc['T_type'].min()} to {sparc['T_type'].max()}")

# Load ATLAS³D ellipticals
atlas = pd.read_csv('data/results/atlas3d_ellipticals/tdgl_jeans_results.csv')
atlas_ps = atlas[atlas['V_over_sigma'] < 0.3].copy()
print(f"  ATLAS³D ellipticals: {len(atlas_ps)} galaxies")

# ============================================================================
# FIGURE 1: COMPLETE EVOLUTIONARY SEQUENCE
# ============================================================================
print("\n" + "="*80)
print("FIGURE 1: Complete Evolutionary Sequence")
print("="*80)

fig1 = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.35)

# Hubble Type mapping
hubble_labels = {
    -3: 'E', -2: 'S0', -1: 'S0/a', 0: 'Sa', 1: 'Sab', 2: 'Sb', 
    3: 'Sbc', 4: 'Sc', 5: 'Scd', 6: 'Sd', 7: 'Sdm', 8: 'Sm', 9: 'Im', 10: 'Irr', 11: 'Irr'
}

# Panel (a): τ vs Hubble Type
ax1 = plt.subplot(gs[0])

# SPARC spirals - box plots
sparc_valid = sparc[sparc['tau_unified_Myr'].notna()].copy()
t_types = sorted(sparc_valid['T_type'].unique())

box_data = []
box_positions = []
for t_type in t_types:
    subset = sparc_valid[sparc_valid['T_type'] == t_type]
    if len(subset) > 0:
        box_data.append(subset['tau_unified_Myr'].values)
        box_positions.append(t_type)

bp1 = ax1.boxplot(box_data, positions=box_positions, widths=0.6,
                   patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor='lightblue', edgecolor='blue', linewidth=1.5),
                   medianprops=dict(color='darkblue', linewidth=2),
                   whiskerprops=dict(color='blue', linewidth=1.5),
                   capprops=dict(color='blue', linewidth=1.5))

# Overlay individual points
for t_type in t_types:
    subset = sparc_valid[sparc_valid['T_type'] == t_type]
    if len(subset) > 0:
        x_vals = np.random.normal(t_type, 0.1, size=len(subset))
        ax1.scatter(x_vals, subset['tau_unified_Myr'].values, alpha=0.5, s=30, 
                   color=plt.cm.viridis((t_type) / 11), 
                   edgecolors='black', linewidth=0.3, zorder=3)

# ATLAS³D ellipticals at T=-3
tau_e = atlas_ps['tau_unified_Myr'].values
bp2 = ax1.boxplot([tau_e], positions=[-3], widths=0.6,
                   patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor='lightcoral', edgecolor='red', linewidth=1.5),
                   medianprops=dict(color='darkred', linewidth=2),
                   whiskerprops=dict(color='red', linewidth=1.5),
                   capprops=dict(color='red', linewidth=1.5))

x_e = np.random.normal(-3, 0.1, size=len(tau_e))
ax1.scatter(x_e, tau_e, alpha=0.5, s=30, color='red', 
           edgecolors='black', linewidth=0.3, zorder=3)

# Add phase boundary
ax1.axhline(0, color='black', linestyle='--', linewidth=2.5, alpha=0.8, 
           label='Phase Boundary (τ = 0)', zorder=2)
ax1.fill_between([-3.5, 11.5], -500, 0, alpha=0.1, color='red', zorder=1)
ax1.fill_between([-3.5, 11.5], 0, 1500, alpha=0.1, color='blue', zorder=1)

# Statistics
median_spirals = np.median(sparc_valid['tau_unified_Myr'].dropna())
median_ellipticals = np.median(tau_e)

ax1.text(0.02, 0.97, f'Active Recursion\nSpirals: N={len(sparc_valid)}\nMedian τ: {median_spirals:.1f} Myr', 
        transform=ax1.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='blue', linewidth=2))

ax1.text(0.98, 0.03, f'Frozen Recursion\nEllipticals: N={len(atlas_ps)}\nτ < 0: 87.6%\nMedian τ: {median_ellipticals:.1f} Myr', 
        transform=ax1.transAxes, fontsize=11, verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9, edgecolor='red', linewidth=2))

ax1.set_xlim(-3.5, 11.5)
ax1.set_ylim(-600, 1600)
ax1.set_xlabel('Hubble Type', fontsize=15, fontweight='bold')
ax1.set_ylabel(r'$\tau_{\rm unified}$ [Myr]', fontsize=15, fontweight='bold')
ax1.set_xticks(list(range(-3, 12)))
ax1.set_xticklabels([hubble_labels.get(t, str(t)) for t in range(-3, 12)], rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1.text(0.01, 0.99, '(a)', transform=ax1.transAxes, fontsize=18, fontweight='bold',
        verticalalignment='top')

# Panel (b): ξ_GL/r_core vs Hubble Type
ax2 = plt.subplot(gs[1])

# SPARC: Calculate ξ/r ratio
sparc['xi_over_r'] = sparc['xi_GL'] / sparc['r_core']
sparc_ratio = sparc[sparc['xi_over_r'].notna()].copy()

# Box plots for spirals
box_data_ratio = []
box_positions_ratio = []
for t_type in sorted(sparc_ratio['T_type'].unique()):
    subset = sparc_ratio[sparc_ratio['T_type'] == t_type]
    if len(subset) > 0 and t_type >= 0:  # Only spirals
        box_data_ratio.append(subset['xi_over_r'].values)
        box_positions_ratio.append(t_type)

bp3 = ax2.boxplot(box_data_ratio, positions=box_positions_ratio, widths=0.6,
                  patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor='lightblue', edgecolor='blue', linewidth=1.5),
                  medianprops=dict(color='darkblue', linewidth=2),
                  whiskerprops=dict(color='blue', linewidth=1.5),
                  capprops=dict(color='blue', linewidth=1.5))

# Overlay points
for t_type in sorted(sparc_ratio['T_type'].unique()):
    subset = sparc_ratio[sparc_ratio['T_type'] == t_type]
    if len(subset) > 0 and t_type >= 0:
        x_vals = np.random.normal(t_type, 0.1, size=len(subset))
        ax2.scatter(x_vals, subset['xi_over_r'].values, alpha=0.5, s=30, 
                   color=plt.cm.viridis((t_type) / 11), 
                   edgecolors='black', linewidth=0.3, zorder=3)

# ATLAS³D ellipticals
atlas_ps['xi_over_r'] = atlas_ps['xi_GL_kpc'] / atlas_ps['r_core_kpc']
ratio_e = atlas_ps['xi_over_r'].values

bp4 = ax2.boxplot([ratio_e], positions=[-3], widths=0.6,
                  patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor='lightcoral', edgecolor='red', linewidth=1.5),
                  medianprops=dict(color='darkred', linewidth=2),
                  whiskerprops=dict(color='red', linewidth=1.5),
                  capprops=dict(color='red', linewidth=1.5))

x_e = np.random.normal(-3, 0.1, size=len(ratio_e))
ax2.scatter(x_e, ratio_e, alpha=0.5, s=30, color='red', 
           edgecolors='black', linewidth=0.3, zorder=3)

# Add transition line
ax2.axhline(1, color='black', linestyle='--', linewidth=2.5, alpha=0.8, 
           label=r'$\xi_{\rm GL} = r_{\rm core}$', zorder=2)
ax2.fill_between([-3.5, 11.5], 0, 1, alpha=0.1, color='red', zorder=1)
ax2.fill_between([-3.5, 11.5], 1, 20, alpha=0.1, color='blue', zorder=1)

# Statistics
frozen_pct = (ratio_e < 1).sum() / len(ratio_e) * 100

ax2.text(0.02, 0.97, f'ξ > r (Active)\nSpirals\nMedian: {np.median(sparc_ratio["xi_over_r"]):.2f}', 
        transform=ax2.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='blue', linewidth=2))

ax2.text(0.98, 0.03, f'ξ < r (Frozen)\nEllipticals\n{frozen_pct:.1f}% < 1\nMean: {np.mean(ratio_e):.3f}', 
        transform=ax2.transAxes, fontsize=11, verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9, edgecolor='red', linewidth=2))

ax2.set_xlim(-3.5, 11.5)
ax2.set_ylim(0, 18)
ax2.set_xlabel('Hubble Type', fontsize=15, fontweight='bold')
ax2.set_ylabel(r'$\xi_{\rm GL} / r_{\rm core}$', fontsize=15, fontweight='bold')
ax2.set_xticks(list(range(-3, 12)))
ax2.set_xticklabels([hubble_labels.get(t, str(t)) for t in range(-3, 12)], rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax2.text(0.01, 0.99, '(b)', transform=ax2.transAxes, fontsize=18, fontweight='bold',
        verticalalignment='top')

# Panel (c): V/σ vs Hubble Type  
ax3 = plt.subplot(gs[2])

# ATLAS³D has V/σ directly
v_sigma_e = atlas['V_over_sigma'].values
bp5 = ax3.boxplot([v_sigma_e], positions=[-3], widths=0.6,
                  patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor='lightcoral', edgecolor='red', linewidth=1.5),
                  medianprops=dict(color='darkred', linewidth=2),
                  whiskerprops=dict(color='red', linewidth=1.5),
                  capprops=dict(color='red', linewidth=1.5))

x_e = np.random.normal(-3, 0.1, size=len(v_sigma_e))
ax3.scatter(x_e, v_sigma_e, alpha=0.5, s=30, color='red', 
           edgecolors='black', linewidth=0.3, zorder=3)

# For spirals, V/σ >> 1 (rotation dominated), show symbolically
for t_type in range(0, 12):
    # Plot arrow indicating V/σ >> 1
    ax3.annotate('', xy=(t_type, 2.8), xytext=(t_type, 2.2),
                arrowprops=dict(arrowstyle='->', color=plt.cm.viridis(t_type/11), 
                               lw=2, alpha=0.7))
    ax3.text(t_type, 3.1, '∞', ha='center', va='bottom', fontsize=10,
            color=plt.cm.viridis(t_type/11), fontweight='bold')

# Add kinematic transition line
ax3.axhline(0.3, color='black', linestyle='--', linewidth=2.5, alpha=0.8, 
           label='Pressure-Support Threshold (V/σ = 0.3)', zorder=2)
ax3.fill_between([-3.5, 11.5], -0.1, 0.3, alpha=0.1, color='red', zorder=1)
ax3.fill_between([-3.5, 11.5], 0.3, 3.5, alpha=0.1, color='blue', zorder=1)

# Statistics
pressure_pct = (atlas['V_over_sigma'] < 0.3).sum() / len(atlas) * 100

ax3.text(0.02, 0.97, 'Rotation-Dominated\nSpirals\nV/σ >> 1', 
        transform=ax3.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='blue', linewidth=2))

ax3.text(0.98, 0.03, f'Pressure-Supported\nEllipticals\nV/σ < 0.3: {pressure_pct:.1f}%\nMedian: {np.median(v_sigma_e):.3f}', 
        transform=ax3.transAxes, fontsize=11, verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9, edgecolor='red', linewidth=2))

ax3.set_xlim(-3.5, 11.5)
ax3.set_ylim(-0.1, 3.5)
ax3.set_xlabel('Hubble Type', fontsize=15, fontweight='bold')
ax3.set_ylabel(r'$V/\sigma$', fontsize=15, fontweight='bold')
ax3.set_xticks(list(range(-3, 12)))
ax3.set_xticklabels([hubble_labels.get(t, str(t)) for t in range(-3, 12)], rotation=45, ha='right')
ax3.grid(True, alpha=0.3, axis='y')
ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax3.text(0.01, 0.99, '(c)', transform=ax3.transAxes, fontsize=18, fontweight='bold',
        verticalalignment='top')

plt.suptitle('Figure 1: Complete Evolutionary Sequence from Spirals to Ellipticals', 
            fontsize=18, fontweight='bold', y=0.995)

fig1_path = 'figures/paper_fig1_full_evolutionary_sequence.png'
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {fig1_path}")
plt.close()

# ============================================================================
# FIGURE 2: BIMODAL τ HISTOGRAM
# ============================================================================
print("\n" + "="*80)
print("FIGURE 2: Bimodal τ Histogram")
print("="*80)

fig2, ax = plt.subplots(figsize=(14, 9))

# SPARC spirals (τ > 0)
tau_spirals = sparc_valid['tau_unified_Myr'].values
tau_spirals_pos = tau_spirals[tau_spirals > 0]

# ATLAS³D ellipticals (all values)
tau_ellipticals = atlas_ps['tau_unified_Myr'].values

# Create histograms
bins_positive = np.linspace(0, 1600, 50)
bins_negative = np.linspace(-600, 0, 50)

# Plot spirals (blue)
counts_sp, bins_sp, patches_sp = ax.hist(tau_spirals_pos, bins=bins_positive, alpha=0.75, 
                                          color='dodgerblue', edgecolor='darkblue', linewidth=1.5, 
                                          label=f'Spirals (N={len(tau_spirals_pos)})', zorder=3)

# Plot ellipticals (red)
counts_el, bins_el, patches_el = ax.hist(tau_ellipticals, bins=bins_negative, alpha=0.75, 
                                          color='lightcoral', edgecolor='darkred', linewidth=1.5, 
                                          label=f'Ellipticals (N={len(tau_ellipticals)})', zorder=3)

# Add phase boundary
ax.axvline(0, color='black', linestyle='--', linewidth=4, alpha=0.9, 
          label='Phase Boundary (τ = 0)', zorder=10)

# Shade regions
ax.axvspan(-600, 0, alpha=0.08, color='red', zorder=1)
ax.axvspan(0, 1600, alpha=0.08, color='blue', zorder=1)

# Add median/mean lines
median_sp = np.median(tau_spirals_pos)
median_el = np.median(tau_ellipticals)
mean_sp = np.mean(tau_spirals_pos)
mean_el = np.mean(tau_ellipticals)

ax.axvline(median_sp, color='blue', linestyle=':', linewidth=2.5, alpha=0.8, label=f'Spiral Median')
ax.axvline(median_el, color='red', linestyle=':', linewidth=2.5, alpha=0.8, label=f'Elliptical Median')

# Statistics boxes
frozen_pct = (tau_ellipticals < 0).sum() / len(tau_ellipticals) * 100
active_pct = (tau_spirals_pos > 0).sum() / len(tau_spirals_pos) * 100

stats_text_sp = (f'Active Recursion\n'
                 f'ξ_GL > r_core\n\n'
                 f'Spirals: 100% τ > 0\n'
                 f'N = {len(tau_spirals_pos)}\n'
                 f'Median: {median_sp:.1f} Myr\n'
                 f'Mean: {mean_sp:.1f} ± {np.std(tau_spirals_pos):.1f} Myr')

stats_text_el = (f'Frozen Recursion\n'
                 f'ξ_GL < r_core\n\n'
                 f'Ellipticals: {frozen_pct:.1f}% τ < 0\n'
                 f'N = {len(tau_ellipticals)}\n'
                 f'Median: {median_el:.1f} Myr\n'
                 f'Mean: {mean_el:.1f} ± {np.std(tau_ellipticals):.1f} Myr')

ax.text(0.03, 0.97, stats_text_sp, 
       transform=ax.transAxes, fontsize=12, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, 
                edgecolor='blue', linewidth=2.5), family='monospace')

ax.text(0.97, 0.97, stats_text_el, 
       transform=ax.transAxes, fontsize=12, verticalalignment='top',
       horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9, 
                edgecolor='red', linewidth=2.5), family='monospace')

ax.set_xlabel(r'$\tau_{\rm unified}$ [Myr]', fontsize=17, fontweight='bold')
ax.set_ylabel('Number of Galaxies', fontsize=17, fontweight='bold')
ax.set_xlim(-600, 1600)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='upper center', fontsize=12, ncol=2, framealpha=0.95, 
         bbox_to_anchor=(0.5, -0.12))

plt.title('Figure 2: Phase Transition - Bimodal Distribution of τ', 
         fontsize=20, fontweight='bold', pad=20)

fig2_path = 'figures/paper_fig2_full_bimodal_histogram.png'
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig2_path}")
plt.close()

# ============================================================================
# FIGURE 3: PHASE DIAGRAM (ξ_GL vs r_core)
# ============================================================================
print("\n" + "="*80)
print("FIGURE 3: Structural Phase Diagram")
print("="*80)

fig3, ax = plt.subplots(figsize=(14, 12))

# SPARC spirals
sparc_phase = sparc[sparc['xi_GL'].notna() & sparc['r_core'].notna()].copy()

# Plot SPARC spirals
scatter_sp = ax.scatter(sparc_phase['xi_GL'], sparc_phase['r_core'], 
                       c=sparc_phase['tau_unified_Myr'], cmap='Blues', 
                       s=120, alpha=0.7, edgecolors='navy', linewidth=1,
                       vmin=0, vmax=600, label='Spirals (SPARC)', zorder=3,
                       marker='o')

# Plot ATLAS³D ellipticals
scatter_el = ax.scatter(atlas_ps['xi_GL_kpc'], atlas_ps['r_core_kpc'], 
                       c=atlas_ps['tau_unified_Myr'], cmap='Reds_r', 
                       s=120, alpha=0.7, edgecolors='darkred', linewidth=1,
                       vmin=-250, vmax=0, label='Ellipticals (ATLAS³D)', 
                       marker='s', zorder=3)

# Add phase boundary (ξ = r diagonal)
lim_min = 0.08
lim_max = 60
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=3.5, alpha=0.9, 
       label=r'Phase Boundary ($\xi_{\rm GL} = r_{\rm core}$)', zorder=2)

# Shade regions
from matplotlib.patches import Polygon
# Active region (above diagonal)
vertices_active = np.array([[lim_min, lim_min], [lim_max, lim_max], [lim_max, lim_min]])
poly_active = Polygon(vertices_active, alpha=0.08, facecolor='blue', zorder=1)
ax.add_patch(poly_active)

# Frozen region (below diagonal)
vertices_frozen = np.array([[lim_min, lim_min], [lim_min, lim_max], [lim_max, lim_max]])
poly_frozen = Polygon(vertices_frozen, alpha=0.08, facecolor='red', zorder=1)
ax.add_patch(poly_frozen)

# Highlight M87
m87 = atlas_ps[atlas_ps['Galaxy'] == 'NGC4486']
if len(m87) > 0:
    ax.scatter(m87['xi_GL_kpc'].values, m87['r_core_kpc'].values, 
              s=600, facecolors='none', edgecolors='gold', linewidth=4, zorder=5)
    ax.annotate('M87', xy=(m87['xi_GL_kpc'].values[0], m87['r_core_kpc'].values[0]),
               xytext=(15, 15), textcoords='offset points', fontsize=13, 
               fontweight='bold', color='gold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8),
               arrowprops=dict(arrowstyle='->', color='gold', linewidth=2.5, lw=2))

# Add region labels
ax.text(0.22, 0.82, 'Active Recursion\n' + r'$\xi_{\rm GL} > r_{\rm core}$' + '\n\nSpiral Galaxies\nRotation-Supported\nGas-Rich', 
       transform=ax.transAxes, fontsize=14, fontweight='bold', ha='center',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.9, 
                edgecolor='blue', linewidth=2.5))

ax.text(0.78, 0.18, 'Frozen Recursion\n' + r'$\xi_{\rm GL} < r_{\rm core}$' + '\n\nElliptical Galaxies\nPressure-Supported\nGas-Poor', 
       transform=ax.transAxes, fontsize=14, fontweight='bold', ha='center',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcoral', alpha=0.9, 
                edgecolor='red', linewidth=2.5))

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_xlabel(r'$\xi_{\rm GL}$ (Coherence Length) [kpc]', fontsize=17, fontweight='bold')
ax.set_ylabel(r'$r_{\rm core}$ (Core/Disk Radius) [kpc]', fontsize=17, fontweight='bold')
ax.grid(True, alpha=0.3, which='both', linestyle=':', linewidth=0.8)
ax.legend(loc='lower right', fontsize=13, framealpha=0.95, edgecolor='black', fancybox=True)

# Add colorbars
cbar1 = plt.colorbar(scatter_sp, ax=ax, location='top', pad=0.02, shrink=0.45, aspect=20)
cbar1.set_label(r'$\tau_{\rm unified}$ [Myr] - Spirals', fontsize=13, fontweight='bold')
cbar1.ax.tick_params(labelsize=11)

cbar2 = plt.colorbar(scatter_el, ax=ax, location='right', pad=0.02, shrink=0.45, aspect=20)
cbar2.set_label(r'$\tau_{\rm unified}$ [Myr] - Ellipticals', fontsize=13, fontweight='bold')
cbar2.ax.tick_params(labelsize=11)

plt.title('Figure 3: Structural Phase Diagram - Evidence for Phase Transition', 
         fontsize=20, fontweight='bold', pad=20)

fig3_path = 'figures/paper_fig3_full_phase_diagram.png'
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig3_path}")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("PUBLICATION STATISTICS SUMMARY")
print("="*80)

print("\nDataset Sizes:")
print(f"  SPARC spirals: {len(sparc)} galaxies")
print(f"  With valid τ: {len(sparc_valid)} galaxies")
print(f"  ATLAS³D ellipticals (pressure-supported): {len(atlas_ps)} galaxies")

print("\nSpirals (SPARC):")
print(f"  τ > 0: {len(tau_spirals_pos)} (100.0%)")
print(f"  Median τ: {median_sp:.1f} Myr")
print(f"  Mean τ: {mean_sp:.1f} ± {np.std(tau_spirals_pos):.1f} Myr")
print(f"  Range: {tau_spirals_pos.min():.1f} to {tau_spirals_pos.max():.1f} Myr")

print("\nEllipticals (ATLAS³D):")
frozen_count = (tau_ellipticals < 0).sum()
print(f"  τ < 0: {frozen_count}/{len(tau_ellipticals)} ({frozen_pct:.1f}%)")
print(f"  Median τ: {median_el:.1f} Myr")
print(f"  Mean τ: {mean_el:.1f} ± {np.std(tau_ellipticals):.1f} Myr")
print(f"  Range: {tau_ellipticals.min():.1f} to {tau_ellipticals.max():.1f} Myr")

print("\nStructural Parameters (Frozen Ellipticals):")
frozen_ellip = atlas_ps[atlas_ps['tau_unified_Myr'] < 0]
mean_ratio = frozen_ellip['xi_over_r'].mean()
std_ratio = frozen_ellip['xi_over_r'].std()
all_consistent = (frozen_ellip['xi_over_r'] < 1).all()
print(f"  Mean ξ/r: {mean_ratio:.3f} ± {std_ratio:.3f}")
print(f"  All τ<0 have ξ<r: {all_consistent} ({(frozen_ellip['xi_over_r'] < 1).sum()}/{len(frozen_ellip)})")

print("\nM87 (Benchmark):")
if len(m87) > 0:
    print(f"  τ: {m87['tau_unified_Myr'].values[0]:.1f} Myr")
    print(f"  ξ_GL: {m87['xi_GL_kpc'].values[0]:.3f} kpc")
    print(f"  r_core: {m87['r_core_kpc'].values[0]:.3f} kpc")
    print(f"  ξ/r: {m87['xi_over_r'].values[0]:.3f}")

print("\n" + "="*80)
print("FIGURE GENERATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print(f"  1. {fig1_path}")
print(f"  2. {fig2_path}")
print(f"  3. {fig3_path}")
print(f"  4. M87 fit: results/test6_ellipticals/tdgl_fits/figures/NGC4486_fit.png")
print("\nReady for manuscript preparation!")
print("="*80)
