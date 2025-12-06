"""
extended_coherence_params.py
-----------------------------
Compute extended RD coherence parameters from galaxy master + TDGL fits.

Derived parameters:
    η (eta)       = xi_GL / r_core (dimensionless scale ratio)
    τ_unified     = sign(Δ) * sqrt(|Δ|) / V_max, where Δ = xi_GL² - r_core²

Purpose:
    Test correlations between RD-derived length scales and morphology (Hubble type).
    
Usage:
    Run from repository root: python scripts/analysis/extended_coherence_params.py
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.stats import spearmanr

# Get repository root (2 levels up from this script)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# 1. Load TDGL results
print("=== Extended Coherence Parameter Analysis ===")
print("\nLoading data...")
tdgl = pd.read_csv(REPO_ROOT / "data/results/sparc_spirals/tdgl_fits.csv")

tdgl = pd.read_csv(REPO_ROOT / "data/results/sparc_spirals/tdgl_fits.csv")
galaxy_props = pd.read_csv(REPO_ROOT / "data/results/sparc_spirals/galaxy_properties.csv")

print(f"Galaxies in properties: {len(galaxy_props)}")
print(f"Galaxies in TDGL: {len(tdgl)}")

df = galaxy_props.merge(tdgl[["Galaxy", "xi_GL", "r_core", "V0"]], on="Galaxy", how="inner")
print(f"Galaxies after merge: {len(df)}")

# 2. Compute basic quantities
xi = df["xi_GL"].values
rc = df["r_core"].values
# Use V0 from TDGL fits if V_max not available
Vmax = df["V0"].values if "V0" in df.columns else df.get("V_max", df["V0"]).values
T = df.get("Hubble_Type_Num", np.zeros(len(df))).values

delta = xi**2 - rc**2

eta = xi / rc
tau_unified = np.sign(delta) * np.sqrt(np.abs(delta)) / Vmax

df["eta"] = eta
df["tau_unified"] = tau_unified

# 3. Diagnostic output
print("\n---- Debug: η vs T ----")
print(f"N rows total: {len(df)}")
print(f"η nulls: {df['eta'].isna().sum()}")
if "Hubble_Type_Num" in df.columns:
    print(f"T nulls: {df['Hubble_Type_Num'].isna().sum()}")
else:
    print("Note: Hubble_Type_Num not available in dataset")

eta_array = df['eta'].to_numpy()
T_array = df['Hubble_Type_Num'].to_numpy()

finite_mask = np.isfinite(eta_array) & np.isfinite(T_array)
print(f"Finite pairs: {finite_mask.sum()}")

if finite_mask.sum() > 0:
    eta_finite = eta_array[finite_mask]
    T_finite = T_array[finite_mask]
    print(f"η unique (first 10): {np.unique(eta_finite)[:10]}")
    print(f"T unique (first 10): {np.unique(T_finite)[:10]}")
    print(f"η std: {np.nanstd(eta_array):.4f}, T std: {np.nanstd(T_array):.4f}")
    print(f"η range: [{np.nanmin(eta_array):.4f}, {np.nanmax(eta_array):.4f}]")
    print(f"T range: [{np.nanmin(T_array):.1f}, {np.nanmax(T_array):.1f}]")
else:
    print("WARNING: No finite pairs available for correlation!")

print("\n---- Debug: τ_unified vs T ----")
print(f"τ_unified nulls: {df['tau_unified'].isna().sum()}")
tau_array = df['tau_unified'].to_numpy()
finite_mask_tau = np.isfinite(tau_array) & np.isfinite(T_array)
print(f"Finite pairs: {finite_mask_tau.sum()}")
if finite_mask_tau.sum() > 0:
    print(f"τ std: {np.nanstd(tau_array):.4f}")
    print(f"τ range: [{np.nanmin(tau_array):.4f}, {np.nanmax(tau_array):.4f}]")

# 4. Statistical tests (with NaN handling)
print("\n=== Correlation Results ===")
try:
    rho_eta, p_eta = spearmanr(eta, T, nan_policy='omit')
    print(f"η vs T: ρ = {rho_eta:+.3f}, p = {p_eta:.3e}")
except Exception as e:
    print(f"η vs T correlation failed: {e}")

try:
    rho_tau, p_tau = spearmanr(tau_unified, T, nan_policy='omit')
    print(f"τ_unified vs T: ρ = {rho_tau:+.3f}, p = {p_tau:.3e}")
except Exception as e:
    print(f"τ_unified vs T correlation failed: {e}")

# 5. Save results
output_file = "results/stage8/extended_coherence_params.csv"
df[["Galaxy", "eta", "tau_unified", "Hubble_Type_Num"]].to_csv(output_file, index=False)
print(f"\n✓ Saved extended parameters → {output_file}")
print("=== DONE ===")
