"""
fit_tdgl.py

Fresh TDGL-style fits for SPARC rotation curves, using ONLY the new pipeline.

Input:
    results/stage1/sparc_cleaned.csv

Output:
    results/stage8/tdgl_results.csv

Model:
    V(r) = V0 * tanh( (r / xi_GL) ** alpha )

Where:
    V0      = asymptotic rotation velocity
    xi_GL   = GL coherence length / transition scale
    alpha   = non-linearity / inner shape parameter
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[2]  # .../RD_physics
SPARC_CLEAN_PATH = ROOT / "results" / "stage1" / "sparc_cleaned.csv"
OUT_DIR = ROOT / "results" / "stage8"
OUT_FILE = OUT_DIR / "tdgl_results.csv"


# -----------------------------------------------------------------------------
# TDGL-style rotation curve model
# -----------------------------------------------------------------------------

def tdgl_velocity(r: np.ndarray, V0: float, xi_GL: float, alpha: float) -> np.ndarray:
    """
    TDGL-style rotation curve:
        V(r) = V0 * tanh( (r / xi_GL) ** alpha )

    r      : radius [kpc]
    V0     : asymptotic velocity [km/s]
    xi_GL  : scale length [kpc]
    alpha  : shape parameter (inner slope / non-linearity)
    """
    r = np.asarray(r, dtype=float)
    x = r / np.maximum(xi_GL, 1e-4)  # avoid division by zero
    # Guard against negative / zero xi_GL
    x = np.clip(x, 0.0, np.inf)

    return V0 * np.tanh(np.power(x, alpha))


def compute_core_radius(xi_GL: float, alpha: float, frac: float = 0.5) -> float:
    """
    Define a "core radius" as the radius where V(r) reaches frac * V0.

    For V(r)/V0 = tanh((r/xi)^alpha) = frac:

        (r_core/xi)^alpha = arctanh(frac)
        r_core = xi * [arctanh(frac)]^(1/alpha)
    """
    if xi_GL <= 0 or alpha <= 0:
        return np.nan

    try:
        at = np.arctanh(frac)  # arctanh(0.5) ~ 0.5493
        return float(xi_GL * at ** (1.0 / alpha))
    except Exception:
        return np.nan


@dataclass
class FitResult:
    Galaxy: str
    n_points: int
    V0: float | None = None
    xi_GL: float | None = None
    alpha: float | None = None
    chi2_red: float | None = None
    rms_resid: float | None = None
    r_core: float | None = None
    success: bool = False
    note: str | None = None


# -----------------------------------------------------------------------------
# Fitting per galaxy
# -----------------------------------------------------------------------------

def fit_galaxy(group: pd.DataFrame) -> FitResult:
    name = str(group["Galaxy"].iloc[0])
    # Require minimum number of points to fit anything meaningful
    if len(group) < 6:
        return FitResult(
            Galaxy=name,
            n_points=len(group),
            success=False,
            note="Too few points for TDGL fit (<6).",
        )

    # Extract basic arrays
    r = group["R"].to_numpy(dtype=float)
    if "V_obs" in group.columns:
        v = group["V_obs"].to_numpy(dtype=float)
    elif "V" in group.columns:
        v = group["V"].to_numpy(dtype=float)
    else:
        return FitResult(
            Galaxy=name,
            n_points=len(group),
            success=False,
            note="No V_obs or V column in SPARC cleaned file.",
        )

    # Optional errors
    if "e_V" in group.columns:
        sigma = group["e_V"].to_numpy(dtype=float)
        # Avoid zero errors
        sigma[sigma <= 0] = np.nan
    else:
        sigma = None

    # Clean NaN/inf
    mask = np.isfinite(r) & np.isfinite(v)
    if sigma is not None:
        mask = mask & np.isfinite(sigma)
    r = r[mask]
    v = v[mask]
    if sigma is not None:
        sigma = sigma[mask]

    if len(r) < 6:
        return FitResult(
            Galaxy=name,
            n_points=len(group),
            success=False,
            note="Too few valid points after cleaning.",
        )

    # Sort by radius
    order = np.argsort(r)
    r = r[order]
    v = v[order]
    if sigma is not None:
        sigma = sigma[order]

    # Initial guesses
    V0_guess = float(np.nanmax(v))
    # rough guess for transition scale ~ mid-radius
    xi_guess = float(0.5 * (np.nanmin(r) + np.nanmax(r)))
    alpha_guess = 2.0

    p0 = [V0_guess, xi_guess, alpha_guess]
    # Bounds: positive V0, xi_GL, alpha not insane
    bounds = (
        [1.0, 0.01, 0.2],  # lower
        [500.0, 100.0, 8.0],  # upper
    )

    try:
        if sigma is not None and np.all(np.isfinite(sigma)):
            popt, pcov = curve_fit(
                tdgl_velocity,
                r,
                v,
                p0=p0,
                sigma=sigma,
                absolute_sigma=True,
                bounds=bounds,
                maxfev=20000,
            )
        else:
            popt, pcov = curve_fit(
                tdgl_velocity,
                r,
                v,
                p0=p0,
                bounds=bounds,
                maxfev=20000,
            )

        V0_fit, xi_fit, alpha_fit = [float(x) for x in popt]

        # Parameter validation
        param_warnings = []
        if xi_fit < 0.1:
            param_warnings.append(f"xi_GL={xi_fit:.4f} kpc is very small")
        if xi_fit > 50.0:
            param_warnings.append(f"xi_GL={xi_fit:.4f} kpc is very large")
        if alpha_fit < 0.3:
            param_warnings.append(f"alpha={alpha_fit:.4f} is very shallow")
        if alpha_fit > 4.0:
            param_warnings.append(f"alpha={alpha_fit:.4f} is very steep")
        
        # Residuals and goodness of fit
        v_model = tdgl_velocity(r, V0_fit, xi_fit, alpha_fit)
        resid = v - v_model

        if sigma is not None and np.all(np.isfinite(sigma)):
            chi2 = np.sum(((resid) / sigma) ** 2)
        else:
            # If no errors, treat sigma = 1
            chi2 = np.sum(resid**2)

        dof = max(len(v) - len(popt), 1)
        chi2_red = chi2 / dof
        rms_resid = float(np.sqrt(np.mean(resid**2)))

        r_core = compute_core_radius(xi_fit, alpha_fit, frac=0.5)

        # Add warnings to note if present
        note_str = None
        if param_warnings:
            note_str = "WARNING: " + "; ".join(param_warnings)

        return FitResult(
            Galaxy=name,
            n_points=len(r),
            V0=V0_fit,
            xi_GL=xi_fit,
            alpha=alpha_fit,
            chi2_red=float(chi2_red),
            rms_resid=rms_resid,
            r_core=r_core,
            success=True,
            note=note_str,
        )

    except Exception as e:
        return FitResult(
            Galaxy=name,
            n_points=len(r),
            success=False,
            note=f"Fit failed: {type(e).__name__}: {e}",
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    print("=== TDGL-style rotation curve fitting ===")
    print(f"ROOT: {ROOT}")
    print(f"Reading SPARC cleaned file: {SPARC_CLEAN_PATH}")

    if not SPARC_CLEAN_PATH.exists():
        raise FileNotFoundError(f"SPARC cleaned file not found: {SPARC_CLEAN_PATH}")

    df = pd.read_csv(SPARC_CLEAN_PATH)
    if "Galaxy" not in df.columns:
        raise ValueError("SPARC cleaned file must contain a 'Galaxy' column.")

    # Group by galaxy
    groups = df.groupby("Galaxy")
    results: List[Dict[str, Any]] = []

    for name, g in groups:
        res = fit_galaxy(g)
        results.append(asdict(res))

    # Build dataframe
    df_out = pd.DataFrame(results)

    # Ensure output dir exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_FILE, index=False)

    print(f"Fitted galaxies: {len(df_out)}")
    print(f"Successful fits: {df_out['success'].sum()} / {len(df_out)}")
    print(f"Saved TDGL results → {OUT_FILE}")

    # Quick sanity summary
    good = df_out[df_out["success"]]
    if not good.empty:
        print("\nSummary of successful fits (alpha, xi_GL):")
        print(good[["alpha", "xi_GL", "r_core"]].describe())
    else:
        print("\nNo successful fits; check model or bounds.")


if __name__ == "__main__":
    main()
