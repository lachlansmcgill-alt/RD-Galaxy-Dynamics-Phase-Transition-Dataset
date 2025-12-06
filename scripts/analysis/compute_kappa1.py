"""
compute_kappa1.py
-----------------
Stage 4: Compute a rotation-curve coherence metric κ₁ for each SPARC galaxy.

Input:
    RD_physics/results/stage1/sparc_cleaned.csv

Output:
    RD_physics/results/stage4/sparc_kappa1.csv

Definition:
    For each galaxy:
        - Take (R, V) points
        - Normalise radius x = R / R_max
        - Fit a 3rd-degree polynomial V_fit(x)
        - Compute RMS_resid = sqrt(mean( (V - V_fit)^2 ))
        - Let V_scale = max(V)
        - RMS_frac = RMS_resid / V_scale
        - κ1 = max(0, 1 - RMS_frac)

    Interpretation:
        κ1 ~ 1   → very smooth/coherent rotation curve
        κ1 ~ 0   → highly structured / rippled / incoherent curve
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

SPARC_CLEAN = os.path.join(ROOT, "results", "stage1", "sparc_cleaned.csv")
OUTPUT_DIR = os.path.join(ROOT, "results", "stage4")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sparc_kappa1.csv")


# ---------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------

POSSIBLE_R_COLS = [
    "R", "R_kpc", "Radius_kpc", "Rgal", "Radius"
]

POSSIBLE_V_COLS = [
    "V", "Vobs", "V_obs", "V_obs_kms", "Vrot", "V_rot"
]


def detect_column(df: pd.DataFrame, candidates: list, label: str) -> str:
    """
    Try to find a column in `df` matching any of the `candidates`.
    Raise a clear error if none are found.
    """
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(
        f"Could not find a suitable {label} column. "
        f"Tried {candidates}, available columns are: {sorted(df.columns)}"
    )


# ---------------------------------------------------------
# Core per-galaxy computation
# ---------------------------------------------------------

def compute_kappa1_for_galaxy(
    R: np.ndarray,
    V: np.ndarray,
    poly_deg: int = 3,
    min_points: int = 6
) -> Optional[Tuple[float, float, float]]:
    """
    Compute κ1 for a single galaxy given its radii R (kpc) and velocities V (km/s).

    Returns:
        (kappa1, rms_resid, rms_frac)
        or None if galaxy is not usable.
    """
    if len(R) < min_points or len(V) < min_points:
        return None

    # Basic sanity: finite, positive
    mask = np.isfinite(R) & np.isfinite(V) & (R > 0) & (V > 0)
    R = R[mask]
    V = V[mask]

    if len(R) < min_points:
        return None

    # Sort by radius
    order = np.argsort(R)
    R = R[order]
    V = V[order]

    R_max = np.max(R)
    V_scale = np.max(V)

    if R_max <= 0 or V_scale <= 0:
        return None

    # Normalised radius
    x = R / R_max

    # Fit polynomial to V(x)
    try:
        coeffs = np.polyfit(x, V, deg=poly_deg)
        V_fit = np.polyval(coeffs, x)
    except np.linalg.LinAlgError:
        # Singular matrix or numerical issue
        return None

    # Residuals and RMS
    resid = V - V_fit
    rms_resid = float(np.sqrt(np.mean(resid ** 2)))
    rms_frac = float(rms_resid / V_scale)

    # κ1 as 1 - fractional RMS, clipped at [0,1]
    kappa1 = 1.0 - rms_frac
    if kappa1 < 0.0:
        kappa1 = 0.0
    if kappa1 > 1.0:
        kappa1 = 1.0

    return kappa1, rms_resid, rms_frac


# ---------------------------------------------------------
# Driver
# ---------------------------------------------------------

def main():
    print("=== Stage 4: Compute κ1 from SPARC rotation curves ===")
    print(f"ROOT: {ROOT}")
    print(f"Reading cleaned SPARC file: {SPARC_CLEAN}")

    if not os.path.exists(SPARC_CLEAN):
        raise FileNotFoundError(f"SPARC cleaned file not found at: {SPARC_CLEAN}")

    df = pd.read_csv(SPARC_CLEAN)

    # Detect Galaxy, R, V columns
    if "Galaxy" not in df.columns:
        raise ValueError(
            f"Expected a 'Galaxy' column in {SPARC_CLEAN}, "
            f"found: {sorted(df.columns)}"
        )

    # Auto-detect radius and velocity columns
    R_col = detect_column(df, POSSIBLE_R_COLS, "radius (R)")
    V_col = detect_column(df, POSSIBLE_V_COLS, "velocity (V)")

    print(f"Using radius column:   {R_col}")
    print(f"Using velocity column: {V_col}")

    galaxies = sorted(df["Galaxy"].unique())
    print(f"Number of galaxies: {len(galaxies)}")

    records = []
    for gal in galaxies:
        df_gal = df[df["Galaxy"] == gal]

        R = df_gal[R_col].values
        V = df_gal[V_col].values

        result = compute_kappa1_for_galaxy(R, V)

        if result is None:
            continue

        kappa1, rms_resid, rms_frac = result
        N_points = len(R)
        R_max = float(np.max(R))
        V_max = float(np.max(V))

        records.append(
            {
                "Galaxy": gal,
                "N_points": N_points,
                "R_max": R_max,
                "V_max": V_max,
                "kappa1": kappa1,
                "RMS_resid": rms_resid,
                "RMS_frac": rms_frac,
                "Poly_deg": 3,
                "R_col": R_col,
                "V_col": V_col,
            }
        )

    df_out = pd.DataFrame.from_records(records)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved κ1 table → {OUTPUT_FILE}")
    print(f"Usable galaxies: {len(df_out)}")

    if len(df_out) > 0:
        print("\nκ1 summary:")
        print(df_out["kappa1"].describe())
        print("\nSample:")
        print(df_out.head())

    print("=== DONE ===")


if __name__ == "__main__":
    main()
