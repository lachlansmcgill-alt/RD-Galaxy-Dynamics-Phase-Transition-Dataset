"""
tdgl_model.py

Simple phenomenological TDGL-inspired rotation-curve model and fitting helpers.

The goal here is *traceability*, not final physics. This gives us an explicit,
well-documented mapping from SPARC radii/velocities to a scale length xi_GL
plus shape parameters, which we can later refine.

Model form (3-parameter):

    V_model(R) = V_flat * (1 - exp(-(R / xi_GL)**alpha))

- V_flat  : asymptotic flat rotation speed
- xi_GL   : characteristic "coherence length" of the transition
- alpha   : controls inner steepness (solid-body ~ 1, cuspier > 1)

This is intentionally simple and transparent.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.optimize import curve_fit


def tdgl_velocity(R: np.ndarray, V_flat: float, xi_GL: float, alpha: float) -> np.ndarray:
    """TDGL-inspired rotation curve model."""
    R = np.asarray(R, dtype=float)
    # Prevent division by zero at R ~ 0
    R_safe = np.where(R <= 0, 1e-4, R)
    return V_flat * (1.0 - np.exp(-(R_safe / xi_GL) ** alpha))


@dataclass
class TDGLFitResult:
    galaxy: str
    V_flat: float
    xi_GL: float
    alpha: float
    chi2_red: float
    rms_resid: float
    n_points: int
    success: bool
    message: str


def fit_tdgl_curve(
    R: np.ndarray,
    V: np.ndarray,
    Verr: Optional[np.ndarray] = None,
    galaxy: str = ""
) -> TDGLFitResult:
    """
    Fit the TDGL velocity model to a single galaxy rotation curve.

    Parameters
    ----------
    R : array-like
        Radii (kpc)
    V : array-like
        Observed rotation speeds (km/s)
    Verr : array-like or None
        1σ uncertainties on V. If None, uniform weights are used.
    galaxy : str
        Name for bookkeeping in the result.

    Returns
    -------
    TDGLFitResult
    """
    R = np.asarray(R, dtype=float)
    V = np.asarray(V, dtype=float)

    # Basic sanity
    mask = np.isfinite(R) & np.isfinite(V)
    if Verr is not None:
        Verr = np.asarray(Verr, dtype=float)
        mask &= np.isfinite(Verr)

    R = R[mask]
    V = V[mask]
    if Verr is not None:
        Verr = Verr[mask]

    n = len(R)
    if n < 6:
        return TDGLFitResult(
            galaxy=galaxy,
            V_flat=np.nan,
            xi_GL=np.nan,
            alpha=np.nan,
            chi2_red=np.nan,
            rms_resid=np.nan,
            n_points=n,
            success=False,
            message="Not enough points for TDGL fit (< 6).",
        )

    # Initial guesses
    V_flat0 = np.nanmax(V)
    xi0 = np.nanmedian(R)
    alpha0 = 1.5

    p0 = [V_flat0, xi0, alpha0]

    # Bounds to keep parameters physical-ish
    bounds = (
        [1.0, 0.01, 0.2],   # V_flat, xi_GL, alpha lower
        [1e4, 100.0, 5.0],  # upper
    )

    try:
        if Verr is None:
            popt, pcov = curve_fit(
                tdgl_velocity,
                R,
                V,
                p0=p0,
                bounds=bounds,
                maxfev=20000,
            )
        else:
            sigma = np.where(Verr <= 0, np.nanmedian(Verr[Verr > 0]), Verr)
            popt, pcov = curve_fit(
                tdgl_velocity,
                R,
                V,
                p0=p0,
                sigma=sigma,
                absolute_sigma=True,
                bounds=bounds,
                maxfev=20000,
            )

        V_flat_fit, xi_GL_fit, alpha_fit = popt

        V_model = tdgl_velocity(R, *popt)
        resid = V - V_model
        rms_resid = float(np.sqrt(np.mean(resid**2)))

        if Verr is not None:
            sigma = np.where(Verr <= 0, np.nanmedian(Verr[Verr > 0]), Verr)
            chi2 = float(np.sum((resid / sigma) ** 2))
            dof = max(n - len(popt), 1)
            chi2_red = chi2 / dof
        else:
            chi2_red = np.nan

        return TDGLFitResult(
            galaxy=galaxy,
            V_flat=float(V_flat_fit),
            xi_GL=float(xi_GL_fit),
            alpha=float(alpha_fit),
            chi2_red=chi2_red,
            rms_resid=rms_resid,
            n_points=n,
            success=True,
            message="OK",
        )

    except Exception as e:
        return TDGLFitResult(
            galaxy=galaxy,
            V_flat=np.nan,
            xi_GL=np.nan,
            alpha=np.nan,
            chi2_red=np.nan,
            rms_resid=np.nan,
            n_points=n,
            success=False,
            message=f"Fit failed: {e}",
        )
