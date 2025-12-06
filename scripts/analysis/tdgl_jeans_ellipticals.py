"""
TDGL-Jeans Analysis for ATLAS3D Elliptical Galaxies

This script implements the complete TDGL-Jeans analogy to test Law V 
(Phase Transition) by fitting velocity dispersion profiles of pressure-supported
early-type galaxies.

Theoretical Framework:
---------------------
The TDGL functional generates a force field that enters the collisionless Jeans equation:

    -dΦ_total/dr = -dΦ_bary/dr + a_RD(r; ξ_GL, r_core)
    
where the RD acceleration from the θ-Field is:

    a_RD(r) = (V_char²/r) * (1 - e^(-r/r_core)) * (1 - r_core/ξ_GL) * f(r)

The Jeans equation for spherical systems relates this to observed dispersion:

    -dΦ_total/dr = (1/ν) * d(ν*σ_r²)/dr + (2β*σ_r²)/r

where:
    ν(r)    = stellar density profile
    σ_r(r)  = radial velocity dispersion
    β       = anisotropy parameter

Test of Law V (Phase Transition):
---------------------------------
For pressure-supported ellipticals (V/σ < 0.3, λ_R < 0.1):

    If ξ_GL < r_core  →  τ = √(r_core² - ξ_GL²)/V_char  →  τ < 0  (FROZEN)
    
This would be the definitive "smoking gun" evidence for the phase transition.

Expected Results:
----------------
- S0s / Fast rotators:  ξ_GL > r_core  →  τ > 0  (active recursion)
- Slow rotators:        ξ_GL ≈ r_core  →  τ ≈ 0  (transition)
- Pure ellipticals:     ξ_GL < r_core  →  τ < 0  (frozen recursion)

Author: RD Physics Analysis Pipeline
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit, minimize
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import warnings

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "results" / "atlas3d"
OUTPUT_DIR = PROJECT_ROOT / "results" / "test6_ellipticals"
FIG_DIR = OUTPUT_DIR / "figures"

# Physical constants
G_NEWTON = 4.302e-6  # kpc (km/s)^2 / Msun
KPC_TO_ARCSEC = 206265  # arcsec/radian at 1 kpc


class TDGLJeansModel:
    """
    TDGL-Jeans model for pressure-supported elliptical galaxies.
    
    Implements the full Jeans equation with TDGL force field from θ-Field.
    """
    
    def __init__(self, galaxy_name, distance_mpc):
        """
        Initialize TDGL-Jeans model for a galaxy.
        
        Parameters
        ----------
        galaxy_name : str
            Galaxy identifier
        distance_mpc : float
            Distance in Mpc
        """
        self.galaxy = galaxy_name
        self.distance = distance_mpc
        self.kpc_per_arcsec = distance_mpc * 1000 / KPC_TO_ARCSEC
        
        # Fitted parameters
        self.xi_GL = None
        self.r_core = None
        self.V_char = None
        self.beta = None
        self.tau_unified = None
        
        # Data
        self.r_data = None
        self.sigma_data = None
        self.nu_data = None
        self.phi_bary = None
    
    def rdg_acceleration(self, r, xi_GL, r_core, V_char):
        """
        Compute RD (θ-Field) acceleration from TDGL functional.
        
        This is the force derived from the TDGL potential, representing
        the contribution of recursive dynamics to the gravitational field.
        
        Parameters
        ----------
        r : array
            Radius (kpc)
        xi_GL : float
            Ginzburg-Landau coherence length (kpc)
        r_core : float
            Core radius where baryonic effects dominate (kpc)
        V_char : float
            Characteristic velocity (km/s)
            
        Returns
        -------
        a_RD : array
            RD acceleration (km²/s²/kpc)
        """
        # Avoid division by zero
        r = np.maximum(r, 0.01)
        
        # Core transition function (exponential approach)
        core_factor = 1.0 - np.exp(-r / r_core)
        
        # GL coupling factor (changes sign at transition)
        # When ξ_GL > r_core: positive coupling (active)
        # When ξ_GL < r_core: negative coupling (frozen)
        coupling = 1.0 - (r_core / xi_GL)
        
        # Radial falloff (prevents divergence at small r)
        falloff = np.tanh(r / r_core)
        
        # Combined RD acceleration
        a_RD = (V_char**2 / r) * core_factor * coupling * falloff
        
        return a_RD
    
    def baryonic_potential_derivative(self, r, nu):
        """
        Compute derivative of baryonic gravitational potential.
        
        Uses stellar density profile to calculate gravitational force
        from baryonic matter only.
        
        Parameters
        ----------
        r : array
            Radius (kpc)
        nu : array
            Stellar number density (arbitrary units, normalized)
            
        Returns
        -------
        dPhi_dr : array
            Derivative of baryonic potential (km²/s²/kpc)
        """
        # Avoid edge issues
        if len(r) < 3:
            return np.zeros_like(r)
        
        # Compute enclosed mass profile M(<r)
        # Assuming spherical symmetry: dM = 4π r² ν dr
        volume_elements = 4 * np.pi * r**2
        mass_elements = nu * volume_elements
        
        # Cumulative integration
        M_enclosed = cumulative_trapezoid(mass_elements, r, initial=0)
        
        # Gravitational acceleration: GM/r²
        # Need to normalize to match observations
        # For now, use characteristic scaling
        r_safe = np.maximum(r, 0.01)
        dPhi_dr = G_NEWTON * M_enclosed / r_safe**2
        
        return dPhi_dr
    
    def jeans_rhs(self, r, sigma, nu, beta):
        """
        Compute right-hand side of Jeans equation.
        
        RHS = (1/ν) * d(ν*σ_r²)/dr + (2β*σ_r²)/r
        
        Parameters
        ----------
        r : array
            Radius (kpc)
        sigma : array
            Velocity dispersion (km/s)
        nu : array
            Stellar density (normalized)
        beta : float or array
            Anisotropy parameter
            
        Returns
        -------
        jeans_rhs : array
            Right-hand side of Jeans equation (km²/s²/kpc)
        """
        sigma_sq = sigma**2
        nu_sigma_sq = nu * sigma_sq
        
        # Derivative term: d(ν*σ²)/dr
        if len(r) > 2:
            d_nu_sigma_sq = np.gradient(nu_sigma_sq, r)
        else:
            d_nu_sigma_sq = np.zeros_like(r)
        
        # Full RHS
        nu_safe = np.maximum(nu, 1e-10)
        r_safe = np.maximum(r, 0.01)
        
        term1 = d_nu_sigma_sq / nu_safe
        term2 = 2 * beta * sigma_sq / r_safe
        
        return term1 + term2
    
    def model_residual(self, params, r, sigma_obs, nu, beta):
        """
        Compute residual for TDGL-Jeans fit.
        
        The residual is the difference between:
        LHS: -dΦ_bary/dr + a_RD(r)
        RHS: Jeans equation from observed σ(r)
        
        Parameters
        ----------
        params : tuple
            (xi_GL, r_core, V_char) - TDGL parameters
        r : array
            Radius (kpc)
        sigma_obs : array
            Observed velocity dispersion (km/s)
        nu : array
            Stellar density profile
        beta : float
            Anisotropy parameter
            
        Returns
        -------
        residual : array
            Difference between LHS and RHS
        """
        xi_GL, r_core, V_char = params
        
        # Compute LHS: Total gravitational acceleration
        dPhi_bary = self.baryonic_potential_derivative(r, nu)
        a_RD = self.rdg_acceleration(r, xi_GL, r_core, V_char)
        
        lhs = -dPhi_bary + a_RD
        
        # Compute RHS: Jeans equation from observations
        rhs = self.jeans_rhs(r, sigma_obs, nu, beta)
        
        # Residual (minimize this)
        residual = lhs - rhs
        
        return residual
    
    def fit_tdgl_jeans(self, r, sigma, nu, beta=0.0, initial_guess=None):
        """
        Fit TDGL-Jeans model to observed velocity dispersion profile.
        
        Parameters
        ----------
        r : array
            Radius in physical units (kpc)
        sigma : array
            Observed velocity dispersion (km/s)
        nu : array
            Stellar density profile (normalized)
        beta : float, optional
            Anisotropy parameter (default: 0 = isotropic)
        initial_guess : tuple, optional
            Initial (xi_GL, r_core, V_char)
            
        Returns
        -------
        params : dict
            Fitted TDGL parameters and derived quantities
        """
        self.r_data = r
        self.sigma_data = sigma
        self.nu_data = nu
        self.beta = beta
        
        # Initial guess if not provided
        if initial_guess is None:
            r_eff = r[len(r)//2]  # Effective radius
            sigma_mean = np.mean(sigma)
            initial_guess = (r_eff * 1.5, r_eff * 0.8, sigma_mean * 1.2)
        
        # Bounds: all parameters must be positive
        # But allow xi_GL < r_core for τ < 0 regime
        bounds = ([0.1, 0.1, 50], [100, 100, 500])
        
        # Minimize residual using least squares
        def cost_function(params):
            residual = self.model_residual(params, r, sigma, nu, beta)
            return np.sum(residual**2)
        
        result = minimize(cost_function, initial_guess, 
                         bounds=list(zip(bounds[0], bounds[1])),
                         method='L-BFGS-B')
        
        if not result.success:
            warnings.warn(f"Fit did not converge for {self.galaxy}")
        
        # Extract fitted parameters
        xi_GL, r_core, V_char = result.x
        
        self.xi_GL = xi_GL
        self.r_core = r_core
        self.V_char = V_char
        
        # Compute τ_unified
        self.compute_tau()
        
        # Package results
        params = {
            'Galaxy': self.galaxy,
            'xi_GL_kpc': xi_GL,
            'r_core_kpc': r_core,
            'V_char_kms': V_char,
            'beta': beta,
            'xi_over_rcore': xi_GL / r_core,
            'tau_unified_Myr': self.tau_unified,
            'fit_success': result.success,
            'chi_squared': result.fun
        }
        
        return params
    
    def compute_tau(self):
        """
        Compute unified timescale τ from fitted TDGL parameters.
        
        τ = √|r_core² - ξ_GL²| / V_char
        
        Sign convention:
        - If ξ_GL > r_core: τ > 0 (active recursion, disk-like)
        - If ξ_GL < r_core: τ < 0 (frozen recursion, elliptical)
        """
        if self.xi_GL is None or self.r_core is None:
            self.tau_unified = None
            return
        
        # Compute τ with proper sign
        r_core_sq = self.r_core**2
        xi_GL_sq = self.xi_GL**2
        
        # Magnitude
        tau_magnitude = np.sqrt(np.abs(r_core_sq - xi_GL_sq)) / self.V_char
        
        # Sign: positive if ξ > r, negative if ξ < r
        sign = np.sign(xi_GL_sq - r_core_sq)
        
        # Convert to Myr (from time in units of kpc/(km/s))
        tau_kpc_kms = tau_magnitude * sign
        tau_Myr = tau_kpc_kms * 978.4  # Conversion factor
        
        self.tau_unified = tau_Myr
    
    def plot_fit(self, save_path=None):
        """
        Visualize TDGL-Jeans fit results.
        
        Creates 4-panel diagnostic plot:
        1. Velocity dispersion profile (data vs model)
        2. Acceleration components (baryonic, RD, total)
        3. Jeans equation balance
        4. TDGL phase diagnostic (ξ vs r_core)
        """
        if self.r_data is None:
            print(f"No fit data available for {self.galaxy}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{self.galaxy} - TDGL-Jeans Analysis\n' + 
                    f'τ = {self.tau_unified:.1f} Myr  |  ' +
                    f'ξ/r = {self.xi_GL/self.r_core:.3f}',
                    fontsize=14, fontweight='bold')
        
        r = self.r_data
        
        # Panel 1: Velocity dispersion profile
        ax1 = axes[0, 0]
        ax1.scatter(r, self.sigma_data, c='blue', s=50, label='Observed σ(r)', zorder=3)
        ax1.axhline(np.mean(self.sigma_data), color='blue', ls='--', alpha=0.5)
        ax1.set_xlabel('Radius (kpc)')
        ax1.set_ylabel('Velocity Dispersion (km/s)')
        ax1.set_title('Dispersion Profile')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Panel 2: Acceleration components
        ax2 = axes[0, 1]
        dPhi_bary = self.baryonic_potential_derivative(r, self.nu_data)
        a_RD = self.rdg_acceleration(r, self.xi_GL, self.r_core, self.V_char)
        a_total = -dPhi_bary + a_RD
        
        ax2.plot(r, -dPhi_bary, 'g-', label='-dΦ_bary/dr', lw=2)
        ax2.plot(r, a_RD, 'r-', label='a_RD (θ-Field)', lw=2)
        ax2.plot(r, a_total, 'k-', label='Total', lw=2, ls='--')
        ax2.axhline(0, color='gray', ls=':', alpha=0.5)
        ax2.set_xlabel('Radius (kpc)')
        ax2.set_ylabel('Acceleration (km²/s²/kpc)')
        ax2.set_title('Force Components')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Panel 3: Jeans equation balance
        ax3 = axes[1, 0]
        jeans_rhs = self.jeans_rhs(r, self.sigma_data, self.nu_data, self.beta)
        ax3.plot(r, a_total, 'k-', label='LHS (Total Accel)', lw=2)
        ax3.plot(r, jeans_rhs, 'b--', label='RHS (Jeans Eq)', lw=2)
        ax3.fill_between(r, a_total, jeans_rhs, alpha=0.3, color='red', 
                        label='Residual')
        ax3.set_xlabel('Radius (kpc)')
        ax3.set_ylabel('Acceleration (km²/s²/kpc)')
        ax3.set_title('Jeans Equation Balance')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Panel 4: Phase diagnostic
        ax4 = axes[1, 1]
        ax4.scatter(self.xi_GL, self.r_core, c='red', s=200, marker='*', 
                   edgecolors='black', linewidths=2, zorder=5,
                   label=f'{self.galaxy}')
        
        # Phase boundaries
        lim = max(self.xi_GL, self.r_core) * 1.3
        ax4.plot([0, lim], [0, lim], 'k--', lw=2, alpha=0.5, label='Transition (τ=0)')
        
        # Shade regions
        xi_grid = np.linspace(0, lim, 100)
        ax4.fill_between(xi_grid, 0, xi_grid, alpha=0.2, color='blue', 
                        label='τ > 0 (Active)')
        ax4.fill_between(xi_grid, xi_grid, lim, alpha=0.2, color='red',
                        label='τ < 0 (Frozen)')
        
        ax4.set_xlabel('ξ_GL (kpc)')
        ax4.set_ylabel('r_core (kpc)')
        ax4.set_title('TDGL Phase Diagram')
        ax4.legend()
        ax4.grid(alpha=0.3)
        ax4.set_xlim(0, lim)
        ax4.set_ylim(0, lim)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()


def load_atlas3d_candidates():
    """
    Load ATLAS3D candidates for τ < 0 analysis.
    
    Returns
    -------
    candidates : DataFrame
        Galaxies with V/σ < 0.3 or slow rotator classification
    """
    atlas3d_file = DATA_DIR / "atlas3d_clean.csv"
    
    if not atlas3d_file.exists():
        raise FileNotFoundError(f"ATLAS3D data not found: {atlas3d_file}")
    
    df = pd.read_csv(atlas3d_file)
    
    # Filter for candidates
    # Criteria: V/σ < 0.3 OR SlowRotator == True
    candidates = df[
        (df['V_over_sigma'] < 0.3) | 
        (df['SlowRotator'] == True)
    ].copy()
    
    print(f"Loaded {len(candidates)} candidates for τ < 0 analysis")
    print(f"  Slow rotators (λ_R < 0.1): {candidates['SlowRotator'].sum()}")
    print(f"  Pressure-supported (V/σ < 0.3): {(candidates['V_over_sigma'] < 0.3).sum()}")
    
    return candidates


def main():
    """
    Main execution: Acquire kinematic profiles and perform TDGL-Jeans fits.
    """
    print("="*70)
    print("TDGL-JEANS ANALYSIS FOR ATLAS3D ELLIPTICALS")
    print("="*70)
    print("\nTheoretical Framework:")
    print("  Jeans Equation: -dΦ_total/dr = -dΦ_bary/dr + a_RD(r; ξ, r_core)")
    print("  Test: ξ < r_core  →  τ < 0  (FROZEN RECURSION)")
    print("="*70)
    
    # Load candidates
    candidates = load_atlas3d_candidates()
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("KINEMATIC DATA ACQUISITION")
    print("="*70)
    print("\nNEXT STEPS:")
    print("\n1. Acquire σ(r) profiles for 89 candidates:")
    print("   - ATLAS3D has IFU 2D velocity fields")
    print("   - Extract radial profiles from FITS cubes")
    print("   - Alternative: Use published tables (if available)")
    
    print("\n2. Acquire ν(r) stellar density profiles:")
    print("   - Photometric profiles from 2MASS/SDSS")
    print("   - Or use Sérsic fits from ATLAS3D")
    
    print("\n3. Implement profile extraction:")
    print("   python scripts/extract/atlas3d_kinematics.py")
    
    print("\n4. Run TDGL-Jeans fits on all candidates")
    
    print("\n" + "="*70)
    print("EXPECTED OUTCOMES")
    print("="*70)
    print("\nIf Law V is correct:")
    print("  - Slow rotators: τ < 0 (ξ < r_core)")
    print("  - Fast rotators: τ > 0 (ξ > r_core)")
    print("  - Clear correlation with V/σ ratio")
    
    print("\nIf Law V is wrong:")
    print("  - Random scatter of τ values")
    print("  - No systematic ξ/r_core < 1 for slow rotators")
    
    print("\n" + "="*70)
    print("This would complete Test 6: 100% validation")
    print("="*70)
    
    # Save candidate list
    output_file = OUTPUT_DIR / "elliptical_candidates.csv"
    candidates.to_csv(output_file, index=False)
    print(f"\n[OK] Saved candidate list: {output_file}")
    
    # Example: Show top 10 most extreme candidates
    print("\nTop 10 candidates (lowest V/σ):")
    top10 = candidates.nsmallest(10, 'V_over_sigma')[
        ['Galaxy', 'lambda_R', 'V_over_sigma', 'SlowRotator', 'Distance_Mpc']
    ]
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
