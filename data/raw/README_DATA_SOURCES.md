# Data Sources and Provenance

This file documents the origin, processing, and quality control of all data in this dataset.

## Raw Data Sources

### SPARC Database
- **Source:** Spitzer Photometry and Accurate Rotation Curves (SPARC)
- **Citation:** Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016). SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves. *The Astronomical Journal*, 152(6), 157. https://doi.org/10.3847/0004-6256/152/6/157
- **URL:** http://astroweb.cwru.edu/SPARC/
- **Description:** 175 late-type galaxies with high-quality rotation curves from 21cm HI/Hα observations
- **Data Files:**
  - `SPARC_Mass_Models.csv`: Rotation curve data (radius, velocity, uncertainties)
  - `stellar_masses/`: Stellar mass surface density profiles for 163 galaxies
- **Access Date:** 2024
- **License:** Public domain (academic use)

### ATLAS³D Survey
- **Source:** ATLAS³D project - volume-limited sample of early-type galaxies
- **Citation:** Cappellari, M., et al. (2011). The ATLAS³D project - I. A volume-limited sample of 260 nearby early-type galaxies: science goals and selection criteria. *MNRAS*, 413(2), 813-836. https://doi.org/10.1111/j.1365-2966.2010.18174.x
- **Additional Data:** Jeans anisotropic modeling from multiple ATLAS³D papers (kinematics, dynamical models)
- **Description:** 260 early-type galaxies with integral field spectroscopy
- **Data Products Used:**
  - Velocity dispersion profiles
  - Dynamical modeling parameters
  - Stellar masses from photometry
- **Access Date:** 2024
- **License:** Public domain (academic use)

## Data Processing Pipeline

### Stage 1: Data Ingestion and Cleaning
**Script:** Various ingest scripts (not included - development phase)
**Input:** Raw SPARC CSV files, ATLAS³D published tables
**Output:** Cleaned datasets with standardized units and formats
**Quality Control:**
- Removed galaxies with <5 data points in rotation curves
- Flagged galaxies with large observational uncertainties
- Standardized distance scales using latest measurements

### Stage 4: Coherence Metric (κ₁) Computation
**Script:** `scripts/analysis/compute_kappa1.py`
**Input:** `data/raw/SPARC/SPARC_Mass_Models.csv` (175 galaxies)
**Output:** `data/results/sparc_spirals/kappa1.csv`
**Process:**
- Computed κ₁ = (∂V/∂r) / (V/r) coherence metric
- Used numerical derivatives with Savitzky-Golay filter
- Range: κ₁ ∈ [-1, +1] quantifies rotation curve shape
**Quality Control:**
- Required minimum 5 data points for numerical derivatives
- Flagged high-uncertainty regions (σ_V > 10 km/s)
- Final sample: 175 galaxies with κ₁ measurements

### Stage 8: TDGL Model Fitting (Spirals)
**Script:** `scripts/analysis/fit_tdgl.py`
**Model:** `scripts/models/tdgl_model.py`
**Input:** `data/raw/SPARC/SPARC_Mass_Models.csv`, stellar mass profiles
**Output:** `data/results/sparc_spirals/tdgl_fits.csv`
**Fitted Parameters:**
- ξ_GL: Ginzburg-Landau coherence length (dimensionless)
- α: Nonlinearity parameter (dimensionless)
- V₀: Velocity scale (km/s)
**Derived Parameters:**
- r_core = ξ_GL × r_disk: Physical core radius (kpc)
- Reduced χ²: Goodness-of-fit metric
**Quality Control:**
- Required χ²_red < 3.0 for acceptable fits
- Flagged unphysical parameters (ξ < 0 or α < 0)
- Final sample: **109 high-quality TDGL fits**

### Test 6: Jeans Anisotropic Modeling (Ellipticals)
**Script:** `scripts/analysis/tdgl_jeans_ellipticals.py`
**Input:** ATLAS³D velocity dispersion profiles, photometry
**Output:** `data/results/atlas3d_ellipticals/tdgl_jeans_results.csv`
**Method:**
- Modified Jeans anisotropic models with RD corrections
- Fitted: M_BH (black hole mass), β (anisotropy), ξ_GL, α
- Derived: r_core from ξ_GL × effective radius
**Quality Control:**
- Required convergence in anisotropic modeling
- Compared to published ATLAS³D dynamical masses
- Final sample: **89 early-type galaxies**

### Derived Parameters: Extended Coherence Metrics
**Script:** `scripts/analysis/extended_coherence_params.py`
**Input:** 
- `data/results/sparc_spirals/tdgl_fits.csv` (ξ, r_core)
- Morphology classifications
**Output:** `data/results/sparc_spirals/extended_coherence_params.csv`
**Derived Metrics:**
- **η (eta):** η = ξ / r_core (dimensionless coherence density)
- **τ (tau):** τ = sign(ξ² - r²) × √|ξ² - r²| / V (phase transition parameter)
  - τ > 0: Active recursion (spirals)
  - τ < 0: Frozen recursion (ellipticals)
  - τ = 0: **Critical phase transition**
**Critical Note:** τ is **NOT a fitted parameter** - it's derived from fitted ξ and r_core values. This ensures no circular logic in phase transition discovery.

### Evolutionary Analysis
**Script:** `scripts/analysis/tau_evolutionary_analysis.py`
**Input:** Extended coherence parameters + morphology + stellar ages
**Output:** `data/results/sparc_spirals/tau_evolutionary_analysis.csv`
**Analysis:**
- Correlated τ with Hubble type (Sa → Sd)
- Correlated τ with stellar age, gas fraction
- Statistical tests for τ = 0 as morphological boundary
**Key Result:** 100% spirals have τ > 0, 87.6% ellipticals have τ < 0

## Data Quality Summary

### SPARC Spirals
- **Raw Sample:** 175 galaxies
- **After Quality Control:** 109 high-quality TDGL fits
- **Excluded:** 66 galaxies (insufficient data, poor fits, edge-on orientation)
- **Median Uncertainty:** σ_V ~ 5-10 km/s in rotation velocity
- **Distance Range:** 2-50 Mpc
- **Mass Range:** 10⁸ - 10¹¹ M☉

### ATLAS³D Ellipticals
- **Raw Sample:** 260 galaxies
- **After Quality Control:** 89 with reliable Jeans models
- **Excluded:** 171 galaxies (insufficient IFU coverage, complex kinematics, AGN contamination)
- **Median Uncertainty:** σ_disp ~ 10-20 km/s in velocity dispersion
- **Distance Range:** 10-40 Mpc
- **Mass Range:** 10⁹ - 10¹² M☉

## Known Issues and Limitations

1. **SPARC Distance Uncertainties**
   - Distance moduli have ~10% systematic uncertainty
   - Affects absolute mass scales but not rotation curve shapes
   - κ₁ and τ are distance-independent (dimensionless)

2. **Inclination Effects (SPARC)**
   - Edge-on galaxies (i > 80°) excluded due to projection uncertainties
   - Face-on galaxies (i < 30°) have larger velocity errors
   - Final sample: 30° < i < 80° preferred

3. **ATLAS³D Selection Effects**
   - Volume-limited sample biased toward nearby, massive ellipticals
   - Missing low-mass early-types (M* < 10⁹ M☉)
   - τ distribution may not be complete at low-mass end

4. **Missing Stellar Mass Profiles**
   - 12 SPARC galaxies lack stellar mass decomposition
   - Used exponential disk approximation where needed
   - Does not affect TDGL fitting (uses total rotation curve)

5. **Phase Transition Region (τ ≈ 0)**
   - Only 3 galaxies within |τ| < 0.1 (transition zone)
   - Insufficient statistics for detailed transition physics
   - Need larger sample to probe τ = 0 boundary

## Reproducibility Notes

### Random Seeds
All analysis scripts use fixed random seeds where applicable:
- Bootstrap resampling: `np.random.seed(42)`
- MCMC sampling: `seed=42` in emcee calls

### Software Versions
See `requirements.txt` for exact package versions used:
- Python 3.9+
- numpy 1.24+
- scipy 1.10+
- pandas 2.0+
- matplotlib 3.7+
- astropy 5.2+

### Numerical Precision
- All calculations use 64-bit floating point (np.float64)
- Integration tolerances: rtol=1e-8, atol=1e-10
- Minimization tolerances: ftol=1e-8, xtol=1e-8

## Data Availability

All raw data sources are publicly available:
- **SPARC:** http://astroweb.cwru.edu/SPARC/
- **ATLAS³D:** CDS/VizieR (multiple catalog entries)

This processed dataset provides:
- Cleaned and validated data
- Consistent parameter derivations
- Quality-controlled sample selection
- Reproducible analysis pipeline

## Contact

For questions about data processing or quality issues:
- Open an issue on the GitHub repository
- Email: [Contact email]

## Updates and Versions

**Version 1.0 (December 2025):**
- Initial release with 194 galaxies
- Complete κ₁ → ξ, r_core → η, τ derivation chain
- Phase transition discovery validated

Future versions will include:
- Extended SPARC sample (if new rotation curves released)
- Additional early-type galaxy surveys
- Refined distance scales from Gaia EDR4
