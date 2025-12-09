# RD Galaxy Phase Transition Dataset v1.0

**DOI:** 10.5281/zenodo.17838464  
**Version:** 1.0  
**Date:** December 2025  
**License:** MIT (Code) + CC BY 4.0 (Data)

---

## Overview

This dataset contains the complete analysis pipeline and results validating Recursive Dynamics (RD) cosmology through galaxy observations. The central discovery is a **phase transition in galactic dynamics at τ = 0**, where τ is the unified dynamical timescale.

### Key Result

**100% of spiral galaxies show τ > 0** (active, expanding coherence) while **87.6% of elliptical galaxies show τ < 0** (frozen, contracting coherence), with S0 galaxies clustered at the phase boundary (τ ≈ 0).

This phase transition is predicted by RD cosmology but has no analog in ΛCDM or MOND frameworks.

---

## Dataset Contents

### Galaxies Analyzed
- **109 SPARC spiral galaxies** with TDGL rotation curve fits
- **89 ATLAS³D elliptical galaxies** with TDGL-Jeans velocity dispersion fits
- **Total: 194 galaxies** spanning the complete Hubble sequence

### Files Included
- Complete analysis scripts for reproducibility
- TDGL fitting code (rotation curves + Jeans equation)
- Derived parameter calculations (κ₁ → ξ, r_core → η, τ)
- Publication-quality figures (5 main figures)
- Comprehensive documentation

---

## Quick Start

### Requirements

- Python 3.8+
- pandas, numpy, scipy, matplotlib, seaborn
- See `requirements.txt` for complete dependencies

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify data files
python -c "import pandas as pd; print(pd.read_csv('data/results/sparc_spirals/tau_evolutionary_analysis.csv').shape)"
```

### Generate Figures

```bash
# Generate main publication figures (from repo root)
python scripts/figures/generate_paper_figures_full.py
python scripts/figures/generate_figure5_unified_sequence.py

# Figures saved to figures/ directory
```

---

## Directory Structure

```
RD_galaxy_phase_transition_dataset_v1.0/
├── data/
│   ├── raw/                         # Original datasets (SPARC, ATLAS³D)
│   └── results/                     # Analysis results
│       ├── sparc_spirals/           # SPARC rotation curve analysis
│       │   ├── kappa1.csv           # κ₁ coherence metric (175 galaxies)
│       │   ├── tdgl_fits.csv        # TDGL fits: ξ, α, r_core (109 galaxies)
│       │   ├── extended_coherence_params.csv  # η, τ derived parameters
│       │   └── tau_evolutionary_analysis.csv  # τ + morphology combined
│       └── atlas3d_ellipticals/     # ATLAS³D elliptical analysis
│           ├── tdgl_jeans_results.csv  # Test 6 results (89 galaxies)
│           └── individual_fits/     # Individual galaxy fits (PNG)
│
├── scripts/
│   ├── models/                      # TDGL model implementation
│   ├── analysis/                    # Analysis pipeline scripts
│   │   ├── compute_kappa1.py        # Stage 4: κ₁ computation
│   │   ├── fit_tdgl.py              # Stage 8: TDGL fitting
│   │   ├── extended_coherence_params.py  # Derive η, τ
│   │   ├── tau_evolutionary_analysis.py  # Merge τ + morphology
│   │   └── tdgl_jeans_ellipticals.py     # Test 6: Ellipticals
│   └── figures/                     # Figure generation
│       ├── generate_paper_figures_full.py  # Figures 1-3
│       └── generate_figure5_unified_sequence.py  # Figure 5
│
├── figures/                         # Publication figures (PNG, 300 DPI)
│   ├── paper_fig1_full_evolutionary_sequence.png
│   ├── paper_fig2_full_bimodal_histogram.png
│   ├── paper_fig3_full_phase_diagram.png
│   └── paper_fig5_complete_evolutionary_sequence.png
│
├── docs/                            # Documentation
│   └── PARAMETERS.md                # Parameter definitions (ξ, τ, α, etc.)
│
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

---

## Parameter Derivation Chain

The τ phase parameter is derived through a systematic 5-stage pipeline:

```
Raw Data (R, V_obs)
    ↓ Stage 4: compute_kappa1.py
κ₁ (coherence metric)
    ↓ Stage 8: fit_tdgl.py
ξ_GL, α, r_core (TDGL parameters)
    ↓ extended_coherence_params.py
η = ξ_GL / r_core
τ = sign(ξ²-r²) × √|ξ²-r²| / V
    ↓ tau_evolutionary_analysis.py
τ + Morphology (Hubble Type)
    ↓ Phase Transition Discovery
```

**Critical insight:** τ is NOT a free parameter but is derived from fitted values of ξ and r_core, making the morphological correlation a genuine prediction rather than a fit.

---

## Key Parameters

| Symbol | Name | Units | Formula | Physical Meaning |
|--------|------|-------|---------|------------------|
| **ξ_GL** | GL coherence length | kpc | Fitted from TDGL | Transition scale in rotation curve |
| **α** | Shape parameter | - | Fitted from TDGL | Steepness of inner rise |
| **r_core** | Core radius | kpc | ξ × [arctanh(0.5)]^(1/α) | Radius where V = 0.5×V₀ |
| **κ₁** | Coherence metric | - | 1 - (RMS/V_max) | Rotation curve smoothness |
| **η** | Scale ratio | - | ξ_GL / r_core | Coherence-to-core ratio |
| **τ** | Phase parameter | Myr | sign(ξ²-r²)×√\|ξ²-r²\|/V | Dynamical timescale |

### τ Physical Interpretation

- **τ > 0:** ξ > r (extended coherence) → **ACTIVE** spirals
- **τ = 0:** ξ = r (critical point) → **TRANSITION** S0 galaxies  
- **τ < 0:** ξ < r (confined coherence) → **FROZEN** ellipticals

---

## Results Summary

### SPARC Spirals (109 galaxies)
- **τ > 0:** 105/105 (100%)
- **Median τ:** +15.2 Myr
- **Range:** 0.0 to +1299.4 Myr
- **Mean ξ/r:** 3.39 ± 4.13 (all > 1)

### ATLAS³D Ellipticals (89 galaxies)
- **τ < 0:** 78/89 (87.6%)
- **Median τ:** -7.9 Myr
- **Range:** -1925.3 to +588.0 Myr
- **Mean ξ/r (frozen):** 0.62 ± 0.21 (all < 1)

### Phase Transition Statistics
- **Complete separation:** 100% of τ>0 have ξ>r, 100% of τ<0 have ξ<r
- **S0 galaxies:** Mean τ = 0.7 Myr (at transition)
- **M87 benchmark:** τ = -22.5 Myr, ξ/r = 0.552

---

## Data Sources

### SPARC Catalog
- **Reference:** Lelli et al. (2016), AJ, 152, 157
- **DOI:** 10.3847/0004-6256/152/6/157
- **URL:** http://astroweb.cwru.edu/SPARC/
- **Sample:** 175 late-type galaxies with high-quality rotation curves
- **Used in this study:** 109 galaxies with successful TDGL fits

### ATLAS³D Survey
- **Reference:** Cappellari et al. (2011), MNRAS, 413, 813
- **DOI:** 10.1111/j.1365-2966.2010.18174.x
- **URL:** http://www-astro.physics.ox.ac.uk/atlas3d/
- **Sample:** 260 early-type galaxies
- **Used in this study:** 89 pressure-supported galaxies (V/σ < 0.3)

---

## Citation

If you use this dataset, please cite:

### Dataset
```bibtex
@dataset{rd_galaxy_2025,
  author = {Lachlan McGill},
  title = {RD Galaxy Phase Transition Dataset v1.0},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.17838464},
  url = {https://doi.org/10.5281/zenodo.17838464}
}
```



### Original Data Sources
- **SPARC:** Lelli et al. (2016), AJ, 152, 157
- **ATLAS³D:** Cappellari et al. (2011), MNRAS, 413, 813

---

## License

### Code (MIT License)
All Python scripts in `scripts/` are licensed under the MIT License.

### Data (CC BY 4.0)
All data files in `data/` and `figures/` are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

See LICENSE.txt for full license text.

---

## Reproducibility

All results can be reproduced by running the scripts in order:

1. **Stage 4:** `python scripts/analysis/compute_kappa1.py`  
   → Generates κ₁ coherence metric

2. **Stage 8:** `python scripts/analysis/fit_tdgl.py`  
   → Fits TDGL model to rotation curves → ξ, α, r_core

3. **Extended Coherence:** `python scripts/analysis/extended_coherence_params.py`  
   → Derives η, τ from ξ and r_core

4. **τ + Morphology:** `python scripts/analysis/tau_evolutionary_analysis.py`  
   → Merges τ with Hubble Types

5. **Test 6 (Ellipticals):** `python scripts/analysis/tdgl_jeans_ellipticals.py`  
   → Applies TDGL-Jeans to ATLAS³D

6. **Figures:** 
   ```bash
   python scripts/figures/generate_paper_figures_full.py
   python scripts/figures/generate_figure5_unified_sequence.py
   ```

**Note:** Raw data files are provided for completeness, but the analysis starts from pre-cleaned rotation curves.

---

## Contact

For questions about this dataset:
- **GitHub Issues:** [Repository URL when published]
- **Email:** [Your email]
- **ORCID:** [Your ORCID if applicable]

---

## Acknowledgments

This work uses data from:
- SPARC collaboration (Lelli et al. 2016)RD Galaxy Phase-Transition Dataset v1.0

DOI: (assigned upon Zenodo publication)
Version: 1.0
Date: December 2025
License: MIT (Code) • CC BY 4.0 (Data)

Overview

This dataset provides the complete analysis pipeline and results used to evaluate whether a structural phase transition may exist in galactic dynamics, defined through the unified dynamical parameter τ.

The dataset contains:

109 SPARC spiral galaxies with TDGL rotation-curve fits

89 ATLAS³D elliptical galaxies with TDGL-Jeans dispersion fits

All derived parameters (κ₁, ξ_GL, r_core, η, τ)

Full reproducibility code

Publication-quality figures

Core Empirical Finding

Across the sample analyzed:

All spirals show τ > 0

Most ellipticals (87.6%) show τ < 0

S0 galaxies cluster near τ ≈ 0

This suggests a possible dynamical phase boundary at τ = 0, with spirals on one side and ellipticals on the other. No equivalent scalar exists in ΛCDM or MOND formalism, making τ a potentially useful new diagnostic.

The dataset does not claim a new theory but provides the data and tools for independent verification.

Contents
Included Galaxy Samples
Survey	Type	N	Method
SPARC	Late-type spirals	109	TDGL rotation-curve fitting
ATLAS³D	Early-type ellipticals	89	TDGL-Jeans dispersion fitting
Included Files

Python analysis scripts (scripts/)

Derived parameter tables (data/results/)

Raw data (SPARC + ATLAS³D) for transparency

Five publication-ready figures (figures/)

Documentation (docs/)

Everything required for full reproduction of the τ sequence is included.

Quick Start
Requirements
pip install -r requirements.txt

Verify Installation
python - <<EOF
import pandas as pd
df = pd.read_csv('data/results/sparc_spirals/tau_evolutionary_analysis.csv')
print("Loaded:", df.shape)
EOF

Generate Figures
python scripts/figures/generate_paper_figures_full.py
python scripts/figures/generate_figure5_unified_sequence.py


Outputs are written to figures/.

Directory Structure

(unchanged for clarity — your version is perfect)

Derivation Pipeline

A five-stage chain takes raw kinematic data → τ:

Rotation Curves (SPARC)
Velocity Dispersions (ATLAS3D)
       ↓
Stage 4: κ₁ (coherence metric)
       ↓
Stage 8: TDGL fits → ξ_GL, α, r_core
       ↓
Extended Parameters → η, τ
       ↓
Morphology Merge → τ vs Hubble Type
       ↓
Phase Transition Assessment


Important:
τ arises only from ξ and r_core.
It is not fit to the data and has no tunable parameters, which makes its separation across morphological types a clean empirical outcome rather than a calibration.

Parameter Definitions

(your table is excellent — unchanged besides tiny formatting edits)

Results Summary
Spirals (SPARC, 109 galaxies)

τ > 0 for 100% of rotation-supported systems

Median τ = +15.2 Myr

ξ/r_core strongly > 1 (mean 3.39 ± 4.13)

Ellipticals (ATLAS³D, 89 galaxies)

τ < 0 for 78/89 = 87.6%

Median τ = −7.9 Myr

ξ/r_core < 1 for frozen systems (mean 0.62 ± 0.21)

Transition Region

S0 galaxies cluster tightly near τ = 0

M87 (NGC 4486) benchmark: τ = −22.5 Myr, ξ/r_core = 0.552

Phase-Transition Indicator

Every τ>0 system satisfies ξ > r_core,
every τ<0 system satisfies ξ < r_core.

This makes τ a natural separator in the sample studied.

Data Sources

(your section is already perfect — minor smoothing only)

Citation
Dataset
@dataset{mcgill_rd_phase_transition_2025,
  author       = {McGill, Lachlan},
  title        = {RD Galaxy Phase-Transition Dataset v1.0},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}

Recommended: cite original data sources

SPARC: Lelli et al. (2016)

ATLAS³D: Cappellari et al. (2011)

Reproducibility Notes

Run the following scripts in order:

compute_kappa1.py

fit_tdgl.py

extended_coherence_params.py

tau_evolutionary_analysis.py

tdgl_jeans_ellipticals.py (Test 6)

Figures:

python scripts/figures/generate_paper_figures_full.py
python scripts/figures/generate_figure5_unified_sequence.py

Contact

GitHub Issues: (add link when live)

Email: (optional)

ORCID: (optional)

Version History

v1.0 (Dec 2025)
First public release with 194 galaxies and complete pipeline.
- ATLAS³D collaboration (Cappellari et al. 2011)

Thanks to the astronomy community for making these datasets publicly available.

---

## Version History

- **v1.0** (December 2025): Initial release
  - 109 SPARC spirals + 89 ATLAS³D ellipticals
  - Complete analysis pipeline
  - 5 publication figures
  - Comprehensive documentation

---

**Dataset DOI:** [To be assigned by Zenodo]  
**Paper DOI:** [To be assigned upon publication]  
**Last Updated:** December 2025
