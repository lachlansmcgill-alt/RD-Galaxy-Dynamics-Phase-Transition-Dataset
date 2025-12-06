# RD_physics Parameter Definitions

This document provides physical interpretations for all derived parameters in the RD physics pipeline.

## Core TDGL Parameters

### ξ_GL (xi_GL) - GL Coherence Length
**Units:** kpc  
**Range:** Typically 0.1 - 20 kpc for SPARC galaxies  
**Physical Meaning:** The characteristic transition scale where the rotation curve transitions from inner rise to outer flat region. In RD theory, this represents the minimum stable recursion scale at galactic scales.

**Model Context (Two-Stage Approach):**

**Stage 1 - Empirical (Tanh):** `V(r) = V₀ × tanh((r/ξ_GL)^α)`
- Purpose: Robust empirical characterization
- Always fit first to establish baseline
- Best match to observed rotation curves

**Stage 2 - Theoretical (Exponential):** `V(r) = V_flat × (1 - exp(-(r/ξ_GL)^α))`
- Purpose: Test RD theory predictions
- Fit after tanh for comparison
- Parameter agreement validates RD framework

See `docs/MODEL_STRATEGY.md` for complete methodology.

**Typical Values:**
- Dwarf galaxies: 1-5 kpc
- Spiral galaxies: 2-10 kpc
- Massive galaxies: 5-20 kpc

**Warning Signs:**
- ξ_GL < 0.1 kpc: Likely fitting artifact, may indicate poor data quality
- ξ_GL > 50 kpc: Unphysical for rotation curve analysis

---

### α (alpha) - Shape Parameter
**Units:** Dimensionless  
**Range:** Typically 0.5 - 2.5 for physical rotation curves  
**Physical Meaning:** Controls the steepness of the inner rotation curve rise. Related to the inner mass distribution and recursion architecture.

**Interpretation:**
- α ≈ 1: Linear rise (solid-body rotation at small r)
- α < 1: Shallow rise (core-like profile)
- α > 1: Steep rise (cusp-like profile)

**Typical Values:**
- Core-dominated: 0.6 - 1.0
- Mixed: 1.0 - 1.5
- Cusp-dominated: 1.5 - 2.5

**Warning Signs:**
- α < 0.3: Extremely shallow, may indicate data issues or poor fit
- α > 4.0: Unphysically steep, likely fitting artifact

---

### V₀ / V_flat - Asymptotic Velocity
**Units:** km/s  
**Range:** 20 - 350 km/s for SPARC galaxies  
**Physical Meaning:** The flat rotation velocity at large radii, related to total mass enclosed.

---

### r_core - Core Radius
**Units:** kpc  
**Range:** 0.01 - 15 kpc  
**Physical Meaning:** Defined as the radius where V(r_core) = 0.5 × V₀. Represents the effective "core size" of the rotation curve.

**Formula:**
```
r_core = ξ_GL × [arctanh(0.5)]^(1/α)
```

**Physical Context:**
- Smaller r_core: More concentrated mass, steeper inner profile
- Larger r_core: More extended core, shallower inner profile

**Warning Signs:**
- r_core >> ξ_GL: Check for fit convergence issues
- r_core < 0.01 kpc: Likely numerical artifact

---

## Derived RD Parameters

### κ₁ (kappa1) - Coherence Quality Metric
**Units:** Dimensionless  
**Range:** 0 to 1  
**Physical Meaning:** Empirical measure of rotation curve fit quality and "coherence". Higher values indicate better fits and more regular rotation curves.

**Formula:**
```
κ₁ = 1 - (RMS_resid / V_max)
```

**Interpretation:**
- κ₁ > 0.95: Excellent fit, highly coherent
- κ₁ = 0.90 - 0.95: Good fit
- κ₁ < 0.90: Poor fit, may have irregular features

**Physical Context in RD:**
High κ₁ suggests stable recursive structure with minimal perturbations.

---

### η (eta) - Scale Ratio
**Units:** Dimensionless  
**Range:** 0.5 - 10 (typical)  
**Physical Meaning:** Ratio of GL coherence length to core radius.

**Formula:**
```
η = ξ_GL / r_core
```

**Interpretation:**
- η > 1: Coherence scale larger than core (extended coherence)
- η ≈ 1: Coherence and core scales similar
- η < 1: Core larger than coherence scale (compact coherence)

**RD Context:**
May represent the ratio of global recursion scale to local stability scale. Potential correlation with morphology or dynamical state.

---

### τ_unified (tau_unified) - Unified Scale Parameter
**Units:** kpc / (km/s)  
**Range:** -5 to +5 (typical)  
**Physical Meaning:** Signed measure of the relationship between coherence and core scales, normalized by velocity.

**Formula:**
```
Δ = ξ_GL² - r_core²
τ_unified = sign(Δ) × √|Δ| / V_max
```

**Interpretation:**
- τ > 0: ξ_GL > r_core (extended coherence regime)
- τ ≈ 0: ξ_GL ≈ r_core (balanced regime)
- τ < 0: ξ_GL < r_core (compact coherence regime)

**RD Context:**
May represent dimensionless timescale or phase parameter in recursive dynamics. Sign indicates coherence architecture.

---

## Morphology Parameters

### T (Hubble_Type_Num)
**Range:** -6 (elliptical) to +10 (irregular)  
**Physical Meaning:** Numerical encoding of Hubble morphological type.

**Scale:**
- T < 0: Early-type (E, S0)
- T = 0-3: Early spiral (Sa, Sb)
- T = 4-7: Late spiral (Sc, Sd)
- T > 7: Irregular (Sm, Im)

---

## Quality Metrics

### χ²_red (chi2_red)
**Units:** Dimensionless  
**Physical Meaning:** Reduced chi-squared, goodness-of-fit metric.

**Interpretation:**
- χ²_red ≈ 1: Good fit (model matches data within errors)
- χ²_red << 1: Overfit or overestimated errors
- χ²_red >> 1: Poor fit or underestimated errors

### RMS_resid
**Units:** km/s  
**Physical Meaning:** Root-mean-square of velocity residuals (observed - model).

**Typical Values:**
- < 2 km/s: Excellent fit
- 2-5 km/s: Good fit
- > 10 km/s: Poor fit, irregular features present

---

## Notes on Physical Interpretation

### RD Theory Context & Model Strategy

In Recursive Dynamics theory:
- **ξ_GL** is proposed as the minimum stable recursion scale at galactic scales
- **α** may relate to recursive architecture (how sub-scales nest)
- **η** and **τ** are exploratory parameters testing scale relationships
- **κ₁** tests whether rotation curves show coherent recursive structure

### Two-Stage Validation Approach

**Key Insight:** We use two models sequentially to test RD:

1. **Tanh Model (Empirical)**: Establishes robust baseline parameters
   - No theoretical assumptions
   - Best possible fit to data
   - Reference values for ξ_GL, α, V₀

2. **Exponential Model (Theoretical)**: Tests RD predictions
   - Functional form predicted by RD theory
   - Should yield similar parameters if RD correct
   - Agreement = evidence for RD framework

**Validation Criterion:**
If |ξ_GL(exp) - ξ_GL(tanh)| / ξ_GL(tanh) < 0.3 (30%) for most galaxies,
this provides strong evidence that RD correctly predicts rotation curve structure.

See `docs/MODEL_STRATEGY.md` for complete methodology.

### Correlations to Test

1. Does ξ_GL correlate with morphology (T)?
2. Is κ₁ independent of galaxy properties (as RD predicts)?
3. Do η and τ show morphological dependence?
4. How does r_core / ξ_GL vary with dynamical state?

### Caveats

- All parameters assume TDGL model is appropriate
- Fits are phenomenological, not ab initio physics
- Physical interpretation within RD framework is speculative pending validation
- Parameter warnings in fit outputs should be checked for outliers
