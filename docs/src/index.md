# Magrathea.jl Documentation

```@raw html
<div class="magrathea-hero">
  <div class="magrathea-eyebrow">Linear stability in rotating spherical shells</div>
  <h1>Spectral eigenvalue problems for rotating convection &amp; MHD.</h1>
  <p>
    Magrathea.jl uses the Olver&ndash;Townsend ultraspherical method to build ultra-sparse,
    spurious-free generalized eigenvalue problems for onset, biglobal, and triglobal
    stability of rotating (magneto)convection in spherical shells.
  </p>
  <div class="magrathea-actions">
    <a class="magrathea-button primary" href="getting_started.html">Get started</a>
    <a class="magrathea-button secondary" href="examples.html">See examples</a>
  </div>
</div>
```

```@raw html
<div class="magrathea-card-grid">
  <div class="magrathea-card">
    <strong>Onset convection</strong>
    <p>Conductive background, single azimuthal wavenumber; find critical Rayleigh numbers.</p>
  </div>
  <div class="magrathea-card">
    <strong>Biglobal (axisymmetric mean flow)</strong>
    <p>Thermal-wind / meridional basic states with an axisymmetric background.</p>
  </div>
  <div class="magrathea-card">
    <strong>Triglobal (non-axisymmetric)</strong>
    <p>Mode-coupled stability for non-axisymmetric basic states across azimuthal wavenumbers.</p>
  </div>
  <div class="magrathea-card">
    <strong>MHD extension</strong>
    <p>Magnetoconvection and kinematic-dynamo problems with no_field, axial, and dipole fields.</p>
  </div>
  <div class="magrathea-card">
    <strong>Spurious-free Galerkin</strong>
    <p>Banded BC-recombined discretization removes the tau spurious-mode swarm; matches collocation to ~1e-12.</p>
  </div>
  <div class="magrathea-card">
    <strong>Unified solver API</strong>
    <p>One <code>solve(problem)</code> entry point across all problem types, returning a <code>StabilityResult</code>.</p>
  </div>
</div>
```

## What To Read First

```@raw html
<div class="magrathea-path">
  <div class="magrathea-step">
    <a href="getting_started.html">Installation</a>
    <p>Install Magrathea.jl and verify your setup.</p>
  </div>
  <div class="magrathea-step">
    <a href="problem_setup.html">First Problem</a>
    <p>Define an OnsetProblem and solve for leading eigenvalues.</p>
  </div>
  <div class="magrathea-step">
    <a href="analysis/index.html">Analysis Modes</a>
    <p>Pick onset, biglobal, or triglobal for your background state.</p>
  </div>
  <div class="magrathea-step">
    <a href="reference.html">API Reference</a>
    <p>Full parameter, problem, and solver reference.</p>
  </div>
</div>
```

## What You Can Do With Magrathea.jl

- **Convection Onset**: Evaluate the onset of convection in rotating spherical shells with flexible mechanical and thermal boundary conditions
- **Critical Rayleigh Numbers**: Find critical parameters using automated bracket search algorithms
- **Basic States**: Build axisymmetric (`BasicState`) or fully 3-D (`BasicState3D`) basic states and assess their linear stability
- **Eigenvalue Analysis**: Track leading eigenpairs via a unified solver interface (in-process, or distributed SLEPc/PETSc)
- **MHD Stability**: Study magnetoconvection and kinematic dynamo problems with the `MHD` submodule
- **Benchmarking**: Validate against published results using provided example scripts

## Quick Start

```julia
using Magrathea

# Define parameters
params = OnsetParams(E=1e-4, Pr=1.0, Ra=1e6, χ=0.35,
                     m=4, lmax=30, Nr=64)

# Create and solve the problem
problem = OnsetProblem(params)
estimate_size(problem)           # check matrix size first
result = solve(problem; nev=6)

# Inspect results
result                           # pretty-printed summary
result.growth_rate               # fastest-growing mode
result.eigenvalues               # all eigenvalues

# Plot eigenvalue spectrum (requires Plots.jl)
using Plots
plot(result)
```

## Architecture Overview

Magrathea.jl is organized into modular components for flexibility and extensibility. See [Codebase Structure](codebase_structure.md) for full details.

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `Magrathea.jl` | Package entry point | Exports public API |
| `Spectral/chebyshev.jl` | Radial discretization | `ChebyshevDiffn` |
| `Stability/linear.jl` | Core stability analysis | `OnsetParams`, `LinearStabilityOperator` |
| `BasicStates/basic_state.jl` | Basic state construction | `BasicState`, `BasicState3D` |
| `Stability/onset.jl` | Mode 1: No mean flow | `OnsetProblem` |
| `Stability/biglobal.jl` | Mode 2: Axisymmetric mean flow | `BiglobalParams` |
| `Stability/triglobal.jl` | Mode 3: Non-axisymmetric | `TriglobalParams`, `CoupledModeProblem` |
| `Operators/sparse_operator.jl` | Sparse ultraspherical operators | sparse A/B assembly |
| `MHD/` | MHD dynamo extension | `MHDParams`, `MHDStabilityOperator` |

## Physical Problem

Magrathea.jl solves the linearized equations for rotating convection in a spherical shell:

```math
\frac{\partial \mathbf{u}}{\partial t} + 2\boldsymbol{\Omega} \times \mathbf{u} = -\nabla p + E\nabla^2\mathbf{u} + \frac{Ra \cdot E^2}{Pr} \Theta \hat{\mathbf{r}}
```

```math
\frac{\partial \Theta}{\partial t} + \mathbf{u} \cdot \nabla T_0 = \frac{E}{Pr} \nabla^2 \Theta
```

```math
\nabla \cdot \mathbf{u} = 0
```

The velocity field is decomposed into poloidal and toroidal components:

```math
\mathbf{u} = \nabla \times \nabla \times (P \hat{\mathbf{r}}) + \nabla \times (T \hat{\mathbf{r}})
```

Fields are expanded in spherical harmonics ``Y_\ell^m(\theta, \phi)`` and Chebyshev polynomials ``T_n(r)`` for spectral accuracy.

## Dimensionless Parameters

| Parameter | Symbol | Definition | Typical Range |
|-----------|--------|------------|---------------|
| Ekman number | ``E`` | ``\nu/(\Omega L^2)`` | ``10^{-3}`` - ``10^{-7}`` |
| Prandtl number | ``Pr`` | ``\nu/\kappa`` | 0.1 - 10 |
| Rayleigh number | ``Ra`` | ``\alpha g \Delta T L^3/(\nu \kappa)`` | ``10^3`` - ``10^8`` |
| Radius ratio | ``\chi`` | ``r_i/r_o`` | 0.2 - 0.9 |

For MHD problems, additional parameters include:

| Parameter | Symbol | Definition |
|-----------|--------|------------|
| Magnetic Prandtl | ``Pm`` | ``\nu/\eta`` |
| Lehnert number | ``Le`` | ``B_0/(\sqrt{\mu\rho}\Omega L)`` |
| Magnetic Ekman | ``E_m`` | ``E/Pm`` |

## Performance Characteristics

The ultraspherical spectral method achieves remarkable sparsity:

| Resolution (Nr) | Dense Storage | Sparse Storage | Sparsity |
|-----------------|---------------|----------------|----------|
| 32 | 8 KB | ~0.2 KB | ~97% |
| 64 | 33 KB | ~1 KB | ~98% |
| 128 | 131 KB | ~4 KB | ~99% |

This translates to:
- **~20x speedup** for matrix-vector products
- **~50x memory reduction** for large problems
- **Linear scaling** with resolution for many operations

## Documentation Guide

### Getting Started
1. **[Installation](getting_started.md)** - Environment setup and first run
2. **[First Problem](problem_setup.md)** - Assemble an onset problem and find critical Rayleigh numbers
3. **[Examples](examples.md)** - Ready-to-run example scripts

### Analysis Modes
4. **[Onset Convection](analysis/onset_convection.md)** - Classical onset with no mean flow
5. **[Biglobal Stability](analysis/biglobal_stability.md)** - Axisymmetric mean flows (thermal wind)
6. **[Triglobal Stability](analysis/triglobal_stability.md)** - Non-axisymmetric mean flows with mode coupling

### Advanced Topics
7. **[Basic States](basic_states.md)** - Construct custom temperature and flow profiles
8. **[Tri-Global Analysis](triglobal.md)** - Technical details of mode coupling
9. **[MHD Extension](mhd_extension.md)** - Magnetic field effects and dynamo problems

### Reference
10. **[API Reference](reference.md)** - Complete function and type documentation
11. **[Codebase Structure](codebase_structure.md)** - Source code organization and architecture
12. **[FAQ](faq.md)** - Troubleshooting and common questions

## Requirements

- **Julia**: 1.10 or newer
- **Dependencies**: LinearAlgebra, SparseArrays, LinearMaps, JLD2, WignerSymbols, SpecialFunctions, Parameters

## Citation

If you use Magrathea.jl in your research, please cite:

```bibtex
@software{magrathea_jl,
  author = {Kar, Subhajit},
  title = {Magrathea.jl: Spectral Methods for Rotating Convection},
  url = {https://github.com/subhk/Magrathea.jl},
  version = {2.0.0},
  year = {2025}
}
```

## License

Magrathea.jl is released under the MIT License.

## Acknowledgments

Magrathea.jl builds upon the mathematical foundations established by:

- Olver & Townsend (2013) - Ultraspherical spectral methods
- Christensen & Wicht (2015) - Numerical dynamo simulations
- The Kore project - Python implementation reference
