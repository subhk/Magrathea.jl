# Lower-Level API

Most workflows only need the problem/solve interface in the
[API Reference](reference.md). The exports below give direct access to parameter
validation, basic-state construction, operator assembly and spectral couplings.

```@meta
CurrentModule = Magrathea
```

## Results, sizing and plotting

```@docs
AbstractStabilityResult
eigenspectrum
find_growth_rate
estimate_onset_problem_size
estimate_triglobal_problem_size
onset_scaling_laws
plot_meridional
plot_radial
```

## Critical-parameter search

```@docs
find_critical_rayleigh
```

## Validation

```@docs
validate_onset_params
validate_biglobal_params
validate_triglobal_params
validate_mhd_params
validate_basic_state_consistency
validate_basic_state_3d_consistency
```

## Basic-state construction and analysis

```@docs
create_conduction_basic_state
create_custom_basic_state
analyze_basic_state
compare_onset_vs_biglobal
sweep_thermal_wind_amplitude
solve_thermal_wind_balance!
solve_thermal_wind_balance_3d!
solve_meridional_coupled!
solve_meridional_simple!
solve_meridional_circulation_toroidal_poloidal!
solve_poisson_mode
AdvectionDiffusionSolver
compute_full_advection_spectral
BasicStateOperators
build_basic_state_operators
add_basic_state_operators!
potentials_to_velocity
compute_l_sets
setup_coupled_mode_problem
```

## Spherical-harmonic boundary conditions

```@docs
Ylm
Y00
Y10
Y11
Y20
Y21
Y22
Y30
Y31
Y32
Y33
Y40
Y41
Y42
Y43
Y44
to_dict
get_lmax
get_mmax
get_lmax_mmax
is_axisymmetric
```

## Angular coupling coefficients

```@docs
cos_theta_coupling
theta_derivative_coupling
inv_sin_theta_gaunt
inv_sin_theta_coupling
```

## Sparse and MHD operators

```@docs
SparseOnsetParams
SparseStabilityOperator
assemble_sparse_matrices
BackgroundField
no_field
axial
dipole
```
