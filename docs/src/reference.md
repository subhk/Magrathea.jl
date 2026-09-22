# API Reference

```@meta
CurrentModule = Magrathea
```

## Parameters

```@docs
OnsetParams
OnsetConvectionParams
BiglobalParams
TriglobalParams
MHDParams
```

## Problems & solve

```@docs
OnsetProblem
BiglobalProblem
TriglobalProblem
MHDProblem
solve
solve_onset_problem
solve_biglobal_problem
estimate_size
StabilityResult
growth_rate
frequency
leading_mode
```

## Critical-parameter search

```@docs
find_critical_Ra
find_critical_Ra_onset
find_global_critical_onset
find_critical_Ra_biglobal
find_critical_rayleigh_triglobal
```

## Basic states

```@docs
basic_state
conduction_basic_state
meridional_basic_state
nonaxisymmetric_basic_state
basic_state_selfconsistent
nonaxisymmetric_basic_state_selfconsistent
mean_flow_velocity
create_thermal_wind_basic_state
BasicState
BasicState3D
SphericalHarmonicBC
```

## Spectral & operators

```@docs
ChebyshevDiffn
LinearStabilityOperator
assemble_matrices
```

## MHD operators

```@docs
MHDStabilityOperator
assemble_mhd_matrices
```

## Eigensolver lifecycle and field reconstruction

Sparse eigensolves require the [SLEPc setup](getting_started.md#SLEPc-setup).

```@docs
slepc_init!
slepc_finalize!
solve_eigenvalue_problem
perturbation_velocity
perturbation_temperature
perturbation_magnetic
```
