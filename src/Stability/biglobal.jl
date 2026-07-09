# =============================================================================
#  Biglobal Stability Analysis with Axisymmetric Mean Flow
#
#  Linear stability analysis for rotating spherical shells with an axisymmetric
#  (m=0) background mean flow. This extends the classical onset problem by
#  including thermal wind-balanced zonal flows.
#
#  Physical scenario:
#  ------------------
#  - Basic state: θ̄(r,θ) = temperature with latitudinal variation
#                 ū_φ(r,θ) = zonal flow from thermal wind balance
#  - Each perturbation mode m remains INDEPENDENT (no mode coupling)
#  - Mean flow advection and shear modify growth rates and drift frequencies
#
#  Applications:
#  - Thermal wind from pole-equator temperature differences
#  - Differentially rotating boundaries
#  - Pre-existing zonal jets
#  - CMB heat flux variations (axisymmetric component)
#
#  Key distinction from triglobal:
#  - Biglobal: basic state has m_bs = 0 only → NO mode coupling
#  - Triglobal: basic state has m_bs ≠ 0 → modes couple
# =============================================================================

# Dependencies provided by Magrathea module:
# Parameters, LinearAlgebra, Printf
# LinearStabilityOperator, OnsetParams, BasicState,
# assemble_matrices, solve_eigenvalue_problem, find_growth_rate,
# ChebyshevDiffn, conduction_basic_state, meridional_basic_state,
# find_critical_rayleigh are available in the Magrathea namespace

"""
    BiglobalParams{T<:Real}

Parameters for biglobal stability analysis with axisymmetric mean flow.

The biglobal analysis differs from onset analysis by including an axisymmetric
basic state with:
- Temperature variations: θ̄(r,θ) = Σ_ℓ θ̄_ℓ0(r) Y_ℓ0(θ)
- Thermal wind flow: ū_φ(r,θ) = Σ_ℓ ū_φ,ℓ0(r) Y_ℓ0(θ)

Each perturbation mode m is still solved independently, but the operators
include advection and shear terms from the basic state.

# Fields
- `E::T` - Ekman number
- `Pr::T` - Prandtl number
- `Ra::T` - Rayleigh number
- `χ::T` - Radius ratio r_i/r_o
- `m::Int` - Azimuthal wavenumber of perturbation
- `lmax::Int` - Maximum spherical harmonic degree
- `Nr::Int` - Number of radial points
- `basic_state::BasicState{T}` - Axisymmetric basic state (required)
- `mechanical_bc::Symbol` - :no_slip or :stress_free
- `thermal_bc::Symbol` - :fixed_temperature or :fixed_flux
- `equatorial_symmetry::Symbol` - :both, :symmetric, or :antisymmetric

# Example
```julia
# Create basic state with thermal wind
cd = ChebyshevDiffn(64, [0.35, 1.0], 4)
bs = meridional_basic_state(cd, 0.35, 1e-5, 1e7, 1.0, 6, 0.1)

params = BiglobalParams(
    E = 1e-5, Pr = 1.0, Ra = 1e7, χ = 0.35,
    m = 10, lmax = 60, Nr = 64,
    basic_state = bs
)
```

See also: [`solve_biglobal_problem`](@ref), [`create_thermal_wind_basic_state`](@ref)
"""
@with_kw_noshow struct BiglobalParams{T<:Real}
    E::T
    Pr::T = one(T)
    Ra::T
    χ::T
    m::Int
    lmax::Int
    Nr::Int
    basic_state::BasicState{T}
    mechanical_bc::Symbol = :no_slip
    thermal_bc::Symbol = :fixed_temperature
    equatorial_symmetry::Symbol = :both

    function BiglobalParams{T}(E, Pr, Ra, χ, m, lmax, Nr, basic_state,
                               mechanical_bc, thermal_bc,
                               equatorial_symmetry) where T
        0 < χ < 1 || throw(ArgumentError(
            "Radius ratio χ must be in (0,1), got $χ"))
        E > 0 || throw(ArgumentError(
            "Ekman number E must be positive, got $E"))
        Pr > 0 || throw(ArgumentError(
            "Prandtl number Pr must be positive, got $Pr"))
        Ra >= 0 || throw(ArgumentError(
            "Rayleigh number Ra must be non-negative, got $Ra"))
        m >= 0 || throw(ArgumentError(
            "Azimuthal wavenumber m must be non-negative, got $m"))
        lmax >= m || throw(ArgumentError(
            "lmax must be >= m, got lmax=$lmax, m=$m"))
        Nr >= 8 || throw(ArgumentError(
            "Nr must be >= 8 for meaningful resolution, got $Nr"))
        mechanical_bc in (:no_slip, :stress_free) || throw(ArgumentError(
            "mechanical_bc must be :no_slip or :stress_free, got :$mechanical_bc"))
        thermal_bc in (:fixed_temperature, :fixed_flux) || throw(ArgumentError(
            "thermal_bc must be :fixed_temperature or :fixed_flux, got :$thermal_bc"))
        equatorial_symmetry in (:both, :symmetric, :antisymmetric) || throw(ArgumentError(
            "equatorial_symmetry must be :both, :symmetric, or :antisymmetric, got :$equatorial_symmetry"))
        basic_state.Nr == Nr || throw(ArgumentError(
            "basic_state.Nr=$(basic_state.Nr) doesn't match Nr=$Nr"))
        length(basic_state.r) == Nr || throw(ArgumentError(
            "basic_state.r must have length Nr=$Nr, got $(length(basic_state.r))"))
        tol = sqrt(eps(float(T)))
        isapprox(first(basic_state.r), χ; rtol=tol, atol=tol) || throw(ArgumentError(
            "basic_state.r must start at χ=$χ, got $(first(basic_state.r))"))
        isapprox(last(basic_state.r), one(T); rtol=tol, atol=tol) || throw(ArgumentError(
            "basic_state.r must end at 1, got $(last(basic_state.r))"))
        expected_grid = ChebyshevDiffn(Nr, [χ, one(T)], 1).x
        grid_error = _grid_mismatch(basic_state.r, expected_grid)
        grid_tol = T(10) * tol
        grid_error <= grid_tol || throw(ArgumentError(
            "basic_state.r must match Chebyshev nodes (max mismatch = $grid_error)"))
        for coeffs in values(basic_state.theta_coeffs)
            length(coeffs) == Nr || throw(ArgumentError(
                "basic_state.theta_coeffs entries must have length Nr=$Nr"))
        end
        for coeffs in values(basic_state.uphi_coeffs)
            length(coeffs) == Nr || throw(ArgumentError(
                "basic_state.uphi_coeffs entries must have length Nr=$Nr"))
        end
        for coeffs in values(basic_state.dtheta_dr_coeffs)
            length(coeffs) == Nr || throw(ArgumentError(
                "basic_state.dtheta_dr_coeffs entries must have length Nr=$Nr"))
        end
        for coeffs in values(basic_state.duphi_dr_coeffs)
            length(coeffs) == Nr || throw(ArgumentError(
                "basic_state.duphi_dr_coeffs entries must have length Nr=$Nr"))
        end

        new{T}(E, Pr, Ra, χ, m, lmax, Nr, basic_state,
               mechanical_bc, thermal_bc, equatorial_symmetry)
    end
end


# =============================================================================
#  Basic State Creation Functions
# =============================================================================

"""
    create_conduction_basic_state(χ::T, Nr::Int; lmax_bs::Int=6) where T

Create a conduction basic state with no flow (reference state).

This creates a basic state with:
- θ̄(r) = conductive temperature profile (only ℓ=0)
- ū_φ = 0 everywhere

Useful as a reference for comparing with thermal wind cases.

# Arguments
- `χ::Real` - Radius ratio
- `Nr::Int` - Number of radial points
- `lmax_bs::Int` - Maximum ℓ for basic state (default: 6)

# Returns
- `BasicState` - Conduction basic state
- `ChebyshevDiffn` - Chebyshev differentiation structure
"""
function create_conduction_basic_state(χ::T, Nr::Int; lmax_bs::Int=6) where T<:Real
    cd = ChebyshevDiffn(Nr, [χ, one(T)], 4)
    bs = conduction_basic_state(cd, χ, lmax_bs)
    return bs, cd
end


"""
    create_thermal_wind_basic_state(χ, E, Ra, Pr, Nr;
                                    amplitude=0.1, lmax_bs=6,
                                    mechanical_bc=:no_slip)

Create a basic state with meridional temperature variation and thermal wind.

The outer boundary temperature varies with latitude:
    θ̄(r_o, θ) = 1 + amplitude × Y_20(θ)

This drives a zonal flow through thermal wind balance:
    2Ω cos(θ) ∂ū_φ/∂r = -(Ra E²/Pr) × (1/r) × ∂θ̄/∂θ

# Arguments
- `χ::Real` - Radius ratio
- `E::Real` - Ekman number (required for thermal wind scaling)
- `Ra::Real` - Rayleigh number
- `Pr::Real` - Prandtl number
- `Nr::Int` - Number of radial points
- `amplitude::Real` - Amplitude of Y_20 boundary variation (default: 0.1)
- `lmax_bs::Int` - Maximum ℓ for basic state (default: 6)
- `mechanical_bc::Symbol` - Boundary conditions (default: :no_slip)
- `thermal_bc::Symbol` - Thermal boundary conditions (default: :fixed_temperature)
- `outer_flux_mean::Real` - Mean outer flux for fixed-flux states
- `outer_flux_Y20::Real` - Y20 outer flux for fixed-flux states

# Returns
- `BasicState` - Basic state with thermal wind
- `ChebyshevDiffn` - Chebyshev differentiation structure

# Example
```julia
bs, cd = create_thermal_wind_basic_state(0.35, 1e-5, 1e7, 1.0, 64;
                                          amplitude=0.1)
```

# Physics
The thermal wind amplitude scales as:
    ū_φ ~ (Ra E² / Pr) × amplitude

For typical parameters (Ra~10⁷, E~10⁻⁵), this gives ū_φ ~ amplitude.
"""
function create_thermal_wind_basic_state(χ::T, E::T, Ra::T, Pr::T, Nr::Int;
                                          amplitude::T=T(0.1),
                                          lmax_bs::Int=6,
                                          mechanical_bc::Symbol=:no_slip,
                                          thermal_bc::Symbol=:fixed_temperature,
                                          outer_flux_mean::T=zero(T),
                                          outer_flux_Y20::T=zero(T)) where T<:Real
    cd = ChebyshevDiffn(Nr, [χ, one(T)], 4)
    bs = meridional_basic_state(cd, χ, E, Ra, Pr, lmax_bs, amplitude;
                                mechanical_bc=mechanical_bc,
                                thermal_bc=thermal_bc,
                                outer_flux_mean=outer_flux_mean,
                                outer_flux_Y20=outer_flux_Y20)
    return bs, cd
end


"""
    create_custom_basic_state(θ_profile, uphi_profile, r_grid; lmax_bs=6, χ=r_grid[1])

Create a basic state from custom radial profiles.

This allows importing basic states from external simulations or
analytical solutions.

# Arguments
- `θ_profile::Function` - Temperature profile θ̄(r, ℓ) returning θ̄_ℓ0(r)
- `uphi_profile::Function` - Zonal flow profile ū_φ(r, ℓ) returning ū_φ,ℓ0(r)
- `r_grid::Vector` - Radial collocation points (Chebyshev nodes for [χ, 1])
- `lmax_bs::Int` - Maximum ℓ for basic state

# Returns
- `BasicState` - Custom basic state

# Example
```julia
# Gaussian jet profile
r = cd.x
θ_prof(r, ℓ) = ℓ == 0 ? conduction_profile(r) : zeros(length(r))
uphi_prof(r, ℓ) = ℓ == 1 ? 0.1 * exp.(-((r .- 0.7)/0.1).^2) : zeros(length(r))

bs = create_custom_basic_state(θ_prof, uphi_prof, r; lmax_bs=4)
```
"""
function create_custom_basic_state(θ_profile::Function,
                                    uphi_profile::Function,
                                    r_grid::Vector{T};
                                    lmax_bs::Int=6,
                                    χ::T=r_grid[1]) where T<:Real
    Nr = length(r_grid)

    # Build Chebyshev for derivatives
    cd = ChebyshevDiffn(Nr, [χ, one(T)], 4)
    tol = sqrt(eps(float(T)))
    grid_error = _grid_mismatch(r_grid, cd.x)
    grid_error <= tol || throw(ArgumentError(
        "r_grid must match Chebyshev nodes for [χ, 1] (max mismatch = $grid_error)"))

    # Initialize coefficient dictionaries
    theta_coeffs = Dict{Int, Vector{T}}()
    uphi_coeffs = Dict{Int, Vector{T}}()
    dtheta_dr_coeffs = Dict{Int, Vector{T}}()
    duphi_dr_coeffs = Dict{Int, Vector{T}}()

    # Populate coefficients
    for ℓ in 0:lmax_bs
        theta_coeffs[ℓ] = θ_profile(r_grid, ℓ)
        uphi_coeffs[ℓ] = uphi_profile(r_grid, ℓ)

        # Compute derivatives
        dtheta_dr_coeffs[ℓ] = cd.D1 * theta_coeffs[ℓ]
        duphi_dr_coeffs[ℓ] = cd.D1 * uphi_coeffs[ℓ]
    end

    return BasicState(
        lmax_bs = lmax_bs,
        Nr = Nr,
        r = r_grid,
        theta_coeffs = theta_coeffs,
        uphi_coeffs = uphi_coeffs,
        dtheta_dr_coeffs = dtheta_dr_coeffs,
        duphi_dr_coeffs = duphi_dr_coeffs
    )
end


# =============================================================================
#  Main Solver Functions
# =============================================================================

"""
    solve_biglobal_problem(params::BiglobalParams; nev=6, kwargs...)

Solve the biglobal stability eigenvalue problem.

This solves the linearized equations about an axisymmetric basic state:
    A x = σ B x

where A includes advection by the mean flow and shear production terms.

# Arguments
- `params::BiglobalParams` - Problem parameters (including basic state)
- `nev::Int` - Number of eigenvalues to compute (default: 6)
- `tol::Float64` - Eigenvalue solver tolerance (default: 1e-10)
- `which::Symbol` - Target eigenvalues: :LR (largest real), :LM (largest magnitude)

# Returns
- `eigenvalues::Vector{ComplexF64}` - Complex growth rates
- `eigenvectors::Vector{Vector{ComplexF64}}` - Corresponding eigenmodes
- `operator::LinearStabilityOperator` - The assembled operator
- `info` - Solver convergence information

# Example
```julia
bs, cd = create_thermal_wind_basic_state(0.35, 1e-5, 1e7, 1.0, 64)
params = BiglobalParams(E=1e-5, Pr=1.0, Ra=1e7, χ=0.35, m=10,
                        lmax=60, Nr=64, basic_state=bs)

eigenvalues, eigenvectors, op, info = solve_biglobal_problem(params; nev=8)
```
"""
function solve_biglobal_problem(params::BiglobalParams{T};
                                 nev::Int=6,
                                 tol::Float64=1e-10,
                                 maxiter::Int=1000,
                                 which::Symbol=:LR,
                                 sigma=nothing,
                                 backend::Symbol=:slepc,
                                 verbose::Bool=false) where T

    if verbose
        println("Setting up biglobal stability problem...")
        println("  m = $(params.m), lmax = $(params.lmax), Nr = $(params.Nr)")
        println("  Basic state lmax_bs = $(params.basic_state.lmax_bs)")
    end

    # Convert to internal OnsetParams with basic_state
    internal_params = OnsetParams(
        E = params.E,
        Pr = params.Pr,
        Ra = params.Ra,
        χ = params.χ,
        m = params.m,
        lmax = params.lmax,
        Nr = params.Nr,
        mechanical_bc = params.mechanical_bc,
        thermal_bc = params.thermal_bc,
        equatorial_symmetry = params.equatorial_symmetry,
        basic_state = params.basic_state  # Include axisymmetric basic state
    )

    # Build operator (will include basic state operators automatically)
    op = LinearStabilityOperator(internal_params)

    if verbose
        println("  Total DOFs: $(op.total_dof)")
    end

    # Solve eigenvalue problem
    if backend === :slepc
        # Distributed constrained-reduction path (see solve_onset_problem): distributes
        # the full tau pencil + S/P, forms S·A·P / S·B·P via MatMatMult, runs the EPS
        # solve, and reconstructs eigenvectors to full DOFs on rank 0.
        eigenvalues, eigenvectors, info = Magrathea._solve_constrained_slepc(op;
            nev=nev, sigma=sigma, which=which, tol=tol, maxiter=maxiter)
    else
        eigenvalues, eigenvectors, info = solve_eigenvalue_problem(op;
            nev=nev, tol=tol, maxiter=maxiter, which=which, sigma=sigma, backend=backend)
    end

    if verbose
        println("  Solved: $(length(eigenvalues)) eigenvalues found")
    end

    return eigenvalues, eigenvectors, op, info
end


function _biglobal_rayleigh_kwargs(mechanical_bc::Symbol,
                                    thermal_bc::Symbol,
                                    equatorial_symmetry::Symbol,
                                    nev::Int,
                                    basic_state,
                                    basic_state_builder)
    rayleigh_kwargs = (; mechanical_bc=mechanical_bc,
                        thermal_bc=thermal_bc,
                        equatorial_symmetry=equatorial_symmetry,
                        nev=nev)

    if basic_state !== nothing
        rayleigh_kwargs = (; rayleigh_kwargs..., basic_state=basic_state)
    end

    if basic_state_builder !== nothing
        builder_wrapper = function (Ra_val)
            bs = basic_state_builder(Ra_val)
            bs isa BasicState || error("basic_state_builder must return a BasicState, got $(typeof(bs))")
            return bs
        end
        rayleigh_kwargs = (; rayleigh_kwargs..., basic_state_builder=builder_wrapper)
    end

    return rayleigh_kwargs
end


"""
    find_critical_Ra_biglobal(; E, Pr, χ, m, lmax, Nr, basic_state, kwargs...)

Find the critical Rayleigh number with an axisymmetric basic state.

Uses bisection to find Ra_c where the leading growth rate σ = 0.

Note: The basic state should be constructed at a reference Ra value.
For self-consistent analysis, the basic state amplitude may need to
scale with Ra.

# Arguments
- `E::Real` - Ekman number
- `Pr::Real` - Prandtl number
- `χ::Real` - Radius ratio
- `m::Int` - Azimuthal wavenumber
- `lmax::Int` - Maximum spherical harmonic degree
- `Nr::Int` - Number of radial points
- `basic_state::BasicState` - Fixed axisymmetric basic state (optional)
- `basic_state_builder::Function` - Callback `f(Ra) -> BasicState` for Ra-dependent states (optional)
- `Ra_guess::Real` - Initial guess for Ra_c
- `tol::Real` - Tolerance for convergence
- `equatorial_symmetry::Symbol` - :both, :symmetric, or :antisymmetric

# Returns
- `Ra_c::Real` - Critical Rayleigh number
- `ω_c::Real` - Drift frequency at onset
- `eigenvector` - Critical eigenmode
"""
function find_critical_Ra_biglobal(; E::Real, Pr::Real, χ::Real, m::Int, lmax::Int, Nr::Int,
                                    basic_state::Union{BasicState,Nothing}=nothing,
                                    basic_state_builder::Union{Nothing,Function}=nothing,
                                    Ra_guess::Real=1e6,
                                    tol::Real=1e-6,
                                    Ra_bracket::Tuple{<:Real,<:Real}=(Ra_guess/10, Ra_guess*10),
                                    mechanical_bc::Symbol=:no_slip,
                                    thermal_bc::Symbol=:fixed_temperature,
                                    equatorial_symmetry::Symbol=:both,
                                    nev::Int=6,
                                    verbose::Bool=false)

    if (basic_state === nothing) && (basic_state_builder === nothing)
        error("find_critical_Ra_biglobal requires either `basic_state` or `basic_state_builder`")
    elseif (basic_state !== nothing) && (basic_state_builder !== nothing)
        error("Provide only one of `basic_state` or `basic_state_builder`")
    end

    equatorial_symmetry in (:both, :symmetric, :antisymmetric) || throw(ArgumentError(
        "equatorial_symmetry must be :both, :symmetric, or :antisymmetric, got :$equatorial_symmetry"))

    T = promote_type(typeof(E), typeof(Pr), typeof(χ), typeof(Ra_guess),
                     typeof(tol), typeof(Ra_bracket[1]), typeof(Ra_bracket[2]))

    rayleigh_kwargs = _biglobal_rayleigh_kwargs(
        mechanical_bc, thermal_bc, equatorial_symmetry, nev,
        basic_state, basic_state_builder)

    Ra_c, ω_c, vec_c = find_critical_rayleigh(
        T(E), T(Pr), T(χ), m, lmax, Nr;
        Ra_guess=T(Ra_guess), tol=T(tol),
        Ra_bracket=(T(Ra_bracket[1]), T(Ra_bracket[2])),
        rayleigh_kwargs...
    )

    if verbose
        @printf("  m = %d (biglobal): Ra_c = %.6e, ω_c = %+.6f\n", m, Ra_c, ω_c)
    end

    return Ra_c, ω_c, vec_c
end


"""
    compare_onset_vs_biglobal(; E, Pr, χ, m, lmax, Nr, Ra,
                               basic_state_amplitude=0.1,
                               verbose=true)

Compare stability with and without mean flow at a given Ra.

This is useful for understanding how the thermal wind affects stability:
- Stabilization: σ_biglobal < σ_onset
- Destabilization: σ_biglobal > σ_onset

# Arguments
- `E, Pr, χ, m, lmax, Nr, Ra` - Standard parameters
- `basic_state_amplitude::Real` - Amplitude of Y_20 variation (default: 0.1)
- `verbose::Bool` - Print comparison (default: true)

# Returns
- `σ_onset::Real` - Growth rate without mean flow
- `σ_biglobal::Real` - Growth rate with thermal wind
- `Δσ::Real` - Difference (positive → destabilization)
- `ω_onset::Real` - Drift frequency without mean flow
- `ω_biglobal::Real` - Drift frequency with thermal wind
"""
function compare_onset_vs_biglobal(; E::Real, Pr::Real, χ::Real, m::Int, lmax::Int, Nr::Int, Ra::Real,
                                    basic_state_amplitude::Real=0.1,
                                    mechanical_bc::Symbol=:no_slip,
                                    thermal_bc::Symbol=:fixed_temperature,
                                    verbose::Bool=true)
    T = float(promote_type(typeof(E), typeof(Pr), typeof(χ), typeof(Ra), typeof(basic_state_amplitude)))
    E, Pr, χ, Ra = T(E), T(Pr), T(χ), T(Ra)
    basic_state_amplitude = T(basic_state_amplitude)

    if verbose
        println("="^60)
        println("Comparison: Onset vs Biglobal")
        println("="^60)
        @printf("  E = %.2e, Pr = %.1f, Ra = %.2e\n", E, Pr, Ra)
        @printf("  χ = %.2f, m = %d\n", χ, m)
        @printf("  Thermal wind amplitude = %.2f\n", basic_state_amplitude)
        println()
    end

    # Onset (no mean flow)
    params_onset = OnsetParams(
        E=E, Pr=Pr, Ra=Ra, χ=χ, m=m, lmax=lmax, Nr=Nr,
        mechanical_bc=mechanical_bc, thermal_bc=thermal_bc,
        basic_state=nothing
    )
    op_onset = LinearStabilityOperator(params_onset)
    σ_onset, ω_onset, _ = find_growth_rate(op_onset)

    # Biglobal (with thermal wind)
    bs, cd = create_thermal_wind_basic_state(χ, E, Ra, Pr, Nr;
                                              amplitude=basic_state_amplitude,
                                              mechanical_bc=mechanical_bc,
                                              thermal_bc=thermal_bc)
    params_biglobal = OnsetParams(
        E=E, Pr=Pr, Ra=Ra, χ=χ, m=m, lmax=lmax, Nr=Nr,
        mechanical_bc=mechanical_bc, thermal_bc=thermal_bc,
        basic_state=bs
    )
    op_biglobal = LinearStabilityOperator(params_biglobal)
    σ_biglobal, ω_biglobal, _ = find_growth_rate(op_biglobal)

    # Compare
    Δσ = σ_biglobal - σ_onset
    Δω = ω_biglobal - ω_onset

    if verbose
        println("Results:")
        @printf("  Onset (no flow):    σ = %+.6e, ω = %+.6f\n", σ_onset, ω_onset)
        @printf("  Biglobal (thermal): σ = %+.6e, ω = %+.6f\n", σ_biglobal, ω_biglobal)
        println()
        @printf("  Difference: Δσ = %+.6e ", Δσ)
        if Δσ > 0
            println("(DESTABILIZING)")
        elseif Δσ < 0
            println("(STABILIZING)")
        else
            println("(NEUTRAL)")
        end
        @printf("              Δω = %+.6f\n", Δω)
    end

    return (
        σ_onset = σ_onset,
        σ_biglobal = σ_biglobal,
        Δσ = Δσ,
        ω_onset = ω_onset,
        ω_biglobal = ω_biglobal,
        Δω = Δω
    )
end


"""
    sweep_thermal_wind_amplitude(; E, Pr, χ, m, lmax, Nr, Ra,
                                  amplitudes, verbose=true)

Sweep over thermal wind amplitudes to study stabilization/destabilization.

# Arguments
- Standard parameters: E, Pr, χ, m, lmax, Nr, Ra
- `amplitudes::Vector` - Amplitudes to sweep (e.g., [0, 0.05, 0.1, 0.15, 0.2])
- `verbose::Bool` - Print results

# Returns
- `results::Vector{NamedTuple}` - Growth rates and frequencies for each amplitude
"""
function sweep_thermal_wind_amplitude(; E::Real, Pr::Real, χ::Real, m::Int, lmax::Int, Nr::Int, Ra::Real,
                                       amplitudes::AbstractVector{<:Real},
                                       mechanical_bc::Symbol=:no_slip,
                                       thermal_bc::Symbol=:fixed_temperature,
                                       verbose::Bool=true)
    T = float(promote_type(typeof(E), typeof(Pr), typeof(χ), typeof(Ra), eltype(amplitudes)))
    E, Pr, χ, Ra = T(E), T(Pr), T(χ), T(Ra)
    amplitudes = T.(amplitudes)

    if verbose
        println("="^60)
        println("Thermal Wind Amplitude Sweep")
        println("="^60)
        @printf("  E = %.2e, Pr = %.1f, Ra = %.2e, m = %d\n", E, Pr, Ra, m)
        println()
        @printf("  %-10s  %-14s  %-14s\n", "Amplitude", "σ", "ω")
        println("  " * "-"^40)
    end

    results = NamedTuple{(:amplitude, :σ, :ω), Tuple{T, T, T}}[]

    for amp in amplitudes
        if amp == 0
            # No mean flow case
            params = OnsetParams(
                E=E, Pr=Pr, Ra=Ra, χ=χ, m=m, lmax=lmax, Nr=Nr,
                mechanical_bc=mechanical_bc, thermal_bc=thermal_bc,
                basic_state=nothing
            )
        else
            bs, _ = create_thermal_wind_basic_state(χ, E, Ra, Pr, Nr;
                                                     amplitude=amp,
                                                     mechanical_bc=mechanical_bc,
                                                     thermal_bc=thermal_bc)
            params = OnsetParams(
                E=E, Pr=Pr, Ra=Ra, χ=χ, m=m, lmax=lmax, Nr=Nr,
                mechanical_bc=mechanical_bc, thermal_bc=thermal_bc,
                basic_state=bs
            )
        end

        op = LinearStabilityOperator(params)
        σ, ω, _ = find_growth_rate(op)

        push!(results, (amplitude=amp, σ=σ, ω=ω))

        if verbose
            @printf("  %-10.3f  %+.8e  %+.8f\n", amp, σ, ω)
        end
    end

    isempty(results) && return results

    amp_tol = sqrt(eps(float(one(T))))
    baseline_idx = findfirst(i -> abs(results[i].amplitude) <= amp_tol, eachindex(results))
    baseline_idx === nothing && (baseline_idx = 1)

    if verbose
        println()
        if abs(results[baseline_idx].amplitude) <= amp_tol
            @printf("  Reference amplitude (≈0): %.3f\n", results[baseline_idx].amplitude)
        else
            @printf("  Reference amplitude (no zero provided): %.3f\n", results[baseline_idx].amplitude)
        end
        σ_ref = results[baseline_idx].σ
        for r in results
            Δσ = r.σ - σ_ref
            effect = Δσ > 1e-10 ? "destabilizing" : (Δσ < -1e-10 ? "stabilizing" : "neutral")
            @printf("  amp = %.3f: Δσ = %+.4e (%s)\n", r.amplitude, Δσ, effect)
        end
    end

    return results
end


"""
    analyze_basic_state(bs::BasicState; verbose=true)

Analyze and summarize a basic state.

# Returns
- Summary of temperature and flow amplitudes per ℓ mode
- Maximum values and their locations
- Diagnostic information
"""
function analyze_basic_state(bs::BasicState{T}; verbose::Bool=true) where T
    if verbose
        println("="^60)
        println("Basic State Analysis")
        println("="^60)
        println("  lmax_bs = $(bs.lmax_bs)")
        println("  Nr = $(bs.Nr)")
        println("  r ∈ [$(minimum(bs.r)), $(maximum(bs.r))]")
        println()
    end

    Summary = NamedTuple{(:θ_max, :uphi_max), Tuple{T, T}}
    results = Dict{Int, Summary}()

    if verbose
        @printf("  %-4s  %-14s  %-14s\n", "ℓ", "max|θ̄_ℓ|", "max|ū_φ,ℓ|")
        println("  " * "-"^35)
    end

    for ℓ in sort(collect(keys(bs.theta_coeffs)))
        θ_max = maximum(abs, bs.theta_coeffs[ℓ])
        uphi_max = haskey(bs.uphi_coeffs, ℓ) ? maximum(abs, bs.uphi_coeffs[ℓ]) : zero(T)

        results[ℓ] = (θ_max=θ_max, uphi_max=uphi_max)

        if verbose
            @printf("  %-4d  %.8e  %.8e\n", ℓ, θ_max, uphi_max)
        end
    end

    return results
end


# Exports are centralized in Magrathea.jl
