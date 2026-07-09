module Magrathea

    using LinearAlgebra
    using SparseArrays
    using JLD2
    using Printf
    using Random

    using Parameters

    using LinearMaps
    using WignerSymbols
    using SpecialFunctions

    # ---- Organized source files ----

    # 1. Spectral methods (ChebyshevDiffn, ultraspherical operators)
    include("Spectral/Spectral.jl")

    # 2. Basic states (needs ChebyshevDiffn from Spectral)
    include("BasicStates/BasicStates.jl")

    # 3. Stability analysis (needs ChebyshevDiffn + BasicState types)
    include("Stability/Stability.jl")

    # 4. Sparse operators and boundary conditions
    include("Operators/Operators.jl")

    # 5. MHD extensions (needs ultraspherical + sparse operator functions)
    include("MHD/MHD.jl")

    # 6. v2.0 API layer
    include("validation.jl")
    include("types.jl")
    include("solve.jl")
    include("show.jl")

    export
        # Core utilities
        ChebyshevDiffn,
        potentials_to_velocity,
        compute_l_sets,

        # Base types and functions (shared)
        OnsetParams,
        LinearStabilityOperator,
        solve_eigenvalue_problem,
        slepc_init!,
        slepc_finalize!,
        find_growth_rate,
        find_critical_rayleigh,
        assemble_matrices,

        # Basic state structures
        BasicState,
        BasicState3D,
        conduction_basic_state,
        meridional_basic_state,
        nonaxisymmetric_basic_state,
        basic_state,

        # Self-consistent basic state (with advection)
        nonaxisymmetric_basic_state_selfconsistent,
        basic_state_selfconsistent,
        AdvectionDiffusionSolver,
        compute_phi_advection_spectral,
        compute_full_advection_spectral,
        solve_poisson_mode,

        # Meridional circulation (toroidal-poloidal decomposition)
        solve_meridional_coupled!,
        solve_meridional_simple!,
        solve_meridional_circulation_toroidal_poloidal!,
        sin_theta_coupling,
        cos_theta_coupling,
        theta_derivative_coupling,
        inv_sin_theta_gaunt,
        inv_sin_theta_coupling,

        # Symbolic spherical harmonic boundary conditions
        SphericalHarmonicBC,
        Ylm,
        Y00, Y10, Y11,
        Y20, Y21, Y22,
        Y30, Y31, Y32, Y33,
        Y40, Y41, Y42, Y43, Y44,
        to_dict,
        get_lmax, get_mmax, get_lmax_mmax,
        is_axisymmetric,

        # Thermal wind solvers
        solve_thermal_wind_balance!,
        solve_thermal_wind_balance_3d!,
        theta_derivative_coeff_3d,

        # Basic state operators
        BasicStateOperators,
        build_basic_state_operators,
        add_basic_state_operators!,

        # =================================================================
        # Onset Convection (No Mean Flow)
        # =================================================================
        OnsetConvectionParams,
        solve_onset_problem,
        find_critical_Ra_onset,
        find_global_critical_onset,
        estimate_onset_problem_size,
        onset_scaling_laws,

        # =================================================================
        # Biglobal Stability (Axisymmetric Mean Flow)
        # =================================================================
        BiglobalParams,
        create_conduction_basic_state,
        create_thermal_wind_basic_state,
        create_custom_basic_state,
        solve_biglobal_problem,
        find_critical_Ra_biglobal,
        compare_onset_vs_biglobal,
        sweep_thermal_wind_amplitude,
        analyze_basic_state,

        # =================================================================
        # Triglobal Stability (Non-Axisymmetric Mean Flow)
        # =================================================================
        TriglobalParams,
        setup_coupled_mode_problem,
        estimate_triglobal_problem_size,
        solve_triglobal_eigenvalue_problem,
        find_critical_rayleigh_triglobal,

        # =================================================================
        # Sparse operators (from Operators submodule)
        # =================================================================
        SparseOnsetParams,
        SparseStabilityOperator,
        assemble_sparse_matrices,

        # =================================================================
        # MHD extensions (from MHD submodule)
        # =================================================================
        MHDParams,
        MHDStabilityOperator,
        BackgroundField,
        no_field, axial, dipole,
        assemble_mhd_matrices,

        # =================================================================
        # v2.0 API types and functions
        # =================================================================
        OnsetProblem,
        BiglobalProblem,
        TriglobalProblem,
        MHDProblem,
        StabilityResult,
        AbstractStabilityResult,
        growth_rate,
        frequency,
        leading_mode,
        estimate_size,
        solve,
        find_critical_Ra,
        eigenspectrum,
        plot_meridional,
        plot_radial,
        perturbation_velocity,
        perturbation_temperature,
        perturbation_magnetic,

        # Validation
        validate_onset_params,
        validate_basic_state_consistency,
        validate_basic_state_3d_consistency,
        validate_biglobal_params,
        validate_triglobal_params,
        validate_mhd_params

end
