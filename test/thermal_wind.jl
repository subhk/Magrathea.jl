# =============================================================================
#  Tests for Thermal Wind Balance
#
#  Tests the thermal wind solver against analytical solutions for both
#  axisymmetric (biglobal) and non-axisymmetric (triglobal) cases.
#
#  Thermal wind equation (non-dimensional, viscous time scale):
#
#    cos(θ) ∂ū_φ/∂r - sin(θ) ū_φ/r = -(Ra E²)/(2 Pr r_o) × ∂Θ̄/∂θ
#
#  Using diagonal approximation (neglecting cos/sin angular operators):
#
#    d(r·ū_L)/dr = prefactor × r² × F_L(r)
#
#  where F_L(r) is the forcing from temperature gradient projected onto mode L.
# =============================================================================

using Test
using LinearAlgebra
using Magrathea

# =============================================================================
#  Helper Functions for Analytical Solutions
# =============================================================================

"""
Spherical harmonic normalization: Y_ℓ0 = N_ℓ × P_ℓ(cos θ)
"""
Y_norm(ℓ::Int, T::Type=Float64) = sqrt(T(2ℓ + 1) / (4 * T(π)))

"""
Coupling coefficient from temperature mode ℓ to velocity mode L = ℓ+1
"""
function coupling_coeff_plus(ℓ::Int, T::Type=Float64)
    base = T(ℓ * (ℓ + 1)) / T(2ℓ + 1)
    L = ℓ + 1
    norm_ratio = Y_norm(ℓ, T) / Y_norm(L, T)
    return -base * norm_ratio  # Negative from ∂Y/∂θ formula
end

"""
Coupling coefficient from temperature mode ℓ to velocity mode L = ℓ-1
"""
function coupling_coeff_minus(ℓ::Int, T::Type=Float64)
    ℓ < 1 && return zero(T)
    base = T(ℓ * (ℓ + 1)) / T(2ℓ + 1)
    L = ℓ - 1
    norm_ratio = Y_norm(ℓ, T) / Y_norm(max(L, 0) == 0 ? 1 : L, T)
    if L == 0
        norm_ratio = Y_norm(ℓ, T) / Y_norm(0, T)
    end
    return base * norm_ratio  # Positive (double negative)
end

"""
Thermal wind prefactor: -(Ra E²)/(2 Pr r_o)
"""
thermal_wind_prefactor(Ra, E, Pr, r_o=1.0) = -(Ra * E^2) / (2 * Pr * r_o)

"""
Analytical solution for thermal wind with CONSTANT temperature coefficient.

For θ̄_ℓ(r) = A (constant), the forcing F_L = c_L × A is constant.
The ODE: d(r·ū_L)/dr = prefactor × c_L × A × r²

Integrating with no-slip BC at r_i:
    r·ū_L = prefactor × c_L × A × (r³ - r_i³)/3

Then: ū_L(r) = prefactor × c_L × A × (r³ - r_i³)/(3r)
             = prefactor × c_L × A × (r² - r_i³/r) / 3

This satisfies ū_L(r_i) = 0. The outer BC ū_L(r_o) ≠ 0 in general because
a first-order ODE can only satisfy one boundary condition.
"""
function analytical_uphi_constant_theta(r, r_i, r_o, Ra, E, Pr, ℓ_theta, A, L)
    prefactor = thermal_wind_prefactor(Ra, E, Pr, r_o)

    # Get coupling coefficient
    if L == ℓ_theta + 1
        c_L = coupling_coeff_plus(ℓ_theta)
    elseif L == ℓ_theta - 1
        c_L = coupling_coeff_minus(ℓ_theta)
    else
        return zeros(length(r))  # No coupling
    end

    # Solution satisfying BC at r_i: ū(r_i) = 0
    return @. prefactor * c_L * A * (r^2 - r_i^3 / r) / 3
end

"""
Analytical solution for thermal wind with LINEAR temperature coefficient.

For θ̄_ℓ(r) = A × r, the forcing F_L = c_L × A × r.
The ODE: d(r·ū_L)/dr = prefactor × c_L × A × r³

Integrating with no-slip BC at r_i:
    r·ū_L = prefactor × c_L × A × (r⁴ - r_i⁴)/4

Then: ū_L(r) = prefactor × c_L × A × (r⁴ - r_i⁴)/(4r)
             = prefactor × c_L × A × (r³ - r_i⁴/r) / 4

This satisfies ū_L(r_i) = 0.
"""
function analytical_uphi_linear_theta(r, r_i, r_o, Ra, E, Pr, ℓ_theta, A, L)
    prefactor = thermal_wind_prefactor(Ra, E, Pr, r_o)

    # Get coupling coefficient
    if L == ℓ_theta + 1
        c_L = coupling_coeff_plus(ℓ_theta)
    elseif L == ℓ_theta - 1
        c_L = coupling_coeff_minus(ℓ_theta)
    else
        return zeros(length(r))
    end

    # Solution satisfying BC at r_i: ū(r_i) = 0
    return @. prefactor * c_L * A * (r^3 - r_i^4 / r) / 4
end

"""
Analytical solution for thermal wind with r² temperature coefficient.

For θ̄_ℓ(r) = A × r², the forcing F_L = c_L × A × r².
The ODE: d(r·ū_L)/dr = prefactor × c_L × A × r⁴

Integrating with BC at r_i:
    r·ū_L = prefactor × c_L × A × (r⁵ - r_i⁵)/5

Then: ū_L(r) = prefactor × c_L × A × (r⁴ - r_i⁵/r) / 5

This satisfies ū_L(r_i) = 0.
"""
function analytical_uphi_quadratic_theta(r, r_i, r_o, Ra, E, Pr, ℓ_theta, A, L)
    prefactor = thermal_wind_prefactor(Ra, E, Pr, r_o)

    if L == ℓ_theta + 1
        c_L = coupling_coeff_plus(ℓ_theta)
    elseif L == ℓ_theta - 1
        c_L = coupling_coeff_minus(ℓ_theta)
    else
        return zeros(length(r))
    end

    # Solution satisfying BC at r_i: ū(r_i) = 0
    return @. prefactor * c_L * A * (r^4 - r_i^5 / r) / 5
end

"""
Coupling coefficient from non-axisymmetric temperature mode `(ℓ, m)` to
velocity mode `L = ℓ + 1` in the diagonal triglobal thermal-wind approximation.
"""
function coupling_coeff_plus_3d(ℓ::Int, m::Int, T::Type=Float64)
    numer = T((ℓ + 1)^2 - m^2)
    denom = T((2ℓ + 1) * (2ℓ + 3))
    norm_ratio = sqrt(T(2ℓ + 1) / T(2 * (ℓ + 1) + 1))
    return -T(ℓ + 1) * sqrt(numer / denom) * norm_ratio
end

"""
Analytical diagonal triglobal thermal-wind solution for constant `θ̄_ℓm(r) = A`
and `L = ℓ + 1`, satisfying the inner no-slip boundary condition.
"""
function analytical_uphi_constant_theta_3d(r, r_i, r_o, Ra, E, Pr, ℓ_theta, m, A, L)
    L == ℓ_theta + 1 || return zeros(length(r))
    prefactor = thermal_wind_prefactor(Ra, E, Pr, r_o)
    c_L = coupling_coeff_plus_3d(ℓ_theta, m)
    return @. prefactor * c_L * A * (r^2 - r_i^3 / r) / 3
end


# =============================================================================
#  Test Suite
# =============================================================================

@testset "Thermal Wind Balance" begin

    # Common parameters
    χ = 0.35
    r_i = χ
    r_o = 1.0
    Nr = 64
    E = 1e-4
    Ra = 1e6
    Pr = 1.0

    # Create Chebyshev grid
    cd = Magrathea.ChebyshevDiffn(Nr, [r_i, r_o], 4)
    r = cd.x

    @testset "Axisymmetric (Biglobal) - Constant θ̄₂₀" begin
        # Test with constant temperature coefficient θ̄₂₀(r) = A
        A = 0.1
        ℓ_theta = 2
        lmax_bs = 4

        # Initialize coefficient dictionaries
        theta_coeffs = Dict{Int, Vector{Float64}}()
        dtheta_dr_coeffs = Dict{Int, Vector{Float64}}()
        uphi_coeffs = Dict{Int, Vector{Float64}}()
        duphi_dr_coeffs = Dict{Int, Vector{Float64}}()

        # Set constant temperature for ℓ=2
        for ℓ in 0:lmax_bs
            if ℓ == ℓ_theta
                theta_coeffs[ℓ] = fill(A, Nr)
                dtheta_dr_coeffs[ℓ] = zeros(Nr)  # Derivative of constant is zero
            else
                theta_coeffs[ℓ] = zeros(Nr)
                dtheta_dr_coeffs[ℓ] = zeros(Nr)
            end
            uphi_coeffs[ℓ] = zeros(Nr)
            duphi_dr_coeffs[ℓ] = zeros(Nr)
        end

        # Solve thermal wind
        Magrathea.solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                    cd, r_i, r_o, Ra, Pr;
                                    mechanical_bc=:no_slip, E=E)

        # Check L=1 mode (from ℓ=2 temperature)
        L = 1
        uphi_analytical_L1 = analytical_uphi_constant_theta(r, r_i, r_o, Ra, E, Pr, ℓ_theta, A, L)

        if haskey(uphi_coeffs, L) && maximum(abs.(uphi_coeffs[L])) > 1e-14
            # Compare in interior (exclude boundary points which have enforced BCs)
            interior = 3:(Nr-2)
            rel_error_L1 = norm(uphi_coeffs[L][interior] - uphi_analytical_L1[interior]) /
                           (norm(uphi_analytical_L1[interior]) + 1e-14)

            @test rel_error_L1 < 0.05  # 5% relative error tolerance
        end

        # Check L=3 mode (from ℓ=2 temperature)
        L = 3
        uphi_analytical_L3 = analytical_uphi_constant_theta(r, r_i, r_o, Ra, E, Pr, ℓ_theta, A, L)

        if haskey(uphi_coeffs, L) && maximum(abs.(uphi_coeffs[L])) > 1e-14
            interior = 3:(Nr-2)
            rel_error_L3 = norm(uphi_coeffs[L][interior] - uphi_analytical_L3[interior]) /
                           (norm(uphi_analytical_L3[interior]) + 1e-14)

            @test rel_error_L3 < 0.05
        end

        # Check inner boundary condition: ū_φ(r_i) = 0
        # NOTE: The first-order thermal wind ODE can only satisfy ONE BC.
        # We enforce the inner BC; the outer BC will have a small non-zero value.
        for L in keys(uphi_coeffs)
            if maximum(abs.(uphi_coeffs[L])) > 1e-14
                @test abs(uphi_coeffs[L][1]) < 1e-10  # Inner BC
            end
        end
    end

    @testset "Axisymmetric (Biglobal) - Linear θ̄₂₀" begin
        # Test with linear temperature coefficient θ̄₂₀(r) = A × r
        A = 0.1
        ℓ_theta = 2
        lmax_bs = 4

        theta_coeffs = Dict{Int, Vector{Float64}}()
        dtheta_dr_coeffs = Dict{Int, Vector{Float64}}()
        uphi_coeffs = Dict{Int, Vector{Float64}}()
        duphi_dr_coeffs = Dict{Int, Vector{Float64}}()

        for ℓ in 0:lmax_bs
            if ℓ == ℓ_theta
                theta_coeffs[ℓ] = A .* r
                dtheta_dr_coeffs[ℓ] = fill(A, Nr)
            else
                theta_coeffs[ℓ] = zeros(Nr)
                dtheta_dr_coeffs[ℓ] = zeros(Nr)
            end
            uphi_coeffs[ℓ] = zeros(Nr)
            duphi_dr_coeffs[ℓ] = zeros(Nr)
        end

        Magrathea.solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                    cd, r_i, r_o, Ra, Pr;
                                    mechanical_bc=:no_slip, E=E)

        # Check L=1 mode
        L = 1
        uphi_analytical = analytical_uphi_linear_theta(r, r_i, r_o, Ra, E, Pr, ℓ_theta, A, L)

        if haskey(uphi_coeffs, L) && maximum(abs.(uphi_coeffs[L])) > 1e-14
            interior = 3:(Nr-2)
            rel_error = norm(uphi_coeffs[L][interior] - uphi_analytical[interior]) /
                        (norm(uphi_analytical[interior]) + 1e-14)

            @test rel_error < 0.05
        end

        # Inner BC only (first-order ODE can only satisfy one BC)
        for L in keys(uphi_coeffs)
            if maximum(abs.(uphi_coeffs[L])) > 1e-14
                @test abs(uphi_coeffs[L][1]) < 1e-10
            end
        end
    end

    @testset "Axisymmetric (Biglobal) - Quadratic θ̄₂₀" begin
        # Test with quadratic temperature coefficient θ̄₂₀(r) = A × r²
        A = 0.1
        ℓ_theta = 2
        lmax_bs = 4

        theta_coeffs = Dict{Int, Vector{Float64}}()
        dtheta_dr_coeffs = Dict{Int, Vector{Float64}}()
        uphi_coeffs = Dict{Int, Vector{Float64}}()
        duphi_dr_coeffs = Dict{Int, Vector{Float64}}()

        for ℓ in 0:lmax_bs
            if ℓ == ℓ_theta
                theta_coeffs[ℓ] = A .* r.^2
                dtheta_dr_coeffs[ℓ] = 2 * A .* r
            else
                theta_coeffs[ℓ] = zeros(Nr)
                dtheta_dr_coeffs[ℓ] = zeros(Nr)
            end
            uphi_coeffs[ℓ] = zeros(Nr)
            duphi_dr_coeffs[ℓ] = zeros(Nr)
        end

        Magrathea.solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                    cd, r_i, r_o, Ra, Pr;
                                    mechanical_bc=:no_slip, E=E)

        L = 1
        uphi_analytical = analytical_uphi_quadratic_theta(r, r_i, r_o, Ra, E, Pr, ℓ_theta, A, L)

        if haskey(uphi_coeffs, L) && maximum(abs.(uphi_coeffs[L])) > 1e-14
            interior = 3:(Nr-2)
            rel_error = norm(uphi_coeffs[L][interior] - uphi_analytical[interior]) /
                        (norm(uphi_analytical[interior]) + 1e-14)

            @test rel_error < 0.05
        end
    end

    @testset "Prefactor scaling" begin
        # Test that thermal wind amplitude scales correctly with Ra, E², and Pr

        A = 0.1
        ℓ_theta = 2
        lmax_bs = 4

        function compute_max_uphi(Ra_test, E_test, Pr_test)
            theta_coeffs = Dict{Int, Vector{Float64}}()
            dtheta_dr_coeffs = Dict{Int, Vector{Float64}}()
            uphi_coeffs = Dict{Int, Vector{Float64}}()
            duphi_dr_coeffs = Dict{Int, Vector{Float64}}()

            for ℓ in 0:lmax_bs
                theta_coeffs[ℓ] = ℓ == ℓ_theta ? fill(A, Nr) : zeros(Nr)
                dtheta_dr_coeffs[ℓ] = zeros(Nr)
                uphi_coeffs[ℓ] = zeros(Nr)
                duphi_dr_coeffs[ℓ] = zeros(Nr)
            end

            Magrathea.solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                        cd, r_i, r_o, Ra_test, Pr_test;
                                        mechanical_bc=:no_slip, E=E_test)

            max_uphi = 0.0
            for (L, uphi) in uphi_coeffs
                max_uphi = max(max_uphi, maximum(abs.(uphi)))
            end
            return max_uphi
        end

        # Reference case
        uphi_ref = compute_max_uphi(Ra, E, Pr)

        # Double Ra → double uphi
        uphi_2Ra = compute_max_uphi(2*Ra, E, Pr)
        @test abs(uphi_2Ra / uphi_ref - 2.0) < 0.01

        # Double E → quadruple uphi (E² scaling)
        uphi_2E = compute_max_uphi(Ra, 2*E, Pr)
        @test abs(uphi_2E / uphi_ref - 4.0) < 0.01

        # Double Pr → halve uphi
        uphi_2Pr = compute_max_uphi(Ra, E, 2*Pr)
        @test abs(uphi_2Pr / uphi_ref - 0.5) < 0.01
    end

    @testset "Mode coupling structure" begin
        # Test that correct velocity modes are excited by each temperature mode

        lmax_bs = 6

        # Test ℓ=2 temperature → L=1,3 velocity
        theta_coeffs = Dict(ℓ => (ℓ == 2 ? fill(0.1, Nr) : zeros(Nr)) for ℓ in 0:lmax_bs)
        dtheta_dr_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
        uphi_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
        duphi_dr_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)

        Magrathea.solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                    cd, r_i, r_o, Ra, Pr; E=E)

        # L=1 and L=3 should be non-zero
        @test maximum(abs.(uphi_coeffs[1])) > 1e-14
        @test maximum(abs.(uphi_coeffs[3])) > 1e-14

        # Other modes should be zero
        @test maximum(abs.(uphi_coeffs[0])) < 1e-14
        @test maximum(abs.(uphi_coeffs[2])) < 1e-14
        @test maximum(abs.(uphi_coeffs[4])) < 1e-14

        # Test ℓ=4 temperature → L=3,5 velocity
        theta_coeffs = Dict(ℓ => (ℓ == 4 ? fill(0.1, Nr) : zeros(Nr)) for ℓ in 0:lmax_bs)
        uphi_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
        duphi_dr_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)

        Magrathea.solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                    cd, r_i, r_o, Ra, Pr; E=E)

        @test maximum(abs.(uphi_coeffs[3])) > 1e-14
        @test maximum(abs.(uphi_coeffs[5])) > 1e-14
        @test maximum(abs.(uphi_coeffs[4])) < 1e-14
    end

    @testset "Zero forcing → zero flow" begin
        # Pure conduction (ℓ=0 only) should produce no thermal wind

        lmax_bs = 4
        theta_coeffs = Dict{Int, Vector{Float64}}()
        dtheta_dr_coeffs = Dict{Int, Vector{Float64}}()
        uphi_coeffs = Dict{Int, Vector{Float64}}()
        duphi_dr_coeffs = Dict{Int, Vector{Float64}}()

        for ℓ in 0:lmax_bs
            # Only ℓ=0 is non-zero (uniform temperature gradient)
            theta_coeffs[ℓ] = ℓ == 0 ? ones(Nr) : zeros(Nr)
            dtheta_dr_coeffs[ℓ] = zeros(Nr)
            uphi_coeffs[ℓ] = zeros(Nr)
            duphi_dr_coeffs[ℓ] = zeros(Nr)
        end

        Magrathea.solve_thermal_wind_balance!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                    cd, r_i, r_o, Ra, Pr; E=E)

        # All velocity modes should be zero (∂Y_00/∂θ = 0)
        for (L, uphi) in uphi_coeffs
            @test maximum(abs.(uphi)) < 1e-14
        end
    end

    @testset "Conduction basic state API" begin
        # Test that conduction_basic_state produces zero zonal flow

        bs = Magrathea.conduction_basic_state(cd, χ, 6)

        for (ℓ, uphi) in bs.uphi_coeffs
            @test maximum(abs.(uphi)) < 1e-14
        end
    end

    @testset "Meridional basic state API" begin
        # Test the meridional_basic_state function

        amplitude = 0.1
        bs = Magrathea.meridional_basic_state(cd, χ, E, Ra, Pr, 6, amplitude;
                                    mechanical_bc=:no_slip)

        # Should have non-zero ℓ=2 temperature
        @test haskey(bs.theta_coeffs, 2)
        @test maximum(abs.(bs.theta_coeffs[2])) > 1e-14

        # Coupled thermal wind: a Y_ℓ0 (equatorially symmetric) temperature drives an
        # equatorially symmetric zonal flow ⇒ EVEN L (2,4,…); odd L stays ~0. (The old
        # diagonal heuristic wrongly placed the flow at odd L=1,3.)
        @test haskey(bs.uphi_coeffs, 2)
        @test maximum(abs.(bs.uphi_coeffs[2])) > 1e-14
        # odd-L flow is negligible vs even-L — correct parity for Y_ℓ0 forcing
        @test maximum(abs.(get(bs.uphi_coeffs, 1, zeros(Nr)))) <
              1e-6 * maximum(abs.(bs.uphi_coeffs[2]))

        # Check inner BC (first-order ODE can only satisfy one BC)
        for (L, uphi) in bs.uphi_coeffs
            if maximum(abs.(uphi)) > 1e-14
                @test abs(uphi[1]) < 1e-10
            end
        end
    end

    @testset "Coupling coefficient consistency" begin
        # Verify coupling coefficients match expected values

        # For ℓ=2: base_coeff = 2×3/5 = 6/5 = 1.2
        @test abs(6/5 - 1.2) < 1e-10

        # Y_norm ratios
        N2 = Y_norm(2)
        N1 = Y_norm(1)
        N3 = Y_norm(3)

        # Expected: N_ℓ = √((2ℓ+1)/(4π))
        @test abs(N2 - sqrt(5/(4π))) < 1e-10
        @test abs(N1 - sqrt(3/(4π))) < 1e-10
        @test abs(N3 - sqrt(7/(4π))) < 1e-10

        # c_plus for ℓ=2 → L=3: -6/5 × N2/N3
        c_plus_2 = coupling_coeff_plus(2)
        expected_c_plus = -1.2 * sqrt(5/7)
        @test abs(c_plus_2 - expected_c_plus) < 1e-10

        # c_minus for ℓ=2 → L=1: +6/5 × N2/N1
        c_minus_2 = coupling_coeff_minus(2)
        expected_c_minus = 1.2 * sqrt(5/3)
        @test abs(c_minus_2 - expected_c_minus) < 1e-10
    end

end  # @testset "Thermal Wind Balance"


# =============================================================================
#  Non-Axisymmetric (Triglobal) Tests
# =============================================================================

@testset "Thermal Wind Balance - Non-Axisymmetric (Triglobal)" begin

    χ = 0.35
    r_i = χ
    r_o = 1.0
    Nr = 64
    E = 1e-4
    Ra = 1e6
    Pr = 1.0

    cd = Magrathea.ChebyshevDiffn(Nr, [r_i, r_o], 4)
    r = cd.x

    @testset "m=0 reduces to axisymmetric" begin
        # For m_bs=0, the 3D solver should give same result as axisymmetric

        # Import the 3D solver
        solve_tw_3d! = Magrathea.solve_thermal_wind_balance_3d!

        A = 0.1
        ℓ_theta = 2
        lmax_bs = 4

        # Axisymmetric solve
        theta_axi = Dict(ℓ => (ℓ == ℓ_theta ? fill(A, Nr) : zeros(Nr)) for ℓ in 0:lmax_bs)
        dtheta_axi = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
        uphi_axi = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
        duphi_axi = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)

        Magrathea.solve_thermal_wind_balance!(uphi_axi, duphi_axi, theta_axi,
                                    cd, r_i, r_o, Ra, Pr; E=E)

        # 3D solve with m_bs=0
        theta_3d = Dict(ℓ => (ℓ == ℓ_theta ? fill(A, Nr) : zeros(Nr)) for ℓ in 0:lmax_bs)
        uphi_3d = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
        duphi_3d = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)

        solve_tw_3d!(uphi_3d, duphi_3d, theta_3d, 0,
                     cd, r_i, r_o, Ra, Pr; E=E)

        # Results should match
        for L in 0:lmax_bs
            @test norm(uphi_3d[L] - uphi_axi[L]) < 1e-12 * (norm(uphi_axi[L]) + 1)
        end
    end

    @testset "Non-axisymmetric diagonal solver matches analytical coefficient (m_bs=2)" begin
        solve_tw_3d! = Magrathea.solve_thermal_wind_balance_3d!

        A = 0.1
        m_bs = 2
        lmax_bs = 6

        # Temperature at (ℓ=2, m=2)
        theta_coeffs = Dict(ℓ => (ℓ == 2 ? fill(A, Nr) : zeros(Nr)) for ℓ in m_bs:lmax_bs)
        uphi_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
        duphi_dr_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)

        solve_tw_3d!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs, m_bs,
                     cd, r_i, r_o, Ra, Pr; E=E)

        expected_L3 = analytical_uphi_constant_theta_3d(r, r_i, r_o, Ra, E, Pr,
                                                        2, m_bs, A, 3)
        interior = 3:(Nr-2)
        rel_error = norm(uphi_coeffs[3][interior] - expected_L3[interior]) /
                    (norm(expected_L3[interior]) + 1e-14)

        @test maximum(abs.(uphi_coeffs[3])) > 1e-14
        @test rel_error < 0.05
        @test maximum(abs.(get(uphi_coeffs, 1, zeros(Nr)))) < 1e-14
        @test abs(uphi_coeffs[3][1]) < 1e-10
    end

    @testset "Coupled triglobal solver enforces only the inner radial BC" begin
        A = 0.1
        m_bs = 2
        lmax_bs = 6

        theta_coeffs = Dict(ℓ => (ℓ == 2 ? fill(A, Nr) : zeros(Nr)) for ℓ in m_bs:lmax_bs)
        uphi_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:(lmax_bs + 1))
        duphi_dr_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:(lmax_bs + 1))

        Magrathea.solve_thermal_wind_coupled!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs,
                                          m_bs, cd, r_i, r_o, Ra, Pr;
                                          mechanical_bc=:no_slip, E=E,
                                          lmax=lmax_bs + 1)

        idx_inner = abs(r[1] - r_i) < abs(r[end] - r_i) ? 1 : length(r)
        idx_outer = idx_inner == 1 ? length(r) : 1
        outer_amplitude = maximum(abs(uphi[idx_outer]) for uphi in values(uphi_coeffs))

        @test all(abs(uphi[idx_inner]) < 1e-10 for uphi in values(uphi_coeffs))
        @test outer_amplitude > 1e-14
    end

    @testset "Singular stress-free m_bs=1 system returns the minimum-norm solution" begin
        # For stress-free + m_bs = 1 the geostrophic mode ū_φ ∝ r·Y_1^1 (= r sinθ)
        # satisfies the interior equations AND the Robin BC, so the coupled system
        # has an exact null vector. The solver must return the minimum-norm
        # solution: finite, equations satisfied, no arbitrary null component.
        m_bs = 1
        theta_coeffs = Dict(2 => fill(0.1, Nr))
        uphi_coeffs = Dict{Int,Vector{Float64}}()
        duphi_dr_coeffs = Dict{Int,Vector{Float64}}()

        @test_logs (:warn, r"minimum-norm") Magrathea.solve_thermal_wind_coupled!(
            uphi_coeffs, duphi_dr_coeffs, theta_coeffs, m_bs, cd, r_i, r_o, Ra, Pr;
            mechanical_bc=:stress_free, E=E)

        @test all(all(isfinite, v) for v in values(uphi_coeffs))

        # null component: projection of U_1 onto the exact null direction r
        nv = r ./ norm(r)
        scale = maximum(maximum(abs, v) for v in values(uphi_coeffs))
        @test abs(LinearAlgebra.dot(uphi_coeffs[1], nv)) < 1e-10 * scale

        # solution still satisfies the coupled equations: spot-check the K=1
        # Galerkin row Σ_L [A_1L dŪ_L/dr − (1/r)B_1L Ū_L] = F_1 at interior nodes
        # (only L=2 contributes: A_{1,2}=α_2⁻, B_{1,2}=−3α_2⁻)
        αm(L) = sqrt((L^2 - m_bs^2) / ((2L - 1) * (2L + 1)))
        Mp = Magrathea._dtheta_sphere_projection([1], [2], m_bs, Float64)
        pref = -(Ra * E^2) / (2 * Pr * r_o)
        dU2 = cd.D1 * uphi_coeffs[2]
        res = 0.0; rhsmax = 0.0
        for i in 5:(Nr - 5)
            lhs = αm(2) * dU2[i] + 3 * αm(2) * uphi_coeffs[2][i] / r[i]
            rhs = pref * Mp[(1, 2)] * theta_coeffs[2][i]
            res = max(res, abs(lhs - rhs)); rhsmax = max(rhsmax, abs(rhs))
        end
        @test res / rhsmax < 1e-8
    end

    @testset "Amplitude scaling for 3D" begin
        # Thermal wind should still scale as Ra × E² / Pr for non-axisymmetric

        solve_tw_3d! = Magrathea.solve_thermal_wind_balance_3d!

        A = 0.1
        m_bs = 0  # Use m=0 for simplicity
        lmax_bs = 4

        function compute_max_uphi_3d(Ra_test, E_test, Pr_test)
            theta_coeffs = Dict(ℓ => (ℓ == 2 ? fill(A, Nr) : zeros(Nr)) for ℓ in 0:lmax_bs)
            uphi_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)
            duphi_dr_coeffs = Dict(ℓ => zeros(Nr) for ℓ in 0:lmax_bs)

            solve_tw_3d!(uphi_coeffs, duphi_dr_coeffs, theta_coeffs, m_bs,
                         cd, r_i, r_o, Ra_test, Pr_test; E=E_test)

            return maximum(maximum(abs.(v)) for v in values(uphi_coeffs))
        end

        uphi_ref = compute_max_uphi_3d(Ra, E, Pr)

        # E² scaling
        uphi_2E = compute_max_uphi_3d(Ra, 2*E, Pr)
        @test abs(uphi_2E / uphi_ref - 4.0) < 0.01
    end

end  # @testset "Triglobal"


# =============================================================================
#  Integration Tests with Full BasicState3D
# =============================================================================

@testset "Full BasicState3D Integration" begin

    χ = 0.35
    Nr = 48
    E = 1e-4
    Ra = 1e6
    Pr = 1.0
    lmax_bs = 4
    mmax_bs = 2

    cd = Magrathea.ChebyshevDiffn(Nr, [χ, 1.0], 4)

    @testset "nonaxisymmetric_basic_state produces valid flow" begin
        # Create a 3D basic state with mixed modes
        amplitudes = Dict(
            (2, 0) => 0.1,   # Axisymmetric Y₂₀
            (2, 2) => 0.05,  # Non-axisymmetric Y₂₂
        )

        bs3d = Magrathea.nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr,
                                                  lmax_bs, mmax_bs, amplitudes)

        # Check that velocity was computed
        has_uphi = false
        for ((ℓ, m), uphi) in bs3d.uphi_coeffs
            if maximum(abs.(uphi)) > 1e-14
                has_uphi = true

                # Check inner BC (first-order ODE can only satisfy one BC)
                @test abs(uphi[1]) < 1e-10
            end
        end

        # At least some velocity should exist (from Y₂₀)
        @test has_uphi
    end

    @testset "nonaxisymmetric_basic_state computes meridional flow for longitudinal forcing" begin
        amplitudes = Dict((2, 2) => 0.05)

        bs3d = Magrathea.nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr,
                                                  lmax_bs, mmax_bs, amplitudes)

        meridional_norm = zero(Float64)
        for ((_, m), ur) in bs3d.ur_coeffs
            if m != 0
                meridional_norm += norm(ur)
            end
        end
        for ((_, m), uθ) in bs3d.utheta_coeffs
            if m != 0
                meridional_norm += norm(uθ)
            end
        end

        @test meridional_norm > 1e-14
    end

    @testset "nonaxisymmetric_basic_state defaults to coupled thermal wind" begin
        amplitudes = Dict((2, 2) => 0.05)

        bs_default = Magrathea.nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr,
                                                       lmax_bs, mmax_bs, amplitudes)
        bs_coupled = Magrathea.nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr,
                                                       lmax_bs, mmax_bs, amplitudes;
                                                       coupled_thermal_wind=true)

        for key in keys(bs_coupled.uphi_coeffs)
            @test get(bs_default.uphi_coeffs, key, zeros(Nr)) ≈ bs_coupled.uphi_coeffs[key]
        end
    end

    @testset "Pure axisymmetric 3D basic state" begin
        # Only Y₂₀ mode
        amplitudes = Dict((2, 0) => 0.1)

        bs3d = Magrathea.nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr,
                                                  lmax_bs, mmax_bs, amplitudes)

        # Should match the axisymmetric result
        bs_axi = Magrathea.meridional_basic_state(cd, χ, E, Ra, Pr, lmax_bs, 0.1)

        # Compare L=2 velocity (dominant even mode; both paths use the coupled m=0 solve)
        if haskey(bs3d.uphi_coeffs, (2, 0)) && haskey(bs_axi.uphi_coeffs, 2)
            uphi_3d_L2 = bs3d.uphi_coeffs[(2, 0)]
            uphi_axi_L2 = bs_axi.uphi_coeffs[2]

            rel_diff = norm(uphi_3d_L2 - uphi_axi_L2) / (norm(uphi_axi_L2) + 1e-14)
            @test rel_diff < 0.1  # 10% tolerance for different code paths
        end
    end

end  # @testset "Full BasicState3D"


# =============================================================================
#  Manufactured-Solution Validation of the Coupled Solver
#
#  The coupled solver discretizes the full thermal-wind PDE
#     [cosθ ∂_r - (sinθ/r) ∂_θ] u_φ = -(Ra E²)/(2 Pr r_o) ∂_θ Θ
#  with the orthonormal cosθ / sinθ∂θ operators (A, B) and a forcing that is
#  the full sphere projection ⟨∂_θ Y_ℓ, Y_K⟩ (modes K = ℓ±1, ℓ±3, …).
#
#  These tests establish ground truth independently: (1) the projection
#  coefficients against an independent quadrature, and (2) the physical-space
#  PDE residual through the public solver, including convergence under lmax.
# =============================================================================

@testset "Coupled thermal wind - manufactured solution" begin
    # Orthonormal associated Legendre P̄_ℓ^m (θ-part of sphere-orthonormal Y_ℓ^m)
    function plmbar(lmax, m, x)
        P = zeros(lmax + 1); somx2 = sqrt((1 - x) * (1 + x)); pmm = 1.0; fact = 1.0
        for _ in 1:m
            pmm *= -fact * somx2; fact += 2.0
        end
        Nlm(l, mm) = Float64(sqrt((2l + 1) / 2 * factorial(big(l - mm)) / factorial(big(l + mm))))
        m <= lmax && (P[m + 1] = Nlm(m, m) * pmm)
        (m + 1 <= lmax) && (P[m + 2] = Nlm(m + 1, m) * x * (2m + 1) * pmm)
        pl2 = pmm; pl1 = x * (2m + 1) * pmm
        for l in (m + 2):lmax
            pl = (x * (2l - 1) * pl1 - (l + m - 1) * pl2) / (l - m)
            P[l + 1] = Nlm(l, m) * pl; pl2 = pl1; pl1 = pl
        end
        P
    end
    Ybar(l, m, θ) = plmbar(l, m, cos(θ))[l + 1]
    dθY(l, m, θ) = (Ybar(l, m, θ + 1e-6) - Ybar(l, m, θ - 1e-6)) / 2e-6
    # independent reference projection ⟨∂θ Y_ℓ, Y_K⟩ over the sphere
    function proj_ref(ℓ, K, m)
        θs = range(1e-5, π - 1e-5, length=40000); h = step(θs); s = 0.0
        for θ in θs
            s += dθY(ℓ, m, θ) * Ybar(K, m, θ) * sin(θ) * h
        end
        s
    end

    @testset "Projection coefficients match independent quadrature" begin
        for (ℓ, m) in [(2, 1), (3, 1), (4, 2), (5, 3)]
            Kset = collect(m:(ℓ + 3))
            M = Magrathea._dtheta_sphere_projection(Kset, [ℓ], m, Float64)
            for K in Kset
                got = get(M, (K, ℓ), 0.0)
                ref = proj_ref(ℓ, K, m)
                @test isapprox(got, ref; atol=1e-3)
            end
        end
    end

    @testset "PDE residual is small and converges under lmax" begin
        χ = 0.35; r_i = χ; r_o = 1.0; Nr = 48; E = 1e-4; Ra = 1e6; Pr = 1.0
        cd = Magrathea.ChebyshevDiffn(Nr, [r_i, r_o], 4); r = cd.x; D1 = cd.D1
        m_bs = 2
        theta = Dict(2 => 0.1 .* r .^ 2, 4 => 0.05 .* r)
        prefactor = -(Ra * E^2) / (2 * Pr * r_o)

        function residual(lmax_vel)
            uc = Dict(ℓ => zeros(Nr) for ℓ in 0:(lmax_vel + 2))
            dc = Dict(ℓ => zeros(Nr) for ℓ in 0:(lmax_vel + 2))
            Magrathea.solve_thermal_wind_coupled!(uc, dc, theta, m_bs, cd, r_i, r_o, Ra, Pr;
                                              E=E, lmax=lmax_vel)
            active = [(L, U, D1 * U) for (L, U) in uc if L >= m_bs && maximum(abs.(U)) > 0]
            uphi(i, θ)  = sum(U[i] * Ybar(L, m_bs, θ) for (L, U, _) in active; init=0.0)
            duphi(i, θ) = sum(dU[i] * Ybar(L, m_bs, θ) for (L, _, dU) in active; init=0.0)
            dTdθ(i, θ)  = sum(θc[i] * dθY(ℓ, m_bs, θ) for (ℓ, θc) in theta)
            maxres = 0.0; maxrhs = 0.0
            for i in 10:(Nr - 10), θ in range(0.4, 2.7, length=9)
                lhs = cos(θ) * duphi(i, θ) -
                      (sin(θ) / r[i]) * ((uphi(i, θ + 1e-6) - uphi(i, θ - 1e-6)) / 2e-6)
                rhs = prefactor * dTdθ(i, θ)
                maxres = max(maxres, abs(lhs - rhs)); maxrhs = max(maxrhs, abs(rhs))
            end
            maxres / maxrhs
        end

        res_lo = residual(7)
        res_hi = residual(15)
        @test res_hi < res_lo          # consistency: refining the truncation helps
        @test res_hi < 0.05            # small absolute residual at moderate truncation
    end
end
