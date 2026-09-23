#!/usr/bin/env julia
#
# Onset eigenvalues of rotating spherical-shell convection over azimuthal orders.
#
# The script fixes the Rayleigh number, solves the linear onset problem for
# each azimuthal wavenumber m, and prints the leading complex growth rate
# λ = σ + iω. It then reconstructs the velocity and temperature of the fastest
# growing mode on a meridional grid. Parameters: E = 1e-5 (outer-radius based),
# Pr = 1, χ = 0.35, Ra = 2.1e7 (shell-thickness based).
#
# Requirements: sparse eigensolves use SLEPc. Install PETSc/SLEPc with complex
# scalars and add PetscWrap and SlepcWrap to the active environment; see
# "SLEPc setup" in docs/src/getting_started.md.
#
# Usage: julia --project=. example/linear_stability_demo.jl [--theta-points=<int>]

using Magrathea
using Printf
import PetscWrap, SlepcWrap

E = 1e-5
Pr = 1.0
Ra = 2.1e7
χ = 0.35
Nr = 64

# ------------------------------------------------------------------------------
# Command-line options (the MAGRATHEA_THETA_POINTS environment variable also works)
# ------------------------------------------------------------------------------

function parse_cli_args(args)
    opts = Dict{Symbol,Any}()
    for arg in args
        if arg in ("-h", "--help")
            println("""
                Usage: julia example/linear_stability_demo.jl [options]

                Options:
                  --theta-points=<int>           Colatitude points for the reconstructed mode (default 96)
                  --help                         Show this message
                """)
            exit(0)
        elseif startswith(arg, "--theta-points=")
            opts[:theta_points] = parse(Int, split(arg, '=' )[2])
        else
            @warn "Ignoring unrecognised argument" arg
        end
    end
    return opts
end

cli_opts = parse_cli_args(ARGS)
meridional_points = get(cli_opts, :theta_points, parse(Int, get(ENV, "MAGRATHEA_THETA_POINTS", "96")))

slepc_init!("-eps_gen_non_hermitian -st_type sinvert -st_pc_type lu " *
            "-st_pc_factor_mat_solver_type mumps")

try
    println("m    Re(λ₁)          Im(λ₁)")
    println("--------------------------------")

    ms = 1:20
    results = map(ms) do m
        lmax = max(48, m + 6)
        params = OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=m, lmax=lmax, Nr=Nr)
        result = solve(OnsetProblem(params); nev=2, which=:LR, tol=1e-6, maxiter=120)
        @printf("%2d  %12.5e  %12.5e\n", m, growth_rate(result), frequency(result))
        result
    end

    # Reconstruct the fastest-growing mode over all m (rank 0 holds the eigenvectors).
    i_max = argmax(growth_rate.(results))
    m_max = ms[i_max]
    best = results[i_max]
    ur, uθ, uφ, r, grid = perturbation_velocity(best, best.leading_index; Nθ=meridional_points)
    θfield, _, _ = perturbation_temperature(best, best.leading_index; Nθ=meridional_points)

    println()
    @printf("Fastest-growing mode: m = %d, σ = %.5e, ω = %.5e\n",
            m_max, growth_rate(best), frequency(best))
    @printf("Meridional grid: %d radial × %d colatitude points\n", length(r), length(grid.θ))
    @printf("  max |u_r| = %.4e, max |u_θ| = %.4e, max |u_φ| = %.4e, max |Θ| = %.4e\n",
            maximum(abs, ur), maximum(abs, uθ), maximum(abs, uφ), maximum(abs, θfield))
finally
    slepc_finalize!()
end
