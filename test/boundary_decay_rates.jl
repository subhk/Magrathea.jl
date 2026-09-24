module BoundaryDecayRates
# Boundary conditions checked against exact decay rates. With m = 0 each field's
# diagonal block is a pure diffusion problem whose modes are spherical Bessel
# functions, so its decay rates follow from the wall conditions alone:
#   temperature   ∂θ/∂t = κ D_ℓ θ,        magnetic   ∂b/∂t = Em D_ℓ b,
#   toroidal flow ∂T/∂t = E D_ℓ T,         poloidal   ∂(D_ℓ P)/∂t = E D_ℓ² P,
# with D_ℓ = ∂²/∂r² + (2/r)∂/∂r − ℓ(ℓ+1)/r², so every rate is −ν k².
using Test, Magrathea, LinearAlgebra, SparseArrays, Logging, SpecialFunctions
const M = Magrathea
quiet(f) = with_logger(f, NullLogger())

# f, f', f'' of z_ℓ(k r), for z = j or y spherical Bessel functions.
function bessel_jet(z, l, k, r)
    x = k * r
    f = z(l, x)
    d = k * (l / x * f - z(l + 1, x))
    return f, d, -2d / r + (l * (l + 1) / r^2 - k^2) * f
end
# f, f', f'' of r^p.
power_jet(p, r) = (r^p, p * r^(p - 1), p * (p - 1) * r^(p - 2))

# One boundary functional applied to a jet (f, f', f'') at radius r.
function functional(kind, l, r, (f, d, dd))
    kind === :value && return f
    kind === :slope && return d
    kind === :curvature && return dd
    kind === :stress && return f - r * d            # T − rT' (stress-free toroidal)
    kind === :outer_potential && return (l + 1) * f + r * d
    kind === :inner_potential && return l * f - r * d
    kind === :conductor_tangential && return f + r * d   # (r g)' = 0
    error("unknown functional $kind")
end

"""Smallest positive k whose general solution satisfies the wall functionals.
`walls` lists (functional, radius) pairs; the poloidal flow adds r^ℓ and r^-(ℓ+1)."""
function exact_roots(l, walls, ri; poloidal=false, count=4, kmax=80.0)
    function det_at(k)
        jets = Any[(r -> bessel_jet(sphericalbesselj, l, k, r)),
                   (r -> bessel_jet(sphericalbessely, l, k, r))]
        poloidal && append!(jets, [r -> power_jet(l, r), r -> power_jet(-l - 1, r)])
        det([functional(kind, l, r, jet(r)) for (kind, r) in walls, jet in jets])
    end
    roots = Float64[]
    ks = range(0.05, kmax; length=40_000)
    for (a, b) in zip(ks[1:end-1], ks[2:end])
        fa, fb = det_at(a), det_at(b)
        sign(fa) == sign(fb) && continue
        for _ in 1:80
            c = (a + b) / 2
            sign(det_at(c)) == sign(fa) ? (a = c; fa = det_at(c)) : (b = c)
        end
        push!(roots, (a + b) / 2)
        length(roots) == count && break
    end
    return roots
end

# Least-damped decay rates of a block pencil, as k = √(−λ/ν).
function block_k(A, B, ν; count=4)
    λ = eigvals(Matrix(A), Matrix(B))
    λ = λ[isfinite.(λ) .& (abs.(λ) .< 1e8)]
    λ = sort(λ, by=z -> -real(z))
    return sqrt.(abs.(real.(λ[1:count])) ./ ν)
end
# Same for a tau block: eliminate its boundary rows first.
function tau_block_k(A, B, ν; count=4)
    A = Matrix(A); B = Matrix(B)
    bc = findall(i -> iszero(B[i, :]), axes(B, 1)); interior = setdiff(axes(B, 1), bc)
    R = nullspace(A[bc, :] ./ maximum(abs, A[bc, :]; dims=2))
    return block_k(A[interior, :] * R, B[interior, :] * R, ν; count=count)
end
relerr(a, b) = maximum(abs.(a .- b) ./ b)

const THERMAL = Dict(:fixed_temperature => :value, :fixed_flux => :slope)
flow_walls(bc) = bc === :no_slip ? (:value, :slope) : (:value, :curvature)
tor_wall(bc) = bc === :no_slip ? :value : :stress

@testset "Hydro collocation: thermal, toroidal and poloidal decay rates" begin
    for ri in (0.35, 0.6), inner in (:fixed_temperature, :fixed_flux),
        outer in (:fixed_temperature, :fixed_flux), mech in (:no_slip, :stress_free)
        p = OnsetParams(E=1e-2, Pr=0.5, Ra=0.0, χ=ri, m=0, lmax=3, Nr=40,
                        mechanical_bc=mech, thermal_bc=(inner, outer))
        op = LinearStabilityOperator(p)
        A, B, idofs, bdofs = assemble_matrices(op)
        Ar, Br, red = M._constrained_reduced_matrices(A, B, op, idofs, bdofs)
        rowpos = Dict(i => k for (k, i) in enumerate(idofs))
        for blk in red.blocks
            ℓ, field = only(k for (k, v) in op.index_map if v == blk.full_indices)
            rows = [rowpos[i] for i in blk.full_indices if haskey(rowpos, i)]
            Ab = Ar[rows, blk.reduced_indices]; Bb = Br[rows, blk.reduced_indices]
            if field === :Θ
                exact = exact_roots(ℓ, [(THERMAL[inner], ri), (THERMAL[outer], 1.0)], ri)
                @test relerr(block_k(Ab, Bb, p.E / p.Pr), exact) < 1e-9
            elseif field === :T
                exact = exact_roots(ℓ, [(tor_wall(mech), ri), (tor_wall(mech), 1.0)], ri; count=5)
                # The stress-free rigid rotation (ℓ = 1, k = 0) is removed by the
                # zero-angular-momentum condition; every other mode is kept.
                @test relerr(block_k(Ab, Bb, p.E), exact[1:4]) < 1e-9
            else
                wall = last(flow_walls(mech))
                exact = exact_roots(ℓ, [(:value, ri), (wall, ri), (:value, 1.0), (wall, 1.0)],
                                    ri; poloidal=true)
                @test relerr(block_k(Ab, Bb, p.E), exact) < 1e-8
            end
        end
    end
end

# Exact roots for the MHD blocks.
mech_walls(bci, bco, ri) = (bci == 1 ? :slope : :curvature, bco == 1 ? :slope : :curvature)
function mhd_exact(field, ℓ, p)
    ri = p.ricb
    if field === :h
        return exact_roots(ℓ, [(p.bci_thermal == 0 ? :value : :slope, ri),
                               (p.bco_thermal == 0 ? :value : :slope, 1.0)], ri)
    elseif field === :v
        return exact_roots(ℓ, [(p.bci == 1 ? :value : :stress, ri),
                               (p.bco == 1 ? :value : :stress, 1.0)], ri; count=5)
    elseif field === :u
        wi, wo = mech_walls(p.bci, p.bco, ri)
        return exact_roots(ℓ, [(:value, ri), (wi, ri), (:value, 1.0), (wo, 1.0)], ri;
                           poloidal=true)
    elseif field === :f
        return exact_roots(ℓ, [(p.bci_magnetic == 0 ? :inner_potential : :value, ri),
                               (p.bco_magnetic == 0 ? :outer_potential : :value, 1.0)], ri)
    else
        return exact_roots(ℓ, [(p.bci_magnetic == 0 ? :value : :conductor_tangential, ri),
                               (p.bco_magnetic == 0 ? :value : :conductor_tangential, 1.0)], ri)
    end
end
diffusivity(field, p) = field === :h ? p.E / p.Pr : field in (:f, :g) ? p.Em : p.E
# Stress-free toroidal ℓ = 1 has the rigid rotation at k = 0; drop it where the
# assembly keeps it (no angular-momentum condition) and skip it where removed.
function compare(k_num, exact, field, ℓ, p, gauge)
    if field === :v
        rigid = p.bci == 0 && p.bco == 0 && ℓ == 1
        if rigid && !gauge
            k_num = k_num[2:end]                # the numerical k ≈ 0 mode
        end
        return relerr(k_num[1:3], exact[1:3])
    end
    return relerr(k_num, exact)
end

@testset "MHD tau and energy-Galerkin decay rates for every wall type" begin
    # Each block depends only on its own walls, so four configurations cover
    # every mechanical, thermal, and magnetic wall pair.
    walls = zip(((1, 1), (0, 0), (1, 0), (0, 1)), ((0, 0), (1, 1), (0, 1), (1, 0)),
                ((0, 0), (2, 2), (0, 2), (2, 0)))
    for bg in (axial, dipole), ri in (0.35, 0.6), ((bci, bco), (ti, to), (mi, mo)) in walls
        p = MHDParams(E=1e-2, Pr=0.5, Pm=2.0, Ra=1e-12, Le=1e-6, ricb=ri, m=0, lmax=3,
                      N=36, symm=0, B0_type=bg, bci=bci, bco=bco, bci_thermal=ti,
                      bco_thermal=to, bci_magnetic=mi, bco_magnetic=mo)
        op = MHDStabilityOperator(p)
        # Energy-conserving Galerkin: diagonal blocks of the assembled pencil.
        A, B, layout = M.assemble_mhd_energy_galerkin(op)
        gauge = M._mhd_angular_momentum_gauge(op)
        for ((field, ℓ), rng) in layout.index_map
            ℓ > 2 && continue
            k = block_k(A[rng, rng], B[rng, rng], diffusivity(field, p); count=field === :v ? 5 : 4)
            @test compare(k, mhd_exact(field, ℓ, p), field, ℓ, p, gauge) < 1e-8
        end
        # Tau pencil: each block with its own boundary rows.
        At, Bt, _, _ = quiet(() -> assemble_mhd_matrices(op))
        for ((ℓ, field), b) in M._mhd_index_map(op)
            ℓ > 2 && continue
            k = tau_block_k(At[b, b], Bt[b, b], diffusivity(field, p); count=field === :v ? 5 : 4)
            @test compare(k, mhd_exact(field, ℓ, p), field, ℓ, p, gauge) < 1e-8
        end
    end
end

end # module
