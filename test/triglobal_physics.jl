using Test
using Logging
using LinearAlgebra
using Magrathea

# End-to-end invariants of the coupled (triglobal) eigenproblem. Each has an exact
# expected answer that does not depend on how the blocks are assembled.
const _TG = Magrathea

@testset "Triglobal eigenproblem invariants" begin
    χ, E, Ra, Pr, Nr, lmax = 0.35, 1e-2, 2e3, 1.0, 12, 6
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    p = OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=0, lmax=lmax, Nr=Nr)
    spectrum(bs, mr; nev=20) = with_logger(NullLogger()) do
        solve(TriglobalProblem(p, bs, mr); nev=nev, backend=:dense, verbose=false).eigenvalues
    end
    setdist(a, b) = maximum(minimum(abs.(z .- b)) for z in a)

    # An axisymmetric 3D state leaves the azimuthal orders uncoupled: the spectrum
    # is the union of biglobal spectra (complex conjugates for m < 0).
    axis3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 0, Dict((2, 0) => 0.1))
    biglobal = ComplexF64[]
    for m in -2:2
        pm = OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=abs(m), lmax=lmax, Nr=Nr)
        v = with_logger(NullLogger()) do
            solve(BiglobalProblem(pm, _TG._axisymmetric_state(axis3d)); nev=8, backend=:dense).eigenvalues
        end
        append!(biglobal, m < 0 ? conj.(v) : v)
    end
    @test setdist(spectrum(axis3d, -2:2; nev=8), biglobal) < 1e-10

    # Rotating the state in longitude, cos 2φ → cos 2(φ-α), leaves every eigenvalue
    # unchanged; this fixes the cosine/sine (signed-m) coupling phases.
    state(α) = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 2,
        Dict((2, 2) => 0.1cos(2α), (2, -2) => 0.1sin(2α), (2, 0) => 0.05))
    λ0 = spectrum(state(0.0), -3:3)
    for α in (0.3, π / 4)
        @test setdist(spectrum(state(α), -3:3)[1:10], λ0) < 1e-10
    end

    # A purely m = 2 state only couples different orders, so non-degenerate
    # eigenvalues move at second order in its amplitude.
    leading(ε) = first(spectrum(nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 2,
                                                            Dict((2, 2) => ε)), 0:4; nev=2))
    λ_ref = leading(0.0)
    @test abs(leading(2e-3) - λ_ref) / abs(leading(1e-3) - λ_ref) ≈ 4 rtol=2e-3
end

@testset "Triglobal stress-free perturbations" begin
    χ, E, Ra, Pr, Nr, lmax = 0.35, 1e-2, 5e3, 1.0, 16, 6
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    p = OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=0, lmax=lmax, Nr=Nr, mechanical_bc=:stress_free)
    quiet(f) = with_logger(f, NullLogger())

    # Uncoupled limit: the stress-free biglobal spectra, including the m = 0 and
    # m = ±1 blocks that carry the angular-momentum constraint.
    axis3d = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 0, Dict((2, 0) => 0.1);
                                         mechanical_bc=:stress_free)
    tri = quiet(() -> solve(TriglobalProblem(p, axis3d, -1:1); nev=8, backend=:dense,
                            verbose=false).eigenvalues)
    biglobal = ComplexF64[]
    for m in -1:1
        pm = OnsetParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m=abs(m), lmax=lmax, Nr=Nr,
                         mechanical_bc=:stress_free)
        v = quiet(() -> solve(BiglobalProblem(pm, _TG._axisymmetric_state(axis3d)); nev=8,
                              backend=:dense).eigenvalues)
        append!(biglobal, m < 0 ? conj.(v) : v)
    end
    @test maximum(minimum(abs.(z .- biglobal)) for z in tri) < 1e-10

    # Coupled solve: no rigid-rotation eigenvalue (0 or ±i) survives, and the
    # leading mode is impermeable and free of tangential stress at both walls.
    bs = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 2, Dict((2, 2) => 0.3, (2, 0) => 0.1);
                                     mechanical_bc=:stress_free)
    params = TriglobalParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m_range=-2:2, lmax=lmax, Nr=Nr,
                             basic_state_3d=bs, mechanical_bc=:stress_free)
    vals, vecs = quiet(() -> _TG.solve_triglobal_eigenvalue_problem(params; nev=12,
                                                                      backend=:dense, verbose=false))
    @test minimum(min(abs(z), abs(z - im), abs(z + im)) for z in vals) > 1e-3
    problem = _TG.setup_coupled_mode_problem(params)
    for φ in (0.0, 0.7)
        ur, uθ, uφ = _TG.eigenvector_to_velocity_triglobal(vecs[:, 1], problem; φ_slice=φ)
        scale = maximum(abs, (maximum(abs, ur), maximum(abs, uθ), maximum(abs, uφ)))
        @test maximum(abs, ur[[1, Nr], :]) < 1e-10 * scale
    end
    # Zero tangential stress in every harmonic of every coupled block. With
    # u = ∇×∇×(P𝐫) + ∇×(T𝐫), ∂(u_θ/r)/∂r and ∂(u_φ/r)/∂r combine P''/r - 2P/r³
    # and (T' - T/r)/r, so the walls need P = P'' = 0 and T' = T/r. (Differentiating
    # the reconstructed u_θ/r is only good to ~1e-4 here: it is not a polynomial
    # this coarse grid represents exactly.)
    r = cd.x; D1 = Matrix(cd.D1); D2 = Matrix(cd.D2)
    for m in -2:2
        P, T = _TG._extract_mode_coefficients(vecs[:, 1], problem, m)
        for prof in values(P), wall in (1, Nr)
            s = maximum(abs, D2 * prof) + eps()
            @test abs(prof[wall]) < 1e-10 * s
            @test abs(dot(D2[wall, :], prof)) < 1e-10 * s
        end
        for prof in values(T), wall in (1, Nr)
            s = maximum(abs, D1 * prof) + eps()
            @test abs(dot(D1[wall, :], prof) - prof[wall] / r[wall]) < 1e-10 * s
        end
    end
end

# Kinetic and thermal energy budgets of the leading coupled eigenmode, evaluated by
# quadrature in physical space from the reconstructed mode and the mean state:
#   σ∫|u|² = -Re∫u*·(u·∇)U + β Re∫r θ u_r* - (E/2)∫|∇u + ∇uᵀ|²,
#   σ∫|θ|² = -Re∫θ*(u·∇T̄) - (E/Pr)∫|∇θ|²,
# with σ = Re λ. Coriolis, pressure and advection by U do no net work; the strain
# form of the dissipation holds for no-slip and stress-free walls alike.
_tg_er(θ, φ) = (sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
_tg_eθ(θ, φ) = (cos(θ) * cos(φ), cos(θ) * sin(φ), -sin(θ))
_tg_eφ(θ, φ) = (-sin(φ), cos(φ), 0.0)
_tg_cart(a, b, c, θ, φ) = a .* _tg_er(θ, φ) .+ b .* _tg_eθ(θ, φ) .+ c .* _tg_eφ(θ, φ)

# (u_r, u_θ, u_φ, θ) of each azimuthal block on radius × colatitude at φ = 0.
function _tg_block_fields(blocks, r, D, θv, lmax)
    n = length(θv)
    out = Dict{Int,Any}()
    for (m, (P, T, Θ)) in blocks
        grid = _TG.MeridionalGrid{Float64}(θv, cos.(θv), sin.(θv), zeros(n, n), abs(m), zeros(n, n),
                                           Dict{Int,Vector{ComplexF64}}(), lmax)
        ur, uθ, uφ = _TG._onset_velocity_from_coefficients(P, T, r, D, grid, m)
        g = _TG.SHGrid{Float64}(maximum(keys(Θ)), abs(m), cos.(θv), zeros(n), [0.0])
        out[m] = (ur, uθ, uφ, sum(p * transpose(_TG._coupling_harmonic(g, l, m)[1]) / sqrt(2l + 1)
                                  for (l, p) in Θ))
    end
    out
end

# Cartesian and spherical velocity and the temperature of the whole mode on an
# (r, θ, φ) grid: the block fields summed with e^{imφ}.
function _tg_mode_fields(F, θv, φv)
    Nr, n = size(first(values(F))[1])
    sph = zeros(ComplexF64, Nr, n, length(φv), 3); temp = zeros(ComplexF64, Nr, n, length(φv))
    for (k, φ) in enumerate(φv), (m, f) in F
        for c in 1:3
            sph[:, :, k, c] .+= f[c] .* cis(m * φ)
        end
        temp[:, :, k] .+= f[4] .* cis(m * φ)
    end
    cart = similar(sph)
    for k in eachindex(φv), j in 1:n, i in 1:Nr
        cart[i, j, k, :] .= _tg_cart(sph[i, j, k, 1], sph[i, j, k, 2], sph[i, j, k, 3], θv[j], φv[k])
    end
    cart, sph, temp
end

# Cartesian mean velocity on colatitude × longitude at radius r, from the state's
# divergence-free vector potentials.
function _tg_mean_velocity(flow, r, θv, φv)
    g = _TG.SHGrid{Float64}(flow.lmax, flow.mmax, cos.(θv), ones(length(θv)), collect(φv))
    at(d) = Dict(key => [_TG._mean_barycentric(flow.r, v, r)] for (key, v) in d)
    f = _TG.SolenoidalMeanFlow(flow.lmax, flow.mmax, [r], at(flow.p), at(flow.t), at(flow.dp),
                               at(flow.d2p), at(flow.dt))
    ur, uθ, uφ = _TG._mean_flow_grid(f, 1, g)
    [_tg_cart(ur[j, k], uθ[j, k], uφ[j, k], θv[j], φv[k])[c]
     for j in eachindex(θv), k in eachindex(φv), c in 1:3]
end

# Mean temperature (or its φ-derivative) summed independently of the solver from the
# public real-harmonic coefficients: no factorial normalization, cos mφ for m ≥ 0
# and sin|m|φ for m < 0.
function _tg_mean_temperature(bs, coeffs, r, θv, φv; dφ=false)
    out = zeros(length(θv), length(φv))
    for ((l, m), prof) in coeffs
        a = abs(m)
        c = _TG._mean_barycentric(bs.r, prof, r) *
            (a == 0 ? 1.0 : Float64(sqrt(2 * factorial(big(l + a)) / factorial(big(l - a)))))
        Q = _TG._normalized_legendre_table(a, l, cos.(θv))[l - a + 1, :]
        trig = m >= 0 ? (dφ ? φ -> -m * sin(m * φ) : φ -> cos(m * φ)) :
                        (dφ ? φ -> a * cos(a * φ) : φ -> sin(a * φ))
        out .+= c .* Q .* transpose(trig.(φv))
    end
    out
end

function _triglobal_energy_budgets(params; bs=params.basic_state_3d, Nq=40, Nθ=16, Nφ=12, h=1e-5)
    χ, E, Pr, Ra = params.χ, params.E, params.Pr, params.Ra
    β, κ = Ra * E^2 / (Pr * (1 - χ)^3), E / Pr
    vals, vecs = _TG.solve_triglobal_eigenvalue_problem(params; nev=4, backend=:dense, verbose=false)
    problem = _TG.setup_coupled_mode_problem(params)

    # Chebyshev quadrature in r, Gauss–Legendre in cos θ and uniform in φ, all exact
    # or spectrally accurate here; angular derivatives are central differences.
    cq = ChebyshevDiffn(Nq, [χ, 1.0], 1); r = collect(cq.x); D = Matrix(cq.D1)
    μ, wμ = _TG._gauss_legendre_nodes(Nθ); θq = acos.(μ); φq = [2π * (k - 1) / Nφ for k in 1:Nφ]
    W = [wr * ri^2 * wj * 2π / Nφ for (wr, ri) in zip(_TG._mean_radial_weights(r), r), wj in wμ]

    # Potentials (P, T, Θ) of each azimuthal block, interpolated onto the quadrature radii.
    rN = collect(ChebyshevDiffn(params.Nr, [χ, 1.0], 1).x)
    blocks = Dict{Int,Any}()
    for m in params.m_range
        rec = _TG._mode_reconstruction(problem, abs(m)); op = rec.op
        full = _TG._reconstruct_full_vector(rec.reduction, vecs[problem.block_indices[m], 1])
        profile(l, f) = [_TG._mean_barycentric(rN, full[op.index_map[(l, f)]], x) for x in r]
        blocks[m] = Tuple(Dict(l => profile(l, f) for l in op.l_sets[f]) for f in (:P, :T, :Θ))
    end
    fields(θv, φv) = _tg_mode_fields(_tg_block_fields(blocks, r, D, θv, params.lmax), θv, φv)
    u, us, θp = fields(θq, φq)
    uθ₊, _, θθ₊ = fields(θq .+ h, φq); uθ₋, _, θθ₋ = fields(θq .- h, φq)
    uφ₊, _, θφ₊ = fields(θq, φq .+ h); uφ₋, _, θφ₋ = fields(θq, φq .- h)
    dru = reshape(D * reshape(u, Nq, :), size(u)); drθ = D * reshape(θp, Nq, :)
    dθu, dφu = (uθ₊ - uθ₋) / 2h, (uφ₊ - uφ₋) / 2h
    dθθ, dφθ = (θθ₊ - θθ₋) / 2h, (θφ₊ - θφ₋) / 2h

    K = Θ2 = shear = buoyancy = dissipation = advection = diffusion = 0.0
    Uat(ri, θv, φv) = _tg_mean_velocity(bs.flow, ri, θv, φv)
    Tat(ri, θv; dφ=false) = _tg_mean_temperature(bs, bs.theta_coeffs, ri, θv, φq; dφ=dφ)
    for i in 1:Nq
        ri = r[i]
        # Mean-state gradients; r ± h leaves the shell on the walls, where u_r = 0.
        dUr = i in (1, Nq) ? zeros(Nθ, Nφ, 3) : (Uat(ri + h, θq, φq) - Uat(ri - h, θq, φq)) / 2h
        dUθ = (Uat(ri, θq .+ h, φq) - Uat(ri, θq .- h, φq)) / 2h
        dUφ = (Uat(ri, θq, φq .+ h) - Uat(ri, θq, φq .- h)) / 2h
        dTr = _tg_mean_temperature(bs, bs.dtheta_dr_coeffs, ri, θq, φq)
        dTθ = (Tat(ri, θq .+ h) - Tat(ri, θq .- h)) / 2h
        dTφ = Tat(ri, θq; dφ=true)
        for k in 1:Nφ, j in 1:Nθ
            θ, φ, w, s = θq[j], φq[k], W[i, j], sin(θq[j])
            vr, vθ, vφ = us[i, j, k, :]; t = θp[i, j, k]
            er, eθ, eφ = _tg_er(θ, φ), _tg_eθ(θ, φ), _tg_eφ(θ, φ)
            G = [dru[i, j, k, c] * er[d] + dθu[i, j, k, c] / ri * eθ[d] + dφu[i, j, k, c] / (ri * s) * eφ[d]
                 for c in 1:3, d in 1:3]
            adv = vr * dUr[j, k, :] + vθ / ri * dUθ[j, k, :] + vφ / (ri * s) * dUφ[j, k, :]
            K += w * sum(abs2, u[i, j, k, :])
            shear -= w * real(dot(u[i, j, k, :], adv))
            buoyancy += β * w * real(ri * t * conj(vr))
            dissipation += E / 2 * w * sum(abs2, G + transpose(G))
            Θ2 += w * abs2(t)
            advection -= w * real(conj(t) * (vr * dTr[j, k] + vθ / ri * dTθ[j, k] + vφ / (ri * s) * dTφ[j, k]))
            diffusion += κ * w * (abs2(drθ[i, j + Nθ * (k - 1)]) + abs2(dθθ[i, j, k]) / ri^2 +
                                  abs2(dφθ[i, j, k]) / (ri * s)^2)
        end
    end
    σ = real(vals[1])
    (kinetic=abs(σ * K - (shear + buoyancy - dissipation)) / dissipation,
     thermal=abs(σ * Θ2 - (advection - diffusion)) / diffusion,
     shear=abs(shear) / dissipation)
end

@testset "Triglobal energy budgets" begin
    # The leading mode of each coupled family closes both budgets to ≤ 2e-7 at
    # Nr = 20 (spectral convergence in Nr). Solving with the m = 2 forcing 1% too
    # strong raises the kinetic mismatch to ≥ 1.9e-4 and the thermal one to ≥ 2.5e-3.
    χ, E, Ra, Pr, Nr = 0.35, 1e-2, 5e3, 1.0, 20
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    for (bc, m_range) in ((:no_slip, 0:2), (:stress_free, -1:1))
        bs = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 2, Dict((2, 2) => 0.3, (2, 0) => 0.1);
                                         mechanical_bc=bc)
        params = TriglobalParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m_range=m_range, lmax=6, Nr=Nr,
                                 basic_state_3d=bs, mechanical_bc=bc)
        b = with_logger(() -> _triglobal_energy_budgets(params), NullLogger())
        @test b.kinetic < 1e-5
        @test b.thermal < 1e-6
        @test b.shear > 1e-3    # the mean flow does work on the mode
    end
end

# MHD version: kinetic plus magnetic budget of the leading coupled mode with
# perfectly conducting no-slip walls, where no Poynting flux leaves the shell and the
# Lorentz work of the mean field cancels its induction:
#   σ(K + Le²M) = shear + buoyancy + Le² Re∫u*·(J̄×b) + Le² Re∫j*·(U×b)
#                 - (E/2)∫|∇u + ∇uᵀ|² - Le² E_m ∫|j|²,
# with J̄ = ∇×b̄ the induced mean current and j = ∇×b, whose potentials are G and
# -D_ℓ F for b = ∇×∇×(F𝐫) + ∇×(G𝐫).
_tg_cross(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])

function _tg_mean_field_grid(f, r, θv, φv; curl=false)
    g = _TG.SHGrid{Float64}(f.lmax, f.mmax, cos.(θv), ones(length(θv)), collect(φv))
    at(d) = Dict(key => [_TG._mean_barycentric(f.r, v, r)] for (key, v) in d)
    one_r = _TG.SolenoidalMeanFlow(f.lmax, f.mmax, [r], at(f.p), at(f.t), at(f.dp), at(f.d2p), at(f.dt))
    curl ? _TG._mean_vorticity_grid(one_r, 1, g) : _TG._mean_flow_grid(one_r, 1, g)
end

function _triglobal_mhd_budget(params; Nq=40, Nθ=16, Nφ=12, h=1e-5)
    bs = params.basic_state_3d
    χ, E, Pr, Ra = params.χ, params.E, params.Pr, params.Ra
    β, Le2, Em = Ra * E^2 / (Pr * (1 - χ)^3), params.Le^2, E / params.Pm
    vals, vecs = _TG.solve_triglobal_eigenvalue_problem(params; nev=4, backend=:dense, verbose=false)
    problem = _TG.setup_coupled_mode_problem(params)
    cq = ChebyshevDiffn(Nq, [χ, 1.0], 2); r = collect(cq.x); D = Matrix(cq.D1); D2 = Matrix(cq.D2)
    μ, wμ = _TG._gauss_legendre_nodes(Nθ); θq = acos.(μ); φq = [2π * (k - 1) / Nφ for k in 1:Nφ]
    W = [wr * ri^2 * wj * 2π / Nφ for (wr, ri) in zip(_TG._mean_radial_weights(r), r), wj in wμ]
    rN = collect(ChebyshevDiffn(params.Nr, [χ, 1.0], 1).x)
    flow_blocks = Dict{Int,Any}(); field_blocks = Dict{Int,Any}(); current_blocks = Dict{Int,Any}()
    for m in params.m_range
        rec = _TG._mode_reconstruction(problem, abs(m)); op = rec.op
        full = _TG._reconstruct_full_vector(rec.reduction, vecs[problem.block_indices[m], 1])
        profile(l, f) = [_TG._mean_barycentric(rN, full[op.index_map[(l, f)]], x) for x in r]
        flow_blocks[m] = Tuple(Dict(l => profile(l, f) for l in op.l_sets[f]) for f in (:P, :T, :Θ))
        F = Dict(l => profile(l, :F) for l in op.l_sets[:F])
        G = Dict(l => profile(l, :G) for l in op.l_sets[:G])
        none = Dict(l => zero(r) for l in op.l_sets[:F])
        field_blocks[m] = (F, G, none)
        current_blocks[m] = (G, Dict(l => -(D2 * v .+ 2 .* (D * v) ./ r .- l * (l + 1) .* v ./ r .^ 2)
                                     for (l, v) in F), none)
    end
    fields(bl, θv, φv) = _tg_mode_fields(_tg_block_fields(bl, r, D, θv, params.lmax), θv, φv)
    u, us, θp = fields(flow_blocks, θq, φq)
    uθ₊, _, _ = fields(flow_blocks, θq .+ h, φq); uθ₋, _, _ = fields(flow_blocks, θq .- h, φq)
    uφ₊, _, _ = fields(flow_blocks, θq, φq .+ h); uφ₋, _, _ = fields(flow_blocks, θq, φq .- h)
    _, bsph, _ = fields(field_blocks, θq, φq); _, jsph, _ = fields(current_blocks, θq, φq)
    dru = reshape(D * reshape(u, Nq, :), size(u))
    dθu, dφu = (uθ₊ - uθ₋) / 2h, (uφ₊ - uφ₋) / 2h
    K = Mag = shear = buoyancy = viscous = lorentz = induction = ohmic = 0.0
    Uat(ri, θv, φv) = _tg_mean_velocity(bs.flow, ri, θv, φv)
    for i in 1:Nq
        ri = r[i]
        dUr = i in (1, Nq) ? zeros(Nθ, Nφ, 3) : (Uat(ri + h, θq, φq) - Uat(ri - h, θq, φq)) / 2h
        dUθ = (Uat(ri, θq .+ h, φq) - Uat(ri, θq .- h, φq)) / 2h
        dUφ = (Uat(ri, θq, φq .+ h) - Uat(ri, θq, φq .- h)) / 2h
        Ū = _tg_mean_field_grid(bs.flow, ri, θq, φq)
        J̄ = _tg_mean_field_grid(bs.field, ri, θq, φq; curl=true)
        for k in 1:Nφ, j in 1:Nθ
            θ, φ, w, s = θq[j], φq[k], W[i, j], sin(θq[j])
            vr, vθ, vφ = us[i, j, k, :]; t = θp[i, j, k]
            er, eθ, eφ = _tg_er(θ, φ), _tg_eθ(θ, φ), _tg_eφ(θ, φ)
            G = [dru[i, j, k, c] * er[d] + dθu[i, j, k, c] / ri * eθ[d] + dφu[i, j, k, c] / (ri * s) * eφ[d]
                 for c in 1:3, d in 1:3]
            adv = vr * dUr[j, k, :] + vθ / ri * dUθ[j, k, :] + vφ / (ri * s) * dUφ[j, k, :]
            b = Tuple(bsph[i, j, k, c] for c in 1:3); jj = Tuple(jsph[i, j, k, c] for c in 1:3)
            K += w * sum(abs2, u[i, j, k, :]); Mag += w * sum(abs2, b)
            shear -= w * real(dot(u[i, j, k, :], adv))
            buoyancy += β * w * real(ri * t * conj(vr))
            viscous += E / 2 * w * sum(abs2, G + transpose(G))
            lorentz += Le2 * w * real(sum(conj.((vr, vθ, vφ)) .* _tg_cross(Tuple(J̄[c][j, k] for c in 1:3), b)))
            induction += Le2 * w * real(sum(conj.(jj) .* _tg_cross(Tuple(Ū[c][j, k] for c in 1:3), b)))
            ohmic += Le2 * Em * w * sum(abs2, jj)
        end
    end
    σ = real(vals[1])
    (mismatch=abs(σ * (K + Le2 * Mag) - (shear + buoyancy + lorentz + induction - viscous - ohmic)) /
              (viscous + ohmic),
     mean_field=(abs(lorentz) + abs(induction)) / (viscous + ohmic))
end

@testset "Triglobal MHD energy budget" begin
    # With the induced field of a non-axisymmetric MHD state the budget closes to
    # ≈ 1e-9 at Nr = 24 (8e-10 measured); its mean-field terms carry ~10% of it.
    χ, E, Ra, Pr, Nr = 0.35, 1e-2, 5e3, 1.0, 24
    cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)
    mag = (B0_type=axial, Le=0.1, Pm=1.0, magnetic_bc=:perfect_conductor)
    bs = nonaxisymmetric_basic_state(cd, χ, E, Ra, Pr, 4, 2, Dict((2, 2) => 0.3, (2, 0) => 0.1); mag...)
    params = TriglobalParams(E=E, Pr=Pr, Ra=Ra, χ=χ, m_range=0:2, lmax=6, Nr=Nr, basic_state_3d=bs; mag...)
    b = with_logger(() -> _triglobal_mhd_budget(params), NullLogger())
    @test b.mismatch < 1e-7
    @test b.mean_field > 1e-2
end
