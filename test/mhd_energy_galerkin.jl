module MHDEnergyGalerkinChecks
using Test, Magrathea, LinearAlgebra, SparseArrays, Logging
const M = Magrathea
quiet(f) = with_logger(f, NullLogger())

mhd(; kw...) = MHDParams(; merge((E=1e-3, Pr=1.0, Pm=1.0, Ra=1e-8, Le=1.0, ricb=0.35,
                                  m=1, lmax=5, N=16, symm=0, B0_type=dipole), (; kw...))...)
fields(layout, which) = sort(reduce(vcat, [collect(rng) for ((f, _), rng) in layout.index_map
                                           if f in which]; init=Int[]))

# Finite spectrum of a tau pencil after eliminating its boundary rows.
function tau_spectrum(A, B)
    A = Matrix(A); B = Matrix(B)
    bc = findall(i -> iszero(B[i, :]), axes(B, 1)); interior = setdiff(axes(B, 1), bc)
    R = nullspace(A[bc, :])
    λ = eigvals(A[interior, :] * R, B[interior, :] * R)
    λ[isfinite.(λ)]
end
leading(λ, k) = sort(λ, by=z -> -real(z))[1:k]

@testset "Energy-conserving MHD Galerkin: exact energy balance" begin
    # B is Hermitian positive definite on the velocity and magnetic blocks, which
    # hold the energy with the field measured as Le·b. The Hermitian part of A is
    # dissipative: Coriolis, Lorentz, and induction exchange energy exactly.
    for B0 in (axial, dipole), mech in ((1, 1), (0, 0), (1, 0)), mag in ((0, 0), (2, 2), (2, 0))
        p = mhd(B0_type=B0, Le=B0 == dipole ? 1.0 : 5.0, bci=mech[1], bco=mech[2],
                bci_magnetic=mag[1], bco_magnetic=mag[2])
        A, B, layout = M.assemble_mhd_energy_galerkin(MHDStabilityOperator(p))
        keep = fields(layout, (:u, :v, :f, :g))
        Bk = Matrix(B[keep, keep])
        @test norm(Bk - Bk') <= 1e-12 * norm(Bk)
        @test isposdef(Hermitian((Bk + Bk') / 2))
        Ak = Matrix(A[keep, keep])
        H = Hermitian((Ak + Ak') / 2)
        @test eigmax(H) <= 1e-10 * opnorm(H)
        # Hence no eigenvalue grows, even at this low resolution.
        @test maximum(real, eigvals(Matrix(A), Matrix(B))) < 1e-10
    end
end

@testset "Energy-conserving MHD Galerkin matches resolved tau spectra" begin
    for B0 in (axial, dipole), mech in ((1, 1), (0, 0)), mag in ((0, 0), (2, 2), (0, 2))
        p = MHDParams(E=1e-2, Pr=1.0, Pm=1.0, Ra=50.0, Le=B0 == dipole ? 0.02 : 0.1,
                      ricb=0.35, m=1, lmax=4, N=32, symm=0, B0_type=B0, bci=mech[1],
                      bco=mech[2], bci_magnetic=mag[1], bco_magnetic=mag[2])
        op = MHDStabilityOperator(p)
        At, Bt, _, _ = quiet(() -> assemble_mhd_matrices(op))
        Ae, Be, _ = M.assemble_mhd_energy_galerkin(op)
        λt = leading(tau_spectrum(At, Bt), 3)
        λe = leading(eigvals(Matrix(Ae), Matrix(Be)), 3)
        @test all(abs.(λe .- λt) .<= 1e-5 .* abs.(λt))
    end
end

@testset "Energy-conserving MHD Galerkin in Float32" begin
    p = MHDParams(E=1f-3, Pr=1f0, Pm=1f0, Ra=1f-3, Le=1f0, ricb=0.35f0, m=1, lmax=4, N=12,
                  symm=1, B0_type=dipole, B0_amplitude=1f0)
    @test p isa MHDParams{Float32}
    A, B, _ = M.assemble_mhd_energy_galerkin(MHDStabilityOperator(p))
    @test eltype(A) === ComplexF32 && eltype(B) === ComplexF32
    r = quiet(() -> solve(MHDProblem(p); nev=2, backend=:dense))
    @test real(r.eigenvalues[1]) < 0
end

@testset "Finite-conductivity core keeps the tau assembly" begin
    p = mhd(B0_type=axial, bci_magnetic=1, Le=0.1)
    @test !M._mhd_energy_galerkin_supported(p)
    @test_throws ArgumentError M.assemble_mhd_energy_galerkin(MHDStabilityOperator(p))
    r = quiet(() -> solve(MHDProblem(p); nev=2, backend=:dense))
    @test !haskey(r.extra, :galerkin_layout)
end

end # module
