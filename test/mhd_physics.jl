using Test, Magrathea, LinearAlgebra, SparseArrays, Logging

function _mhd_check_op(; kwargs...)
    defaults = (; E=.01, Pr=1., Pm=1., Ra=1., Le=.1, ricb=.35,
                m=0, lmax=3, N=16, B0_type=axial, symm=0)
    with_logger(NullLogger()) do
        MHDStabilityOperator(MHDParams(; merge(defaults, (;kwargs...))...))
    end
end
# Independent interpolation, with no coefficient filtering from production code.
function _mhd_check_coeff(f, N; ri=.35, ro=1.)
    x = cos.(π .* (0:N) ./ N)
    V = [cos(n * acos(clamp(t, -1., 1.))) for t in x, n in 0:N]
    ComplexF64.(V \ f.((ri + ro)/2 .+ (ro - ri)/2 .* x))
end
function _mhd_check_raw(op)
    c = Magrathea._assemble_mhd_coo(op)
    sparse(c.A_rows,c.A_cols,c.A_vals,c.n,c.n), sparse(c.B_rows,c.B_cols,c.B_vals,c.n,c.n)
end

# Eliminate homogeneous algebraic boundary constraints before a dense eigensolve;
# raw QZ can represent an infinite tau eigenvalue as a huge finite number.
function _mhd_check_spectrum(A,B)
    bc=findall(i->iszero(B[i,:]),axes(B,1))
    interior=setdiff(axes(B,1),bc)
    R=nullspace(Matrix(A[bc,:]))
    eigvals(Matrix(A[interior,:])*R, Matrix(B[interior,:])*R)
end

@testset "MHD degree-one angular selection and parity" begin
    specs = ((Magrathea.operator_lorentz_poloidal_from_bpol, (-1,1), true),
             (Magrathea.operator_lorentz_toroidal_from_bpol, (0,), true),
             (Magrathea.operator_lorentz_toroidal_from_btor, (-1,1), true),
             (Magrathea.operator_induction_poloidal_from_u, (-1,1), false),
             (Magrathea.operator_induction_poloidal_from_v, (0,), false),
             (Magrathea.operator_induction_toroidal_from_u, (0,), false),
             (Magrathea.operator_induction_toroidal_from_v, (-1,1), false))
    for bg in (axial, dipole), inner in (0,1,2)
        op = _mhd_check_op(B0_type=bg, m=1, N=8, bci_magnetic=inner,
                           bco_magnetic=2, bci=0, bco=0)
        for (fun, allowed, lorentz) in specs, offset in -3:3
            block = lorentz ? fun(op, 3, 1, offset, op.params.Le) : fun(op, 3, 1, offset)
            offset in allowed || @test iszero(block)
        end
        A,B,_,_ = assemble_mhd_matrices(op)
        index = Magrathea._mhd_index_map(op)
        partitions = Vector{Int}[]
        for symmetry in (-1,1)
            half = _mhd_check_op(B0_type=bg, m=1, N=8, symm=symmetry,
                                 bci_magnetic=inner, bco_magnetic=2, bci=0, bco=0)
            ah,bh,_,_ = assemble_mhd_matrices(half)
            hm = Magrathea._mhd_index_map(half)
            perm = reduce(vcat, [collect(index[k]) for k in sort!(collect(keys(hm)); by=k->first(hm[k]))])
            push!(partitions, perm)
            @test A[perm,perm] ≈ ah
            @test B[perm,perm] ≈ bh
        end
        @test norm(A[partitions[1],partitions[2]]) < 1e-10
        @test norm(A[partitions[2],partitions[1]]) < 1e-10
    end
end

@testset "MHD current-free perturbation and analytic induction" begin
    for bg in (axial,dipole), m in (0,1)
        op = _mhd_check_op(B0_type=bg, m=m)
        index = Magrathea._mhd_index_map(op); A,_ = _mhd_check_raw(op)
        x = zeros(ComplexF64,op.matrix_size)
        x[index[(2,:f)]] = _mhd_check_coeff(r->r^2,op.params.N)
        rows = reduce(vcat,[collect(b) for ((_,s),b) in index if s in (:u,:v)])
        @test norm(A[rows,:]*x) < 1e-8
    end
    # U=(-yz,xz,0) in a uniform axial field gives ∂z U=(-y,x,0):
    # native T_20=r² induces native g_10=3r.
    op = _mhd_check_op(); index = Magrathea._mhd_index_map(op); A,B = _mhd_check_raw(op)
    x = zeros(ComplexF64,op.matrix_size); expected = zero(x)
    x[index[(2,:v)]] = _mhd_check_coeff(r->r^2,op.params.N)
    expected[index[(1,:g)]] = _mhd_check_coeff(r->3r,op.params.N)
    rows = reduce(vcat,[collect(b) for ((_,s),b) in index if s in (:f,:g)])
    @test norm(A[rows,:]*x-B[rows,:]*expected)/norm(B[rows,:]*expected) < 1e-10
end

@testset "MHD tau free decay agrees with independent collocation" begin
    cd = ChebyshevDiffn(41,[.35,1.],2); r=cd.x
    L = .01 .* (cd.D2 + 2cd.D1./r - Diagonal(2 ./r.^2))
    reference = maximum(real,eigvals(L[2:end-1,2:end-1]))
    for bg in (axial,dipole), N in (12,20)
        op = _mhd_check_op(B0_type=bg,N=N)
        A,B,_,_ = assemble_mhd_matrices(op)
        b = Magrathea._mhd_index_map(op)[(1,:g)]
        λ = _mhd_check_spectrum(A[b,b],B[b,b])
        @test maximum(real,λ) < 0
        @test maximum(real,λ) ≈ reference atol=(N==12 ? 2e-7 : 2e-10)
        # Only the last two residual coefficients were replaced.
        rawA,rawB = _mhd_check_raw(op)
        @test A[b[1:end-2],:] == rawA[b[1:end-2],:]
        @test B[b[1:end-2],:] == rawB[b[1:end-2],:]
        @test iszero(B[b[end-1:end],:])
    end
end

# j1(kr)/(r/ri), evaluated as a power series in s=(r/ri)^2, also at r=0.
function _mhd_check_core_j1(x,k,ri)
    s=(x+1)/2; term=k*ri/3; value=term
    for n in 1:30
        term *= -k^2*ri^2*s/(2n*(2n+3))
        value += term
    end
    value
end
@testset "MHD conducting core: full-sphere free-decay modes" begin
    # Equal diffusivities make the interface invisible for free decay. The
    # first poloidal root is π, and the first toroidal root solves tan(k)=k.
    for bg in (axial,dipole), (section,core,k) in ((:f,:fi,π),(:g,:gi,4.493409457909064))
        op = _mhd_check_op(B0_type=bg,bci_magnetic=1)
        A,B,_,_ = assemble_mhd_matrices(op); index=Magrathea._mhd_index_map(op)
        rows=vcat(collect(index[(1,section)]),collect(index[(1,core)]))
        λ = -.01*k^2
        spectrum=_mhd_check_spectrum(A[rows,rows],B[rows,rows])
        @test maximum(real,spectrum) < 0
        @test maximum(real,spectrum) ≈ λ atol=2e-9
        x=zeros(ComplexF64,op.matrix_size)
        x[index[(1,section)]]=_mhd_check_coeff(r->sin(k*r)/(k*r)^2-cos(k*r)/(k*r),op.params.N)
        x[index[(1,core)]]=_mhd_check_coeff(t->_mhd_check_core_j1(t,k,.35),op.params.N;ri=-1.,ro=1.)
        @test norm((A-λ*B)[rows,:]*x) < 2e-8
        @test Magrathea._mhd_total_dof(op.params)[1] == op.matrix_size
        # Independent monomials verify the regular-core Laplacian.
        d,mass=Magrathea._mhd_core_operators(op,1)
        constant=_mhd_check_coeff(t->1.,op.params.N;ri=-1.,ro=1.)
        quadratic=_mhd_check_coeff(t->(t+1)/2,op.params.N;ri=-1.,ro=1.)
        @test norm(d*constant) < 2e-9
        @test norm(d*quadratic-(10*.01/.35^2).*constant) < 2e-9
    end
end

@testset "MHD perfect-conductor electric conditions" begin
    op=_mhd_check_op(N=32,bci_magnetic=2,bco_magnetic=2)
    A,B,_,_=assemble_mhd_matrices(op); index=Magrathea._mhd_index_map(op)
    x=zeros(ComplexF64,op.matrix_size); block=index[(1,:g)]
    x[block]=_mhd_check_coeff(r->1/r,op.params.N)
    @test norm((A*x)[block[end-1:end]]) < 1e-9
    # Stress-free solid rotation has Uφ=r sinθ/(2√π). The axial-field EMF
    # is cancelled by g_20=-r²/(9Em) at BOTH walls.
    op=_mhd_check_op(bci=0,bco=0,bci_magnetic=2,bco_magnetic=2)
    A,B,_,_=assemble_mhd_matrices(op); index=Magrathea._mhd_index_map(op)
    x=zeros(ComplexF64,op.matrix_size)
    x[index[(1,:v)]]=_mhd_check_coeff(identity,op.params.N)
    x[index[(2,:g)]]=_mhd_check_coeff(r->-r^2/(9op.params.Em),op.params.N)
    block=index[(2,:g)]
    @test norm((A*x)[block[end-1:end]]) < 1e-9
end

@testset "MHD reconstruction uses radius-vector potentials" begin
    op=_mhd_check_op(); index=Magrathea._mhd_index_map(op)
    x=zeros(ComplexF64,op.matrix_size)
    x[index[(1,:f)]]=_mhd_check_coeff(identity,op.params.N)
    br,bθ,bφ,r,g=perturbation_magnetic(x,op)
    @test br ≈ repeat(transpose(g.cosθ ./ sqrt(π)),length(r),1) atol=1e-12
    @test bθ ≈ repeat(transpose(-g.sinθ ./ sqrt(π)),length(r),1) atol=1e-11
    @test iszero(bφ)
    x.=0; x[index[(1,:v)]]=_mhd_check_coeff(identity,op.params.N)
    ur,uθ,uφ,r,g=perturbation_velocity(x,op)
    @test iszero(ur)
    @test iszero(uθ)
    @test uφ ≈ r*transpose(g.sinθ ./ (2sqrt(π))) atol=1e-12
end

@testset "MHD Galerkin and tau spectra converge together" begin
    for N in (12,20)
        op=_mhd_check_op(m=1,symm=1,N=N)
        A,B,_,_=assemble_mhd_matrices(op)
        ag,bg,_=Magrathea.assemble_mhd_galerkin(op)
        tau=_mhd_check_spectrum(A,B)
        gal=eigvals(ag,bg)
        lead=gal[argmax(real.(gal))]
        @test maximum(real,tau) ≈ real(lead) atol=(N==12 ? 2e-5 : 1e-8)
        @test minimum(abs.(tau.-lead)) < (N==12 ? 2e-5 : 1e-8)
    end
    op=_mhd_check_op(m=1,N=8)
    A,B,layout=Magrathea.assemble_mhd_galerkin(op)
    sectors=Vector{Int}[]
    for symmetry in (-1,1)
        half=_mhd_check_op(m=1,N=8,symm=symmetry)
        ah,bh,lh=Magrathea.assemble_mhd_galerkin(half)
        perm=reduce(vcat,[collect(layout.index_map[k]) for k in sort!(collect(keys(lh.index_map));by=k->first(lh.index_map[k]))])
        push!(sectors,perm)
        @test A[perm,perm] ≈ ah
        @test B[perm,perm] ≈ bh
    end
    @test iszero(A[sectors[1],sectors[2]])
    @test iszero(A[sectors[2],sectors[1]])
end

@testset "MHD conducting-core electric matching with slip" begin
    for bg in (axial,dipole)
        op=_mhd_check_op(B0_type=bg,bci=0,bci_magnetic=1)
        A,B,_,_=assemble_mhd_matrices(op); index=Magrathea._mhd_index_map(op)
        x=zeros(ComplexF64,op.matrix_size)
        x[index[(1,:v)]]=_mhd_check_coeff(identity,op.params.N)
        # Interface g=0 on both sides; the core derivative cancels the EMF
        # of solid rotation in the imposed radial field.
        c=bg==axial ? .35^2/(12op.params.Em) : 1/(12op.params.Em*.35)
        x[index[(2,:gi)]]=_mhd_check_coeff(t->c*(t-1),op.params.N;ri=-1.,ro=1.)
        @test abs((A*x)[last(index[(2,:g)])]) < 1e-10
        @test abs((A*x)[last(index[(2,:gi)])]) < 1e-9
    end
end

@testset "MHD subcritical tau modes decay with all magnetic walls" begin
    for bg in (axial,dipole), inner in (0,1,2), outer in (0,2)
        op=_mhd_check_op(B0_type=bg,m=1,symm=1,N=12,bci_magnetic=inner,bco_magnetic=outer)
        A,B,_,_=assemble_mhd_matrices(op)
        λ=_mhd_check_spectrum(A,B)
        @test maximum(real,λ) < 0
    end
end
