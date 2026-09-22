using Test, Magrathea, LinearAlgebra, SparseArrays

function coupling_test_state(cd; flow=nothing, ur=Dict{Int,Vector{Float64}}())
    z=Dict{Int,Vector{Float64}}()
    BasicState(lmax_bs=flow===nothing ? 0 : flow.lmax,Nr=length(cd.x),r=cd.x,
        theta_coeffs=z,dtheta_dr_coeffs=z,uphi_coeffs=z,duphi_dr_coeffs=z,
        ur_coeffs=ur,flow=flow)
end

function coupling_test_3d(cd; flow=nothing, temp=Dict{Tuple{Int,Int},Vector{Float64}}())
    z=Dict{Tuple{Int,Int},Vector{Float64}}()
    BasicState3D(lmax_bs=3,mmax_bs=3,Nr=length(cd.x),r=cd.x,
        theta_coeffs=temp,dtheta_dr_coeffs=Dict(k=>cd.D1*v for (k,v) in temp),
        ur_coeffs=z,utheta_coeffs=z,uphi_coeffs=z,dur_dr_coeffs=z,
        dutheta_dr_coeffs=z,duphi_dr_coeffs=z,flow=flow)
end

coupling_test_op(m=1; E=.01,weight=true,Nr=16,bs=nothing) =
    LinearStabilityOperator(OnsetParams(E=E,Ra=0.,χ=.35,m=abs(m),lmax=4,Nr=Nr,
        use_sparse_weighting=weight,basic_state=bs))

function coupling_test_flow(cd,key,p,t)
    Magrathea.SolenoidalMeanFlow(key[1],abs(key[2]),cd.x,Dict(key=>p),Dict(key=>t),
        Dict(key=>cd.D1*p),Dict(key=>cd.D2*p),Dict(key=>cd.D1*t))
end

@testset "Basic-state heat equation uses the onset potential convention" begin
    for m in (0,1,2)
        reference=coupling_test_op(m); cd=reference.cd
        bs=conduction_basic_state(cd,.35,4)
        op=coupling_test_op(m;bs)
        a=Magrathea._assemble_onset_coo(reference)
        b=Magrathea._assemble_onset_coo(op)
        mat(c)=Matrix(sparse(c.A_rows,c.A_cols,c.A_vals,c.n,c.n))
        @test mat(a) ≈ mat(b) atol=2e-11 rtol=2e-11
        # With an explicit mean state the weighting flag must only rescale
        # temperature equations, including their advection/gradient terms.
        other=coupling_test_op(m;bs,weight=false)
        c=Magrathea._assemble_onset_coo(other)
        expected=mat(b)
        for l in op.l_sets[:Θ]
            expected[op.index_map[(l,:Θ)],:] ./= op.r
        end
        @test mat(c) ≈ expected atol=2e-10 rtol=2e-11
    end
end

@testset "Native velocity reconstruction agrees with stability coordinates" begin
    op=coupling_test_op(0); r=op.r; x=zeros(ComplexF64,op.total_dof)
    x[op.index_map[(1,:P)]]=sqrt(4π)/2 .*r # uniform axial translation
    ur,ut,up,g=Magrathea.eigenvector_to_velocity(x,op)
    @test ur ≈ ones(length(r))*transpose(g.cosθ) atol=2e-13
    @test ut ≈ -ones(length(r))*transpose(g.sinθ) atol=2e-13
    @test all(iszero,up)
    fill!(x,0); x[op.index_map[(1,:T)]]=sqrt(4π).*r # rigid rotation
    ur,ut,up,g=Magrathea.eigenvector_to_velocity(x,op)
    @test all(iszero,ur)
    @test all(iszero,ut)
    @test up ≈ r*transpose(g.sinθ) atol=2e-13
end

@testset "Mean-flow heat transport: sign and equation weighting" begin
    for weight in (false,true), m in (0,1,2)
        op=coupling_test_op(m;weight); r=op.r
        # Manufactured divergence-free radial flux; boundary constraints are
        # deliberately absent when testing the interior differential operator.
        bs=coupling_test_state(op.cd;ur=Dict(0=>sqrt(4π)./r.^2))
        C=Magrathea._mean_state_matrix(bs,op,op,m,m)
        x=zeros(ComplexF64,op.total_dof); idx=op.index_map[(max(1,m),:Θ)]
        x[idx]=r.^2
        expected=zeros(ComplexF64,op.total_dof)
        expected[idx]=-2 .*r.^(weight ? 2 : 1)
        @test C*x ≈ expected atol=2e-11 rtol=2e-11
        x[idx].=1
        @test norm(C*x)<2e-11
    end
end

@testset "Exact rigid-rotation linearization in onset coordinates" begin
    for m in (0,1,2), weight in (false,true)
        op=coupling_test_op(m;weight); r=op.r; Ω=.3
        # U=Ω zhat×x: -(U·∇)u-(u·∇)U = -imΩ u - 2Ω zhat×u.
        f=coupling_test_flow(op.cd,(1,0),zero(r),-Ω*sqrt(4π/3).*r.^2)
        bs=coupling_test_state(op.cd;flow=f)
        C=Magrathea._mean_state_matrix(bs,op,op,m,m)
        raw=Magrathea._assemble_onset_radial_coo(op)
        raw2=Magrathea._assemble_onset_radial_coo(coupling_test_op(m;E=.02,weight))
        mat(c)=Matrix(sparse(c.A_rows,c.A_cols,c.A_vals,c.n,c.n))
        B=Matrix(sparse(raw.B_rows,raw.B_cols,raw.B_vals,raw.n,raw.n))
        # Eliminate viscosity/diffusion by extrapolating E to zero.
        expected=-im*m*Ω*B+Ω*(2mat(raw)-mat(raw2))
        velocity=sort(vcat([collect(v) for ((l,f),v) in op.index_map if f!==:Θ]...))
        temperature=sort(vcat([collect(v) for ((l,f),v) in op.index_map if f===:Θ]...))
        @test C[velocity,velocity] ≈ expected[velocity,velocity] atol=2e-9 rtol=2e-11
        @test C[temperature,temperature] ≈ expected[temperature,temperature] atol=2e-12
        # Scalar compatibility projections are not authoritative when flow exists.
        bs.ur_coeffs[0]=fill(1e4,length(r))
        @test Magrathea._mean_state_matrix(bs,op,op,m,m) == C
    end
end

@testset "Nonaxisymmetric physical transport and real/complex normalization" begin
    source=coupling_test_op(0); cd=source.cd; r=cd.x
    for sine in (false,true), weight in (false,true)
        source=coupling_test_op(0;weight)
        # The native l=1 poloidal field is the uniform Cartesian vector ex or ey.
        key=(1,sine ? -1 : 1)
        p=-sqrt(4π/3)/2 .*r.^2
        f=coupling_test_flow(cd,key,p,zero(r))
        bs=coupling_test_3d(cd;flow=f)
        for mt in (-1,1)
            target=coupling_test_op(mt;weight)
            C=Magrathea._mean_state_matrix(bs,source,target,0,mt)
            x=zeros(ComplexF64,source.total_dof)
            x[source.index_map[(2,:Θ)]]=r.^2
            # Θ=(3z²-r²)/(2sqrt(4π)); -ex·∇Θ=x/sqrt(4π),
            # and -ey·∇Θ=y/sqrt(4π). Project these exact linear polynomials.
            expected=zeros(ComplexF64,target.total_dof)
            phase=sine ? -im*mt : 1
            expected[target.index_map[(1,:Θ)]]=-mt*phase/sqrt(2) .*r.^(weight ? 4 : 3)
            @test C*x ≈ expected atol=3e-11 rtol=3e-11
        end
        # Public NF coefficients for Tbar=xz (or yz) differ from orthonormal ones.
        temp=Dict((2,sine ? -1 : 1)=>-r.^2 ./ (3sqrt(5/(2π))))
        bs=coupling_test_3d(cd;temp)
        for mt in (-1,1)
            target=coupling_test_op(mt;weight)
            C=Magrathea._mean_state_matrix(bs,source,target,0,mt)
            x=zeros(ComplexF64,source.total_dof)
            x[source.index_map[(1,:T)]]=sqrt(4π).*r # u=zhat×x
            expected=zeros(ComplexF64,target.total_dof)
            # -∂φ(xz)=yz; -∂φ(yz)=-xz.
            factor=sine ? mt : im
            expected[target.index_map[(2,:Θ)]]=factor*sqrt(2π/3).*r.^(weight ? 5 : 4)
            @test C*x ≈ expected atol=3e-11 rtol=3e-11
        end
    end
end

# Independent projection of polynomial Cartesian momentum forcing. No production
# coupling, spherical-coordinate derivative or vorticity routine is used here.
function cartesian_translation_projection(field,lo,mo)
    μ,w=Magrathea._gauss_legendre_nodes(28); nφ=24; a=abs(mo)
    P=Magrathea._associated_legendre_table(a,lo,μ)
    N=Magrathea._normalization_table(a,lo)[end]/sqrt(2lo+1)
    R=0im; S=0im; Z=0im
    for j in eachindex(μ), k in 0:nφ-1
        z=μ[j]; s=sqrt(1-z*z); φ=2π*k/nφ; x=s*cos(φ); y=s*sin(φ)
        X=[x,y,z]; ex=[1.,0.,0.]; K=sqrt(3/(8π))
        H=-K*z*(x+im*y); grad=-K.*[z,im*z,x+im*y]; dxgrad=[0.,0.,-K]
        F=field===:P ? -(10x.*grad+5 .*dxgrad+4K*z.*X-4H.*ex) :
            cross(ex,grad)+cross(X,dxgrad)
        er=X; eθ=[z*cos(φ),z*sin(φ),-s]; eφ=[-sin(φ),cos(φ),0.]
        phase=mo<0 && isodd(a) ? -1. : 1.
        Y=phase*N*P[end,j]*exp(im*mo*φ)
        Pprev=lo>a ? P[end-1,j] : 0.
        dY=phase*N*(lo*z*P[end,j]-(lo+a)*Pprev)/s*exp(im*mo*φ)
        vY=im*mo*Y/s; q=lo*(lo+1); wt=2π*w[j]/nφ*(2lo+1)
        fr=sum(er.*F); ft=sum(eθ.*F); fp=sum(eφ.*F)
        R+=wt*conj(Y)*fr
        S+=wt*(conj(dY)*ft+conj(vY)*fp)/q
        Z+=wt*(-conj(vY)*ft+conj(dY)*fp)/q
    end
    R,S,Z
end

@testset "Vector momentum coupling agrees with Cartesian polynomial derivatives" begin
    source=coupling_test_op(1;Nr=24); cd=source.cd; r=cd.x
    f=coupling_test_flow(cd,(1,1),-sqrt(4π/3)/2 .*r.^2,zero(r))
    bs=coupling_test_3d(cd;flow=f) # U=ex, so the exact forcing is -∂x u.
    for mt in (0,2), field in (:P,:T)
        target=coupling_test_op(mt;Nr=24)
        C=Magrathea._mean_state_matrix(bs,source,target,1,mt)
        x=zeros(ComplexF64,source.total_dof)
        x[source.index_map[(2,field)]]=r.^(field===:P ? 4 : 2)
        expected=zeros(ComplexF64,target.total_dof)
        # P velocity is 5r²∇H₂-4H₂x; T velocity is -x×∇H₂.
        # Their x derivatives are homogeneous of degree 2 and 1, respectively.
        degree=field===:P ? 2 : 1
        for lo in target.l_sets[:P]
            R,S,Z=cartesian_translation_projection(field,lo,mt); q=lo*(lo+1)
            expected[target.index_map[(lo,:P)]]=-q.*r.^(degree+3).*(R-(degree+1)*S)
            expected[target.index_map[(lo,:T)]]=q.*r.^(degree+2).*Z
        end
        @test C*x ≈ expected atol=3e-8 rtol=3e-8
        if mt==2
            # The bilinear momentum linearization is symmetric in mean and
            # perturbation velocities. Swap the polynomial flow and translation
            # to test a spatially varying mean vorticity and its shear.
            p=field===:P ? sqrt(2/5).*r.^5 : zero(r)
            t=field===:T ? -sqrt(2/5).*r.^3 : zero(r)
            other=coupling_test_3d(cd;flow=coupling_test_flow(cd,(2,1),p,t))
            reverse=Magrathea._mean_state_matrix(other,source,target,1,mt)
            fill!(x,0)
            x[source.index_map[(1,:P)]]=-sqrt(π/2).*r
            @test reverse*x ≈ expected atol=3e-8 rtol=3e-8
        end
    end
end

@testset "2D/3D assembly, signed phases and radial interpolation" begin
    cd=ChebyshevDiffn(24,[.35,1.],4); r=cd.x
    f=coupling_test_flow(cd,(1,1),-sqrt(4π/3)/2 .*r.^2,zero(r))
    bs=coupling_test_3d(cd;flow=f,temp=Dict((2,-1)=>.01 .*r.^2))
    source=coupling_test_op(1); target=coupling_test_op(2)
    C=Magrathea._mean_state_matrix(bs,source,target,1,2)
    Cminus=Magrathea._mean_state_matrix(bs,source,target,-1,-2)
    @test Cminus ≈ -conj(C) rtol=2e-12 atol=2e-10
    # Different mean-state and perturbation grids must preserve polynomial fields.
    short=source.cd
    fshort=coupling_test_flow(short,(1,1),-sqrt(4π/3)/2 .*short.x.^2,zero(short.x))
    bshort=coupling_test_3d(short;flow=fshort,temp=Dict((2,-1)=>.01 .*short.x.^2))
    @test C ≈ Magrathea._mean_state_matrix(bshort,source,target,1,2) rtol=2e-10
    params=TriglobalParams(E=.01,Pr=1.,Ra=0.,χ=.35,m_range=-2:2,lmax=4,Nr=16,basic_state_3d=bs)
    problem=Magrathea.setup_coupled_mode_problem(params)
    single=Magrathea.build_single_mode_operators(problem,false)
    blocks=Magrathea.build_mode_coupling_operators(problem,single,false)
    @test blocks[(1,2)] ≈ Magrathea._project_coupling_block(C,single[2],single[1])
    # An explicitly supplied state with no axisymmetric temperature must not
    # silently acquire the default conductive temperature gradient.
    empty_axis=Magrathea.axisymmetric_basic_state(bs)
    @test !Magrathea._has_nonzero_basic_state(empty_axis)
    explicit=coupling_test_op(1;bs=empty_axis)
    Ae,Be,ie,be=assemble_matrices(explicit)
    Are,Bre,_=Magrathea._constrained_reduced_matrices(Ae,Be,explicit,ie,be)
    @test single[1].A ≈ Are
    @test single[1].B ≈ Bre

    # m=0 basic states must give the same diagonal block in the 2D and 3D APIs.
    ax=nonaxisymmetric_basic_state(short,.35,.01,30.,1.,4,0,Dict((2,0)=>.01))
    params=TriglobalParams(E=.01,Pr=1.,Ra=0.,χ=.35,m_range=-1:1,lmax=4,Nr=16,basic_state_3d=ax)
    problem=Magrathea.setup_coupled_mode_problem(params)
    single=Magrathea.build_single_mode_operators(problem,false)
    op=coupling_test_op(1;bs=Magrathea.axisymmetric_basic_state(ax))
    A,B,interior,boundary=assemble_matrices(op)
    Ar,Br,_=Magrathea._constrained_reduced_matrices(A,B,op,interior,boundary)
    @test single[1].A ≈ Ar
    @test single[1].B ≈ Br
    @test single[-1].A ≈ conj(Ar)
    @test isempty(Magrathea.build_mode_coupling_operators(problem,single,false))
end
