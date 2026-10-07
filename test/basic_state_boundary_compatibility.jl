module BasicStateBoundaryCompatibility

using Test, Magrathea, LinearAlgebra

check(bc,bs)=Magrathea._check_basic_state_mechanical_bc(bc,bs)

function native_state(N,::Type{T};p=r->zero(r),t=r->zero(r)) where T
    cd=ChebyshevDiffn(N,T[.35,1],4);r=cd.x;k=(2,0)
    pv=T.(p.(r));tv=T.(t.(r))
    flow=Magrathea.SolenoidalMeanFlow(2,0,r,Dict(k=>pv),Dict(k=>tv),
        Dict(k=>cd.D1*pv),Dict(k=>cd.D2*pv),Dict(k=>cd.D1*tv))
    z=Dict(0=>zero(r))
    BasicState(lmax_bs=2,Nr=N,r=r,theta_coeffs=z,dtheta_dr_coeffs=z,
        uphi_coeffs=z,duphi_dr_coeffs=z,flow=flow)
end

function component_state(;radial=r->0.,azimuthal=r->0.,fake_derivative=false)
    cd=ChebyshevDiffn(20,[.35,1.],4);r=cd.x;v=azimuthal.(r)
    z=Dict(0=>zero(r))
    BasicState(lmax_bs=2,Nr=length(r),r=r,theta_coeffs=z,dtheta_dr_coeffs=z,
        ur_coeffs=Dict(2=>radial.(r)),uphi_coeffs=Dict(2=>v),
        duphi_dr_coeffs=fake_derivative ? Dict(2=>v./r) : Dict{Int,Vector{Float64}}())
end

@testset "Native mean velocity must match stationary mechanical walls" begin
    @test check(:no_slip,nothing)===nothing
    for T in (Float64,Float32), amplitude in (one(T),T(1e-20))
        # t/r ∝ r satisfies ∂r(u_t)-u_t/r=0 at both walls, but moves them.
        rotation=native_state(20,T;t=r->amplitude*r^2)
        @test check(:stress_free,rotation)===nothing
        @test_throws ArgumentError check(:no_slip,rotation)

        # This field vanishes on both walls but has nonzero tangential strain.
        stopped=native_state(20,T;t=r->amplitude*(r-T(.35))*(one(T)-r))
        @test check(:no_slip,stopped)===nothing
        @test_throws ArgumentError check(:stress_free,stopped)
    end
    # No absolute floor: a very small nonzero incompatible field still fails.
    @test_throws ArgumentError check(:no_slip,native_state(12,Float64;t=r->1e-200*r^2))

    # A large tangential channel cannot hide radial penetration.
    penetrating=native_state(16,Float64;p=r->1e-12*r^2,t=r->1e12*r^2)
    @test_throws ArgumentError check(:stress_free,penetrating)
    @test_throws ArgumentError check(:no_slip,penetrating)

    # Conditions are checked from the field, independently of provenance.
    both=native_state(16,Float64;t=r->(r-.35)^2*(1-r)^2)
    @test check(:no_slip,both)===nothing
    @test check(:stress_free,both)===nothing
    @test check((:no_slip,:stress_free),both)===nothing
    zero_state=native_state(12,Float64)
    @test check(:no_slip,zero_state)===nothing
    @test check(:stress_free,zero_state)===nothing

    # Differentiation in double precision must not receive a Float32 N²
    # allowance that masks the real strain of this low-degree polynomial.
    for (N,delta) in ((32,.01f0),(128,.1f0),(256,1f0))
        strained=native_state(N,Float32;t=r->r^2*(1+delta*(r-.35f0)))
        @test_throws ArgumentError check(:stress_free,strained)
    end
end

@testset "Wall derivatives match the physical radial evaluator" begin
    # Off the constructor's CGL grid the evaluator uses the same fixed
    # barycentric weights, so its interpolant need not equal the polynomial.
    custom=native_state(12,Float64)
    custom.r .= range(.35,1.;length=12)
    values=custom.r.^2
    custom.flow.t[(2,0)] .= values
    grid=Magrathea._resolution_grid(custom.r,Float64)
    D=Magrathea._mean_bc_derivative_rows(grid)
    for (wall,i,direction) in ((1,1,1),(2,length(custom.r),-1))
        h=1e-6
        f=[Magrathea._mean_barycentric(custom.r,values,custom.r[i]+direction*j*h) for j in 0:3]
        derivative=direction*(-11f[1]+18f[2]-9f[3]+2f[4])/(6h)
        @test dot(D[wall,:],values) ≈ derivative rtol=1e-7 atol=1e-8
    end
    # General polynomial weights would incorrectly accept t=r² as strain-free.
    @test_throws ArgumentError check(:stress_free,custom)
end

@testset "Constructed fields tolerate numerical wall roundoff" begin
    # High radial order amplifies derivative roundoff; valid double-precision
    # states must pass without accepting a genuinely different wall condition.
    cd=ChebyshevDiffn(64,[.35,1.],4)
    for bc in (:no_slip,:stress_free),m in (0,2)
        bs=nonaxisymmetric_basic_state(cd,.35,.01,100.,1.,6,m,Dict((2,m)=>.1);mechanical_bc=bc)
        @test check(bc,bs)===nothing
        @test_throws ArgumentError check(bc===:no_slip ? :stress_free : :no_slip,bs)
    end
end

@testset "Legacy component states use their represented velocity" begin
    stress_free=component_state(azimuthal=r->r)
    @test check(:stress_free,stress_free)===nothing
    @test_throws ArgumentError check(:no_slip,stress_free)
    no_slip=component_state(azimuthal=r->(r-.35)*(1-r),fake_derivative=true)
    @test check(:no_slip,no_slip)===nothing
    # A supplied derivative manufactured to satisfy the stress condition must
    # not conceal the incompatible actual radial profile.
    @test_throws ArgumentError check(:stress_free,no_slip)
    @test_throws ArgumentError check(:stress_free,component_state(radial=r->1e-12,azimuthal=r->1e12*r))
end

@testset "Mechanical compatibility rejects malformed states" begin
    @test_throws ArgumentError check(:invalid,native_state(12,Float64))
    bad=component_state(azimuthal=r->NaN)
    @test_throws ArgumentError check(:no_slip,bad)
    bad_length=component_state()
    resize!(bad_length.uphi_coeffs[2],1)
    @test_throws DimensionMismatch check(:no_slip,bad_length)

    # A nearly coincident custom grid cannot buy compatibility by making the
    # derivative uncertainty arbitrarily large, even for smooth velocity data.
    unresolved=native_state(4,Float64)
    unresolved.r .= [.35,.35+1e-14,1-1e-14,1.]
    unresolved.flow.t[(2,0)] .= unresolved.r.^2
    @test_throws ArgumentError check(:stress_free,unresolved)
end

end
