using Test, Magrathea, LinearAlgebra

function manufactured_mean_flow(cd, p, t, L, M)
    Magrathea.SolenoidalMeanFlow(L,M,cd.x,p,t,
        Dict(k=>cd.D1*v for (k,v) in p),Dict(k=>cd.D2*v for (k,v) in p),
        Dict(k=>cd.D1*v for (k,v) in t))
end

@testset "Analytical nonlinear mean-flow inertia" begin
    cd=ChebyshevDiffn(18,[.35,1.],4); r=cd.x; N2=sqrt(5/(4π))
    # U=(-yz,xz,0): curl(U)=(-x,-y,2z), U×curl(U)=(2xz²,2yz²,z(x²+y²)).
    # Direct Legendre expansion gives R₂=2r³/(7N₂), S₂=-2r³/(21N₂).
    # Thus R₂-d(rS₂)/dr=2r³/(3N₂); the l=4 contribution is a gradient.
    p=Dict((2,0)=>zero(r)); t=Dict((2,0)=>-r.^3/(3N2))
    flow=manufactured_mean_flow(cd,p,t,5,0)
    force=Magrathea._mean_inertia(flow,cd.D1)
    @test force.p[(2,0)] ≈ 2r.^3/(3N2) rtol=1e-10
    @test force.radial[(2,0)] ≈ 2r.^3/(7N2) rtol=1e-12
    @test force.spheroidal[(2,0)] ≈ -2r.^3/(21N2) rtol=1e-12
    @test maximum(maximum(abs,v) for (k,v) in force.p if k!=(2,0)) < 2e-10
    @test maximum(maximum(abs,v) for v in values(force.t)) < 1e-12

    # Solid rotation has purely centrifugal (gradient) acceleration, removed
    # by the pressure projection. This checks radial metric factors as well.
    rotation=manufactured_mean_flow(cd,Dict((1,0)=>zero(r)),
        Dict((1,0)=>-sqrt(4π/3).*r.^2),4,0)
    gradient=Magrathea._mean_inertia(rotation,cd.D1)
    @test maximum(maximum(abs,v) for v in values(gradient.p)) < 2e-10
    @test maximum(maximum(abs,v) for v in values(gradient.t)) < 1e-12

    # Mixed cosine/sine poloidal and toroidal modes: nonlinear force is
    # quadratic and U·(U×curl(U)) vanishes at every radius after projection.
    profile=(r.-first(r)).^2 .* (last(r).-r).^2
    p=Dict((l,m)=>sin(l+2m).*profile for l in 1:3 for m in -l:l)
    t=Dict((l,m)=>cos(2l+m).*r.*profile for l in 1:3 for m in -l:l)
    flow=manufactured_mean_flow(cd,p,t,6,6)
    force=Magrathea._mean_inertia(flow,cd.D1)
    doubled=manufactured_mean_flow(cd,Dict(k=>2v for (k,v) in p),Dict(k=>2v for (k,v) in t),6,6)
    twice=Magrathea._mean_inertia(doubled,cd.D1)
    @test all(twice.p[k] ≈ 4v for (k,v) in force.p)
    @test all(twice.t[k] ≈ 4v for (k,v) in force.t)
    power=zero(r); scale=zero(r)
    for (k,v) in p
        q=k[1]*(k[1]+1)
        a=q.*v.*force.radial[k]
        b=q.*r.*flow.dp[k].*force.spheroidal[k]
        c=q.*r.*t[k].*force.t[k]
        power .+= a.+b.+c; scale .+= abs.(a).+abs.(b).+abs.(c)
    end
    @test maximum(abs,power) < 1e-12*maximum(scale)
end

@testset "Nonlinear 2D and 3D steady balances" begin
    cd=ChebyshevDiffn(24,[.35,1.],4); E=.1; Ra=100.; Pr=1.; tol=1e-9
    for m in (0,2), mechanical_bc in (:no_slip,:stress_free)
        M=m==0 ? 0 : 4
        bs,info=nonaxisymmetric_basic_state_selfconsistent(cd,.35,E,Ra,Pr,6,M,Dict((2,m)=>.1);
            mechanical_bc,tolerance=tol,max_iterations=60)
        @test info.converged && info.termination_reason===:converged
        @test info.momentum_model===:navier_stokes
        @test max(info.momentum_residual,info.thermal_residual,info.boundary_residual)<=tol
        @test all(diff(info.residual_history).<=0)
        @test all(0 .< info.step_history .<= .5)
        # Stokes flow at exactly the same final temperature does not satisfy
        # nonlinear momentum balance. Inertia measurably changes the velocity.
        stokes=Magrathea._steady_mean_flow(bs.theta_coeffs,cd.x,cd.D1,cd.D2,E,Ra,Pr,6,M;mechanical_bc)
        @test maximum(maximum(abs,v-stokes.p[k]) for (k,v) in bs.flow.p)>1e-8
        defect,_=Magrathea._mean_momentum_residual(bs.theta_coeffs,stokes,cd.D1,cd.D2,
            E,Ra,Pr,mechanical_bc,Magrathea._mean_inertia(stokes,cd.D1),Dict{Int,Any}())
        @test defect>1000tol
        if mechanical_bc===:stress_free
            @test abs(dot(Magrathea._mean_radial_weights(cd.x),cd.x.^2 .*bs.flow.t[(1,0)]))<1e-11
        end
        if m==2
            @test maximum(abs,bs.theta_coeffs[(4,4)])>1e-10
            @test maximum(abs,bs.flow.p[(4,4)])>1e-10
        end
    end
end

@testset "Nonlinear mean-state APIs and flux conditions" begin
    cd=ChebyshevDiffn(20,[.35,1.],4)
    for m in (0,2)
        flux=Y00(-1.)+Ylm(2,m,.1)
        bs,info=basic_state_selfconsistent(cd,.35,.1,30.,1.;flux_bc=flux,lmax_bs=4,tolerance=1e-9)
        @test info.converged
        @test bs isa (m==0 ? BasicState : BasicState3D)
        @test bs.flow.mmax==(m==0 ? 0 : 4)
        θ=bs isa BasicState ? Dict((l,0)=>v for (l,v) in bs.theta_coeffs) : bs.theta_coeffs
        for (k,v) in θ
            # Public BC uses unnormalised P_l^m; the scalar expansion includes
            # sqrt(2) for nonzero m, which is divided out of its coefficient.
            target=k==(0,0) ? -sqrt(4π) : k==(2,m) ? .1/sqrt((m==0 ? 1 : 2)*5/(4π)) : 0.
            @test abs((cd.D1*v)[end]-target)<1e-9
        end
    end
    build(;kw...)=nonaxisymmetric_basic_state_selfconsistent(cd,.35,.1,100.,1.,4,2,Dict((2,2)=>.1);kw...)
    _,info=build(max_iterations=1,tolerance=1e-14)
    @test !info.converged && info.termination_reason===:max_iterations
    @test isfinite(info.momentum_residual) && info.momentum_residual>1e-14
    @test_throws ArgumentError build(momentum_model=:invalid)
    @test_throws ArgumentError build(relaxation=0.)
    @test_throws ArgumentError build(tolerance=NaN)
    @test_throws ArgumentError build(max_iterations=0)
    @test_throws ArgumentError basic_state_selfconsistent(cd,.35,.1,30.,1.;flux_bc=Y22(.1),lmax_bs=4,mmax_bs=1)
    params=OnsetParams(E=.1,Pr=1.,Ra=100.,χ=.35,m=2,lmax=6,Nr=20)
    @test_throws ErrorException basic_state(params;mode=:selfconsistent,max_iterations=1,tol=1e-14)
    @test basic_state(params;mode=:selfconsistent,max_iterations=1,tol=1e-14,allow_unconverged=true) isa BasicState3D
end

# Independent physical-space validation (parameters E=.1, Ra=100, Pr=1).
function _nonlinear_cart_velocity(bs,x)
 r=norm(x); th=acos(x[3]/r); ph=atan(x[2],x[1]); v=mean_flow_velocity(bs,r,th,ph)
 er=x/r; et=[cos(th)*cos(ph),cos(th)*sin(ph),-sin(th)]; ep=[-sin(ph),cos(ph),0.]
 v.ur*er+v.utheta*et+v.uphi*ep
end
function _nonlinear_cart_temperature(bs,x)
 r=norm(x); mu=x[3]/r; phi=atan(x[2],x[1]); total=0.
 P=Dict(a=>Magrathea._associated_legendre_table(a,bs.lmax_bs,[mu]) for a in 0:bs.mmax_bs)
 for ((l,m),v) in bs.theta_coeffs
  phase=m==0 ? 1. : m>0 ? sqrt(2)*cos(m*phi) : sqrt(2)*sin(-m*phi)
  total+=Magrathea._mean_barycentric(bs.r,v,r)*sqrt((2l+1)/(4π))*P[abs(m)][l-abs(m)+1,1]*phase
 end
 total
end
function _nonlinear_cart_momentum(bs,x,h;inertia=true)
 v=_nonlinear_cart_velocity(bs,x); lap=zeros(3); grad=zeros(3,3)
 for a in 1:3
  e=zeros(3);e[a]=h
  pp=_nonlinear_cart_velocity(bs,x+2e);p=_nonlinear_cart_velocity(bs,x+e);m=_nonlinear_cart_velocity(bs,x-e);mm=_nonlinear_cart_velocity(bs,x-2e)
  lap+=(-pp+16p-30v+16m-mm)/(12h^2)
  grad[:,a]=(-pp+8p-8m+mm)/(12h)
 end
 (inertia ? grad*v : zero(v))+2cross([0.,0.,1.],v)-.1lap-(100*.1^2/.65^3)*_nonlinear_cart_temperature(bs,x)*x
end
function _nonlinear_cart_curl_momentum(bs;inertia=true)
 x=.63 .* [sin(1.1)*cos(.43),sin(1.1)*sin(.43),cos(1.1)]
 h=.001; df=zeros(3,3)
 for a in 1:3
  e=zeros(3);e[a]=h
  f(x)=_nonlinear_cart_momentum(bs,x,h;inertia)
  df[:,a]=(-f(x+2e)+8f(x+e)-8f(x-e)+f(x-2e))/(12h)
 end
 [df[3,2]-df[2,3],df[1,3]-df[3,1],df[2,1]-df[1,2]]
end

@testset "Independent nonlinear momentum balance and refinement" begin
    for m in (0,2)
        velocities=Vector{Float64}[]; curls=Float64[]
        for (N,L) in ((24,4),(24,6),(24,8),(32,8))
            M=m==0 ? 0 : L
            bs,info=nonaxisymmetric_basic_state_selfconsistent(
                ChebyshevDiffn(N,[.35,1.],4),.35,.1,100.,1.,L,M,Dict((2,m)=>.1);
                tolerance=1e-9,max_iterations=60)
            @test info.converged
            push!(velocities,collect(mean_flow_velocity(bs,.63,1.1,.43)))
            # Fourth-order Cartesian finite differences evaluate curl of
            # (U·∇)U + 2z×U - EΔU - βTx without using projected momentum code.
            push!(curls,norm(_nonlinear_cart_curl_momentum(bs)))
            if L==8
                without_inertia=norm(_nonlinear_cart_curl_momentum(bs;inertia=false))
                @test curls[end]<1e-4*without_inertia
                @test curls[end]<1e-7
            end
        end
        @test curls[2]<.02curls[1]
        @test curls[3]<.02curls[2]
        @test norm(velocities[3]-velocities[2])<.1norm(velocities[2]-velocities[1])
        @test norm(velocities[4]-velocities[3])<1e-8norm(velocities[4])
    end
end

@testset "Nonlinear cosine/sine rotation covariance" begin
    cd=ChebyshevDiffn(20,[.35,1.],4)
    cosine,ic=nonaxisymmetric_basic_state_selfconsistent(cd,.35,.1,100.,1.,4,4,
        Dict((2,2)=>.1);tolerance=1e-10)
    sine,is=nonaxisymmetric_basic_state_selfconsistent(cd,.35,.1,100.,1.,4,4,
        Dict((2,-2)=>.1);tolerance=1e-10)
    @test ic.converged && is.converged
    for (r,θ,φ) in ((.5,.7,.13),(.63,1.1,.43),(.82,2.1,.9))
        @test collect(mean_flow_velocity(cosine,r,θ,φ-π/4)) ≈
              collect(mean_flow_velocity(sine,r,θ,φ)) rtol=1e-8 atol=1e-11
    end
end
