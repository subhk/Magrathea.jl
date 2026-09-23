using Test, Magrathea, LinearAlgebra

# These replace the old tests of an inner-boundary-only first-order thermal
# wind equation. Check the physical fields and balances independently instead.
function meanflow_energy_balance(bs, cd, E, Ra, Pr)
    f=bs.flow; r=f.r; w=Magrathea._mean_radial_weights(r)
    theta=bs isa BasicState ? Dict((l,0)=>v for (l,v) in bs.theta_coeffs) : bs.theta_coeffs
    theta=Magrathea._sh_rescale(theta,+1)
    work=0.; dissipation=0.
    for (key,p) in f.p
        l,m=key; q=l*(l+1); t=f.t[key]
        lap=cd.D2*p-q.*p./r.^2
        work += Ra*E^2/(Pr*(r[end]-r[1])^3)*q*dot(w,r.*theta[key].*p)
        dissipation += E*dot(w,q^2 .*t.^2 ./r.^2 + q.*(f.dt[key].^2+lap.^2))
    end
    work,dissipation
end

function meanflow_divergence(bs,r,θ,φ)
    h=1e-5; v=mean_flow_velocity(bs,r,θ,φ)
    dr=(mean_flow_velocity(bs,r+h,θ,φ).ur-mean_flow_velocity(bs,r-h,θ,φ).ur)/(2h)
    dt=(mean_flow_velocity(bs,r,θ+h,φ).utheta-mean_flow_velocity(bs,r,θ-h,φ).utheta)/(2h)
    dp=(mean_flow_velocity(bs,r,θ,φ+h).uphi-mean_flow_velocity(bs,r,θ,φ-h).uphi)/(2h)
    dr+2v.ur/r+(dt+cot(θ)*v.utheta)/r+dp/(r*sin(θ))
end

@testset "Viscous mean-flow physical balances" begin
    cd=ChebyshevDiffn(24,[.35,1.],4); E=.01; Ra=100.; Pr=1.
    for m in (0,2), bc in (:no_slip,:stress_free)
        bs=nonaxisymmetric_basic_state(cd,.35,E,Ra,Pr,8,m,Dict((2,m)=>.01);mechanical_bc=bc)
        f=bs.flow; scale=maximum(norm(collect(mean_flow_velocity(bs,.63,θ,.43))) for θ in (.5,1.1,2.2))
        @test scale>0
        # Incompressibility in physical space, with independent finite differences.
        for (r,θ,φ) in ((.53,.7,.13),(.63,1.1,.43),(.82,2.1,.9))
            @test abs(meanflow_divergence(bs,r,θ,φ)) < 2e-7*scale
        end
        for r in (.35,1.), θ in (.4,1.1,2.4)
            v=mean_flow_velocity(bs,r,θ,.43)
            @test abs(v.ur)<1e-9*scale
            if bc==:no_slip
                @test norm(collect(v))<1e-8*scale
            end
        end
        if bc==:stress_free
            for (key,p) in f.p
                @test maximum(abs,(f.d2p[key]-2 .*f.dp[key]./f.r)[[1,end]])<1e-7*scale
                @test maximum(abs,(f.dt[key]-2 .*f.t[key]./f.r)[[1,end]])<1e-7*scale
            end
            if m==0
                @test abs(dot(Magrathea._mean_radial_weights(f.r),f.r.^2 .*f.t[(1,0)]))<1e-10*scale
            end
        else
            # Coriolis and pressure do no work. Buoyancy input equals viscous
            # dissipation, with the public gap-based Rayleigh number.
            work,diss=meanflow_energy_balance(bs,cd,E,Ra,Pr)
            @test work>0
            @test work ≈ diss rtol=2e-8
        end
        # Advection of constant temperature, including compatibility projections
        # (the legacy component form warns that it is approximate in general).
        oneT=Dict((0,0)=>fill(sqrt(4π),length(cd.x)))
        zeroT=Dict((0,0)=>zeros(length(cd.x)))
        adv=@test_logs (:warn,r"approximate") match_mode=:any compute_full_advection_spectral(
            oneT,zeroT,bs.ur_coeffs,bs.dur_dr_coeffs,bs.utheta_coeffs,bs.uphi_coeffs,
            bs.lmax_bs,bs.mmax_bs,bs.r)
        @test maximum(maximum(abs,v) for v in values(adv))==0
        @test maximum(maximum(abs,v) for v in values(compute_full_advection_spectral(oneT,zeroT,bs)))==0
        @test maximum(maximum(abs,v) for v in values(Magrathea._mean_flow_advection(oneT,zeroT,f)))==0
    end
end

@testset "Mean-flow phase, scaling, and resolution" begin
    cd=ChebyshevDiffn(24,[.35,1.],4)
    build(L,m=2;Ra=100.,amp=.01)=nonaxisymmetric_basic_state(cd,.35,.01,Ra,1.,L,abs(m),Dict((2,m)=>amp))
    cosine=build(8); sine=build(8,-2); doubled=build(8;Ra=200.)
    for (r,θ,φ) in ((.5,.7,.13),(.63,1.1,.43),(.82,2.1,.9))
        c=mean_flow_velocity(cosine,r,θ,φ-π/4)
        s=mean_flow_velocity(sine,r,θ,φ)
        @test collect(c) ≈ collect(s) rtol=1e-9 atol=1e-13
        @test collect(mean_flow_velocity(doubled,r,θ,φ)) ≈ 2 .*collect(mean_flow_velocity(cosine,r,θ,φ)) rtol=1e-10
    end
    for m in (0,2)
        states=[build(L,m) for L in (4,8,12)]
        u=[collect(mean_flow_velocity(bs,.63,1.1,.43)) for bs in states]
        @test norm(u[3]-u[2]) < .01*norm(u[2]-u[1])
        @test norm(u[3]-u[2]) < 2e-4*norm(u[3])
        fine=nonaxisymmetric_basic_state(ChebyshevDiffn(36,[.35,1.],4),.35,.01,100.,1.,12,m,Dict((2,m)=>.01))
        @test collect(mean_flow_velocity(fine,.63,1.1,.43)) ≈ u[3] rtol=1e-7
    end
    axis=meridional_basic_state(cd,.35,.01,100.,1.,8,.01)
    three=build(8,0)
    @test collect(mean_flow_velocity(axis,.63,1.1)) ≈ collect(mean_flow_velocity(three,.63,1.1)) rtol=1e-10
    @test maximum(abs,axis.ur_coeffs[2])>0
    @test all(all(iszero,v) for v in values(conduction_basic_state(cd,.35,4).uphi_coeffs))
end

@testset "Self-consistent thermal and momentum state" begin
    cd=ChebyshevDiffn(20,[.35,1.],4)
    for bc in (:fixed_temperature,:fixed_flux), m in (0,2)
        bs,info=nonaxisymmetric_basic_state_selfconsistent(cd,.35,.01,30.,1.,5,m,Dict((2,m)=>.01);
            thermal_bc=bc,max_iterations=35,tolerance=1e-9,momentum_model=:stokes)
        @test info.converged
        @test info.thermal_residual<1e-8
        # Returned velocity must use the final, rather than previous, temperature.
        fresh=Magrathea._steady_mean_flow(bs.theta_coeffs,cd.x,cd.D1,cd.D2,.01,30.,1.,5,m)
        @test all(bs.flow.p[k] ≈ fresh.p[k] for k in keys(fresh.p))
        @test all(bs.flow.t[k] ≈ fresh.t[k] for k in keys(fresh.t))
        for (key,v) in bs.theta_coeffs
            if key[2]<0
                @test abs(bc==:fixed_temperature ? v[end] : bs.dtheta_dr_coeffs[key][end])<1e-9
            end
        end
    end
    _,info=nonaxisymmetric_basic_state_selfconsistent(cd,.35,.01,100.,1.,4,2,Dict((2,2)=>.01);max_iterations=1,tolerance=1e-15)
    @test !info.converged
    bs,info=basic_state_selfconsistent(cd,.35,.01,30.,1.;temperature_bc=Y20(.01),max_iterations=35)
    @test bs isa BasicState
    @test info.converged
end

@testset "Legacy component wrappers use the viscous solver" begin
    cd=ChebyshevDiffn(20,[.35,1.],4); θ=Dict(2=>cd.x.^2)
    for m in (0,2)
        u=Dict{Int,Vector{Float64}}(); du=empty(u)
        Magrathea.solve_thermal_wind_coupled!(u,du,θ,m,cd,.35,1.,100.,1.;E=.01,lmax=6)
        @test maximum(maximum(abs,v) for v in values(u))>0
        @test maximum(maximum(abs,v[[1,end]]) for v in values(u))<1e-10
    end
end

function _mean_cartesian_velocity(bs,x)
 r=norm(x); th=acos(x[3]/r); ph=atan(x[2],x[1]); v=mean_flow_velocity(bs,r,th,ph)
 er=x/r; et=[cos(th)*cos(ph),cos(th)*sin(ph),-sin(th)]; ep=[-sin(ph),cos(ph),0.]
 v.ur*er+v.utheta*et+v.uphi*ep
end
function _mean_cartesian_momentum(bs,x,m,h)
 v=_mean_cartesian_velocity(bs,x); lap=zeros(3)
 for a in 1:3
  e=zeros(3);e[a]=h
  lap+=(-_mean_cartesian_velocity(bs,x+2e)+16_mean_cartesian_velocity(bs,x+e)-30v+16_mean_cartesian_velocity(bs,x-e)-_mean_cartesian_velocity(bs,x-2e))/(12h^2)
 end
 r=norm(x); angle=m==0 ? (3(x[3]/r)^2-1)/2 : 3(x[1]^2-x[2]^2)/r^2
 temp=.01*(r^2-.35^5/r^3)/(1-.35^5)*angle
 2cross([0.,0.,1.],v)-.01lap-(100*.01^2/.65^3)*temp*x
end
@testset "Independent Cartesian momentum curl" begin
for m in (0,2)
 L=12
 bs=nonaxisymmetric_basic_state(ChebyshevDiffn(28,[.35,1.],4),.35,.01,100.,1.,L,m,Dict((2,m)=>.01))
 x=.63 .* [sin(1.1)*cos(.43),sin(1.1)*sin(.43),cos(1.1)]
 h=.001; df=zeros(3,3)
 for a in 1:3
  e=zeros(3); e[a]=h
  df[:,a]=(-_mean_cartesian_momentum(bs,x+2e,m,h)+8_mean_cartesian_momentum(bs,x+e,m,h)-8_mean_cartesian_momentum(bs,x-e,m,h)+_mean_cartesian_momentum(bs,x-2e,m,h))/(12h)
 end
 curl=[df[3,2]-df[2,3],df[1,3]-df[3,1],df[2,1]-df[1,2]]
 @test norm(curl)/(100*.01^2/.65^3*.01) < 1e-5
end

end

@testset "Mean-flow evaluation at poles and on the old axisymmetric API" begin
    cd=ChebyshevDiffn(20,[.35,1.],4)
    bs=meridional_basic_state(cd,.35,.01,100.,1.,6,.01)
    @test mean_flow_velocity(bs,.63,0.).uphi == 0
    @test mean_flow_velocity(bs,.63,π).uphi == 0
    @test Magrathea.evaluate_basic_state(bs,.63,1.1).uphi_bar ≈ mean_flow_velocity(bs,.63,1.1).uphi
    @test all(iszero,mean_flow_velocity(conduction_basic_state(cd,.35,4),.63,1.1))
    three=nonaxisymmetric_basic_state(cd,.35,.01,100.,1.,6,1,Dict((2,1)=>.01))
    for θ in (0.,π)
        pole_vectors=map((0.,.7)) do φ
            v=mean_flow_velocity(three,.63,θ,φ)
            @test all(isfinite,v)
            v.ur.*[0.,0.,cos(θ)]+v.utheta.*[cos(θ)*cos(φ),cos(θ)*sin(φ),0.]+v.uphi.*[-sin(φ),cos(φ),0.]
        end
        @test pole_vectors[1] ≈ pole_vectors[2] rtol=1e-8 atol=1e-12
    end
end

@testset "Original mean-flow audit parameters remain bounded under refinement" begin
    cd=ChebyshevDiffn(32,[.35,1.],4)
    for m in (0,2)
        velocities=map((4,8,12)) do L
            bs=nonaxisymmetric_basic_state(cd,.35,1e-3,1e4,1.,L,m,Dict((2,m)=>.01))
            collect(mean_flow_velocity(bs,.63,1.1,.43))
        end
        @test maximum(norm,velocities)<10minimum(norm,velocities)
        @test norm(velocities[3]-velocities[2])<norm(velocities[2]-velocities[1])
    end
end
