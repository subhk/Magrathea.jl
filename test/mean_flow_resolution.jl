using Test, Magrathea, LinearAlgebra

# Polynomial native potentials give exact, independently integrable physical
# norms: for l=2, p=a*r^3 and t=b*r^2, the radial, horizontal-poloidal and
# toroidal squared norms are (36a²,54a²,6b²)*integral(r^4 dr).
function resolution_polynomial_state(N; a=1., b=2., anomaly=.3, background=10.,
                                     m=2, inner=.35, T=Float64)
    r=T.(ChebyshevDiffn(N,[inner,1.],2).x)
    key=(2,m)
    p=Dict(key=>T(a).*r.^3); t=Dict(key=>T(b).*r.^2)
    flow=Magrathea.SolenoidalMeanFlow(2,abs(m),r,p,t,
        Dict(key=>3T(a).*r.^2),Dict(key=>6T(a).*r),Dict(key=>2T(b).*r))
    # Public no-factorial coefficients have norm sqrt((l+|m|)!/(l-|m|)!).
    normalization=sqrt(T(factorial(2+abs(m))/factorial(2-abs(m))))
    theta=Dict((0,0)=>fill(T(background)*sqrt(4T(π)),N),
               key=>T(anomaly).*r./normalization)
    z=Dict(k=>zero(r) for k in keys(theta))
    BasicState3D(lmax_bs=2,mmax_bs=abs(m),Nr=N,r=r,theta_coeffs=theta,
        dtheta_dr_coeffs=z,ur_coeffs=z,utheta_coeffs=z,uphi_coeffs=z,
        dur_dr_coeffs=z,dutheta_dr_coeffs=z,duphi_dr_coeffs=z,flow=flow)
end

@testset "Physical mean-flow resolution norms" begin
    coarse=resolution_polynomial_state(12; a=.5,b=1.,anomaly=.15)
    fine=resolution_polynomial_state(20)
    report=mean_flow_resolution(coarse,fine;rtol=.01)
    integral=(1-.35^5)/5
    for (name,value) in ((:radial,36.),(:poloidal,54.),(:toroidal,24.),(:total,114.))
        metric=getproperty(report.velocity,name)
        @test metric.scale ≈ sqrt(value*integral) rtol=1e-12
        @test metric.error ≈ .5sqrt(value*integral) rtol=1e-12
        @test metric.relative_error ≈ .5 rtol=1e-12
        @test !metric.converged
    end
    @test report.temperature.anomaly.scale ≈ .3sqrt(integral) rtol=1e-12
    @test report.temperature.anomaly.relative_error ≈ .5 rtol=1e-12
    @test report.temperature.total.scale ≈ sqrt(400π*(1-.35^3)/3+.09integral) rtol=1e-12
    @test !report.converged

    # Different radial grids representing the same field must compare equal.
    identical=mean_flow_resolution(resolution_polynomial_state(10),fine;rtol=1e-11)
    @test identical.converged
    @test identical.velocity.total.relative_error < 1e-12
    @test mean_flow_resolution(fine,fine;rtol=0,atol=0).converged

    # Sine/cosine phases are orthogonal; an absent mode must count as zero.
    sine=resolution_polynomial_state(18;m=-2)
    orthogonal=mean_flow_resolution(sine,fine)
    @test orthogonal.velocity.total.relative_error ≈ sqrt(2) rtol=1e-12
    @test orthogonal.temperature.anomaly.relative_error ≈ sqrt(2) rtol=1e-12
    @test !orthogonal.converged

    axis3d=resolution_polynomial_state(16;m=0)
    axis=Magrathea._axisymmetric_state(axis3d)
    @test mean_flow_resolution(axis,axis3d;rtol=0,atol=0).converged
end

@testset "Small channels cannot be hidden by dominant fields" begin
    fine=resolution_polynomial_state(20;a=2.,b=1e6,anomaly=.6,background=1e6)
    coarse=resolution_polynomial_state(12;a=1.,b=1e6,anomaly=.3,background=1e6)
    report=mean_flow_resolution(coarse,fine;rtol=1e-3)
    @test report.velocity.total.converged
    @test report.velocity.toroidal.converged
    @test report.temperature.total.converged
    @test !report.velocity.radial.converged
    @test !report.velocity.poloidal.converged
    @test !report.temperature.anomaly.converged
    @test !report.converged

    zero_state=resolution_polynomial_state(12;a=0.,b=0.,anomaly=0.,background=0.)
    @test mean_flow_resolution(zero_state,zero_state).converged
    perturbation=resolution_polynomial_state(20;a=1e-12,b=1e-12,anomaly=1e-12,background=0.)
    @test !mean_flow_resolution(perturbation,zero_state;atol=0).converged
    @test mean_flow_resolution(perturbation,zero_state;atol=1e-10).converged
    @test isinf(mean_flow_resolution(perturbation,zero_state).velocity.total.relative_error)
end

@testset "Mean-flow resolution validation and precision" begin
    a=resolution_polynomial_state(12)
    @test_throws ArgumentError mean_flow_resolution(a,a;rtol=-1.)
    @test_throws ArgumentError mean_flow_resolution(a,a;atol=NaN)
    @test_throws ArgumentError mean_flow_resolution(a,resolution_polynomial_state(12;inner=.4))
    invalid=resolution_polynomial_state(12)
    invalid.flow.p[(2,2)][3]=NaN
    @test_throws ArgumentError mean_flow_resolution(invalid,a)
    legacy=BasicState(lmax_bs=2,Nr=12,r=a.r,
        theta_coeffs=Dict(0=>ones(12)),dtheta_dr_coeffs=Dict(0=>zeros(12)),
        uphi_coeffs=Dict(2=>ones(12)),duphi_dr_coeffs=Dict(2=>zeros(12)))
    @test_throws ArgumentError mean_flow_resolution(legacy,legacy)
    for T in (Float32,BigFloat)
        coarse=resolution_polynomial_state(12;T)
        fine=resolution_polynomial_state(20;T)
        result=mean_flow_resolution(coarse,fine;rtol=1e-5)
        @test result.converged
        @test result.velocity.total.error isa T
    end
    conduction=conduction_basic_state(ChebyshevDiffn(16,[.35,1.],2),.35,2)
    finer_conduction=conduction_basic_state(ChebyshevDiffn(24,[.35,1.],2),.35,4)
    comparison=mean_flow_resolution(conduction,finer_conduction;rtol=1e-5)
    @test comparison.converged
    @test comparison.velocity.total.error == 0
end

@testset "Resolution check detects under-resolved rotating jet" begin
    cd=ChebyshevDiffn(32,[.35,1.],4)
    coarse=meridional_basic_state(cd,.35,1e-5,1e7,1.,4,.1)
    fine=meridional_basic_state(cd,.35,1e-5,1e7,1.,8,.1)
    report=mean_flow_resolution(coarse,fine;rtol=.01)
    @test !report.converged
    @test report.velocity.total.relative_error > .1
    @test !report.velocity.radial.converged
end

include(joinpath(@__DIR__,"..","example","mean_flow_resolution.jl"))

@testset "Example refinement requires both directions and completed iterations" begin
    helper=MeanFlowExampleResolution.refine_mean_flow
    visited=Tuple{Int,Int}[]
    function builder(N,L)
        push!(visited,(N,L))
        # Both directions initially fail. Only the second refinement interval
        # represents the same polynomial field along both axes.
        resolution_polynomial_state(N;a=N<16 ? 2. : 1.,b=L<8 ? 2. : 1.)
    end
    report=redirect_stdout(devnull) do
        helper(builder;radial_levels=(12,16,20),angular_levels=(4,8,12),rtol=1e-9)
    end
    @test report.Nr==20 && report.lmax==12
    @test report.radial.converged && report.angular.converged
    @test (16,12) in visited && (20,8) in visited && (20,12) in visited
    @test length(visited)==length(unique(visited))

    redirect_stdout(devnull) do
        @test_throws ErrorException helper(builder;radial_levels=(12,16),angular_levels=(4,8),rtol=1e-9)
        @test_throws ErrorException helper(builder;radial_levels=(12,),angular_levels=(4,8))
        # Even identical fields cannot pass if the nonlinear solve failed.
        info=(converged=false,termination_reason=:max_iterations,
              residual_history=[1.],iterations=1)
        failed(N,L)=(resolution_polynomial_state(N),info)
        @test_throws ErrorException helper(failed;radial_levels=(12,16),angular_levels=(4,8))
    end
end
