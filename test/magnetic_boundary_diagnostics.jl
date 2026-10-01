module MagneticBoundaryDiagnosticChecks

using Test, Magrathea, LinearAlgebra, Logging
const M = Magrathea
quiet(f) = with_logger(f,NullLogger())
function operator(;kwargs...)
    p = MHDParams(;merge((;E=.01,Ra=1.,Le=.05,ricb=.35,m=0,lmax=3,N=8,
                           symm=0,B0_type=axial), (;kwargs...))...)
    quiet(() -> MHDStabilityOperator(p))
end
function empty_mode(op)
    zeros(Complex{typeof(op.params.E)},op.matrix_size), M._mhd_index_map(op)
end
function linear!(v,index,section,l;ri=.35,ro=1.,slope=1.,intercept=0.)
    block=index[(l,section)]
    v[first(block)]=intercept+slope*(ri+ro)/2
    v[first(block)+1]=slope*(ro-ri)/2
end

@testset "Independent uniform-field and vacuum surface norms" begin
    for T in (Float32,Float64)
        op=operator(E=T(.01),Ra=one(T),Le=T(.05),ricb=T(.35),Pr=one(T),Pm=one(T),
                    B0_amplitude=zero(T))
        x,index=empty_mode(op)
        # f_10=r and Y_10/sqrt(3)=cos(theta)/sqrt(4pi) give the
        # Cartesian constant field B=2ez/sqrt(4pi).
        linear!(x,index,:f,1;ri=op.params.ricb)
        d=magnetic_boundary_residuals(op,x;rtol=100eps(T))
        @test d.applicable && d.checked && !d.passed
        @test d.per_mode[1].inner.passed
        outer=d.per_mode[1].outer.tangential_field
        # The exterior vacuum field has Bt=+sin(theta)/sqrt(4pi),
        # whereas the fluid field has Bt=-2sin(theta)/sqrt(4pi).
        @test outer.residual ≈ sqrt(T(6)) rtol=100eps(T)
        @test outer.scale ≈ T(2) rtol=100eps(T)
        @test outer.relative_residual ≈ sqrt(T(6))/2 rtol=100eps(T)
        @test typeof(outer.residual) === T
        @test d.per_mode[1].outer.normal_field === nothing
        @test d.per_mode[1].mantle_outer === nothing
    end
end

@testset "Physical normalization, zero fields and malformed input" begin
    op=operator(bci_magnetic=2,bco_magnetic=2)
    x,index=empty_mode(op)
    linear!(x,index,:f,1)
    base=magnetic_boundary_residuals(op,x)
    # Uniform B has zero curl: the independent bulk scale avoids dividing
    # a tiny cancellation remainder by another tiny wall-current norm.
    for wall in (:inner,:outer)
        d=getproperty(base.per_mode[1],wall)
        @test d.tangential_electric.passed
        @test d.tangential_electric.relative_residual < 1e-12
        @test d.normal_field.relative_residual ≈ inv(sqrt(3)) atol=1e-12
    end
    for amplitude in (1e-150,1e150)
        d=magnetic_boundary_residuals(op,amplitude*x)
        for wall in (:inner,:outer), channel in (:normal_field,:tangential_electric)
            a=getproperty(getproperty(base.per_mode[1],wall),channel)
            b=getproperty(getproperty(d.per_mode[1],wall),channel)
            @test b.relative_residual ≈ a.relative_residual atol=1e-13
            @test b.scale ≈ amplitude*a.scale
        end
    end
    @test magnetic_boundary_residuals(op,x;rtol=0,atol=2).passed
    for options in ((;rtol=-1),(;rtol=Inf),(;atol=NaN),(;atol=-1),(;Nθ=2))
        @test_throws ArgumentError magnetic_boundary_residuals(op,x;options...)
    end
    @test_throws DimensionMismatch magnetic_boundary_residuals(op,x[1:end-1])
    bad=copy(x);bad[1]=NaN
    @test_throws ArgumentError magnetic_boundary_residuals(op,bad)
    for vectors in (zero(x),zeros(ComplexF64,0,2),zeros(ComplexF64,op.matrix_size,0))
        d=magnetic_boundary_residuals(op,vectors)
        @test !d.checked && !d.passed
    end
    thermal=zero(x);thermal[first(index[(1,:h)])]=1
    d=magnetic_boundary_residuals(op,thermal;rtol=0,atol=0)
    @test d.checked && d.passed
    @test iszero(d.maximum.inner.tangential_electric.relative_residual)
    weak=magnetic_boundary_residuals(op,thermal+1e-200x)
    @test weak.per_mode[1].outer.normal_field.relative_residual ≈ inv(sqrt(3)) rtol=1e-12
    @test weak.per_mode[1].inner.tangential_electric.passed
    columns=hcat(thermal,x,2x)
    d=magnetic_boundary_residuals(op,columns)
    @test length(d.per_mode)==3 && d.per_mode[1].passed && !d.per_mode[2].passed
    @test d.maximum.outer.normal_field.mode in (2,3)
    @test !d.maximum.outer.normal_field.passed
    for (residual,scale) in ((Inf,Inf),(0.,Inf),(NaN,1.),(1.,NaN))
        @test !M._mhd_boundary_metric(residual,scale,1.,1e-6,0.).passed
    end
end

@testset "Electric field outside retained angular degrees is checked" begin
    # m=0, parity -1, L=2 retains g_1 and v_2 but no g_3. Set v_2(1)=1
    # and Em*(g_1'+g_1)=-3/5: retained E_1 cancels exactly, leaving E_3.
    # In unnormalised Legendre derivatives, mu*dtheta(P2) =
    # (3/5)dtheta(P1) + (2/5)dtheta(P3).
    op=operator(symm=-1,lmax=2,bci=0,bco=0,bci_magnetic=2,bco_magnetic=2)
    x,index=empty_mode(op)
    linear!(x,index,:v,2)
    x[first(index[(1,:g)])]=-(3/5)/op.params.Em
    d=magnetic_boundary_residuals(op,x)
    residual=d.per_mode[1].outer.tangential_electric
    @test !residual.passed
    @test residual.residual ≈ (2/5)*sqrt(12/7) rtol=2e-13
    @test magnetic_boundary_residuals(op,x;Nθ=4).per_mode[1].outer.tangential_electric.residual ≈
        residual.residual rtol=2e-13
    # A higher quadrature order gives the same full residual, not a smaller
    # residual from projecting onto the solver's angular subspace.
    @test magnetic_boundary_residuals(op,x;Nθ=32).per_mode[1].outer.tangential_electric.residual ≈
        residual.residual rtol=2e-13
end

@testset "Conducting core continuity uses physical B and both E components" begin
    op=operator(bci_magnetic=1)
    x,index=empty_mode(op)
    # f=r, g=r are represented exactly by a constant core polynomial
    # multiplied by the regular prefactor r/ri.
    for (fluid,core) in ((:f,:fi),(:g,:gi))
        linear!(x,index,fluid,1)
        x[first(index[(1,core)])]=op.params.ricb
    end
    d=magnetic_boundary_residuals(op,x)
    @test d.per_mode[1].inner.passed
    @test d.per_mode[1].inner.tangential_electric.relative_residual < 1e-12
    x[first(index[(1,:fi)])]*=2
    wall=magnetic_boundary_residuals(op,x).per_mode[1].inner
    @test !wall.normal_field.passed && !wall.tangential_field.passed
    @test wall.normal_field.residual ≈ 2op.params.ricb/sqrt(3) rtol=1e-12
    @test wall.tangential_field.residual ≈ 2op.params.ricb*sqrt(2/3) rtol=1e-12

    # A quadratic poloidal potential has f=f'=0 at the interface, but
    # nonzero f''. Its toroidal electric field must still be checked.
    fill!(x,0)
    a=(1-op.params.ricb)/2
    x[index[(1,:f)][1:3]] .= a^2 .* [1.5,2.,.5]
    wall=magnetic_boundary_residuals(op,x).per_mode[1].inner
    @test wall.normal_field.passed && wall.tangential_field.passed
    @test !wall.tangential_electric.passed
    @test wall.tangential_electric.residual ≈ 2op.params.Em*op.params.ricb*sqrt(2/3) rtol=1e-12
end

@testset "Full motional electric field includes radial velocity" begin
    for background in (axial,dipole)
        op=operator(B0_type=background,bci_magnetic=2,bco_magnetic=2)
        x,index=empty_mode(op)
        # Constant Cartesian velocity 2ez/sqrt(4pi) is parallel to the
        # axial background, and gives Ephi=-3mu*sin(theta)/(sqrt(4pi)*r^3)
        # for a dipole. These fields deliberately need not satisfy mechanical
        # walls: a magnetic-only diagnostic must evaluate the actual u×B0.
        linear!(x,index,:u,1)
        d=magnetic_boundary_residuals(op,x)
        for (wall,r) in ((:inner,op.params.ricb),(:outer,1.))
            e=getproperty(d.per_mode[1],wall).tangential_electric
            if background==axial
                @test e.passed && e.relative_residual<1e-12
            else
                @test e.residual ≈ 3/r^2*sqrt(2/15) rtol=1e-12
                @test !e.passed
            end
        end
    end
end

@testset "Finite mantle interface and insulating exterior are separate checks" begin
    for ratio in (1.,2.)
        op=operator(bco_magnetic=1,mantle_radius=1.4,mantle_diffusivity_ratio=ratio)
        x,index=empty_mode(op)
        linear!(x,index,:g,1)
        # At r=1 g=gm=1 and eta*(g'+g)=eta_m*(gm'+gm).
        # eta_m/eta=2 therefore needs a constant gm; equal eta needs gm=r.
        linear!(x,index,:gm,1;ri=1.,ro=1.4,slope=2/ratio-1,intercept=2-2/ratio)
        d=magnetic_boundary_residuals(op,x)
        @test d.per_mode[1].outer.passed
        @test d.per_mode[1].outer.tangential_electric.relative_residual < 1e-12
        @test d.per_mode[1].mantle_outer !== nothing
        @test !d.per_mode[1].mantle_outer.tangential_field.passed
        @test d.maximum.mantle_outer.tangential_field.mode==1
        if ratio==2
            linear!(x,index,:gm,1;ri=1.,ro=1.4)
            wall=magnetic_boundary_residuals(op,x).per_mode[1].outer
            @test wall.normal_field.passed && wall.tangential_field.passed
            @test !wall.tangential_electric.passed
            @test wall.tangential_electric.residual ≈ 2op.params.Em*sqrt(2/3) rtol=1e-12
        end
    end
end

@testset "Weak magnetic wall residual decreases with radial resolution" begin
    residuals=Float64[]
    for N in (8,16)
        op=operator(m=1,lmax=1,symm=1,N=N,bci_magnetic=2,bco_magnetic=2)
        result=quiet(() -> solve(MHDProblem(op.params);backend=:dense,nev=1,sigma=-.1,
                                 boundary_check=:none))
        d=magnetic_boundary_residuals(op,result.eigenvectors)
        push!(residuals, max(d.maximum.inner.tangential_electric.relative_residual,
                            d.maximum.outer.tangential_electric.relative_residual))
    end
    @test residuals[2]<residuals[1]/100
    @test residuals[2]<1e-6
end

end # module
