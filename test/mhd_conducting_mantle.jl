module MHDConductingMantleTests

using Test, Magrathea, LinearAlgebra, SparseArrays, SpecialFunctions, Logging
const M=Magrathea
quiet(f)=with_logger(f,NullLogger())

function parameters(::Type{T}=Float64;kwargs...) where T
    defaults=(;E=T(.01),Pr=one(T),Pm=one(T),Ra=one(T),Le=T(.01),ricb=T(.35),
        m=0,lmax=2,N=20,symm=0,B0_type=axial,B0_amplitude=zero(T),
        bco_magnetic=1,mantle_radius=T(1.4),mantle_diffusivity_ratio=one(T))
    MHDParams(;merge(defaults,(;kwargs...))...)
end
operator(args...;kwargs...)=quiet(()->MHDStabilityOperator(parameters(args...;kwargs...)))

# Independent interpolation, with no production coefficient filtering.
function coefficients(f,N,ri,ro)
    x=cos.(pi.*(0:N)./N)
    V=[cos(n*acos(clamp(t,-1.,1.))) for t in x,n in 0:N]
    ComplexF64.(V\f.((ri+ro)/2 .+(ro-ri)/2 .*x))
end

# Remove homogeneous algebraic constraints before finding finite decay modes.
function decay_rates(A,B,Em;count=3)
    boundary=findall(i->iszero(B[i,:]),axes(B,1))
    interior=setdiff(axes(B,1),boundary)
    C=Matrix(A[boundary,:]);C ./= sqrt.(sum(abs2,C;dims=2))
    Z=nullspace(C)
    values=eigvals(Matrix(A[interior,:])*Z,Matrix(B[interior,:])*Z)
    @test all(real.(values).<0)
    # Only the slow resolved modes approximate the continuum spectrum; the
    # most strongly damped tau modes need not yet be accurate at this N.
    leading=sort(values;by=real,rev=true)[1:count]
    @test maximum(abs,imag.(leading))<1e-7
    sqrt.(-real.(leading)./Em)
end

sj(l,x)=sphericalbesselj(l,x)
sy(l,x)=sphericalbessely(l,x)
sjp(l,x)=l*sj(l,x)/x-sj(l+1,x)
syp(l,x)=l*sy(l,x)/x-sy(l+1,x)

# Piecewise spherical-Bessel solution in [.35,1] and [1,R]. This uses the
# physical interface conditions, independently of Chebyshev/tau assembly.
function shell_determinant(k,l,section,ri,R,ratio)
    km=k/sqrt(ratio)
    inner=section===:f ? [l*sj(l,k*ri)-k*ri*sjp(l,k*ri),l*sy(l,k*ri)-k*ri*syp(l,k*ri)] :
                        [sj(l,k*ri),sy(l,k*ri)]
    outer=section===:f ? [(l+1)*sj(l,km*R)+km*R*sjp(l,km*R),(l+1)*sy(l,km*R)+km*R*syp(l,km*R)] :
                        [sj(l,km*R),sy(l,km*R)]
    js,ys,jm,ym=sj(l,k),sy(l,k),sj(l,km),sy(l,km)
    ds,es,dm,em=k*sjp(l,k),k*syp(l,k),km*sjp(l,km),km*syp(l,km)
    interface=section===:f ? [ds,es,-dm,-em] : [ds+js,es+ys,-ratio*(dm+jm),-ratio*(em+ym)]
    det([inner[1] inner[2] 0 0;js ys -jm -ym;transpose(interface);0 0 outer[1] outer[2]])
end

function positive_roots(f;count=3)
    roots=Float64[];a=.05;fa=f(a)
    for b in .075:.025:40.
        fb=f(b)
        if signbit(fa)!=signbit(fb)
            lo=a;hi=b;flo=fa
            for _ in 1:55
                mid=(lo+hi)/2;fm=f(mid)
                if signbit(flo)==signbit(fm)
                    lo=mid;flo=fm
                else
                    hi=mid
                end
            end
            push!(roots,(lo+hi)/2)
            length(roots)==count && return roots
        end
        a=b;fa=fb
    end
    error("Failed to bracket independent magnetic decay roots")
end

@testset "Conducting mantle parameter contract" begin
    @test parameters().mantle_radius==1.4
    @test parameters().mantle_diffusivity_ratio==1.
    for radius in (0.,1.,-1.,Inf,NaN)
        @test_throws ArgumentError parameters(mantle_radius=radius)
    end
    for ratio in (0.,-1.,Inf,NaN)
        @test_throws ArgumentError parameters(mantle_diffusivity_ratio=ratio)
    end
    @test_throws ArgumentError MHDParams(E=.01,Ra=1.,ricb=.35,m=0,lmax=2,N=8,
        Le=.01,B0_type=axial,bco_magnetic=1)
end

@testset "Mantle artificial interface preserves full-sphere magnetic decay" begin
    # Equal material in core/fluid/mantle makes both internal interfaces
    # invisible. Vacuum-sphere roots are j₀(kR)=0 and j₁(kR)=0 for l=1.
    op=operator(bci_magnetic=1);p=op.params
    A,B,_,_=assemble_mhd_matrices(op);index=M._mhd_index_map(op)
    for (section,core,mantle,roots) in ((:f,:fi,:fm,[pi,2pi,3pi]),
            (:g,:gi,:gm,[4.493409457909064,7.725251836937707,10.904121659428899]))
        rows=vcat(collect(index[(1,section)]),collect(index[(1,core)]),collect(index[(1,mantle)]))
        @test decay_rates(A[rows,rows],B[rows,rows],p.Em) ≈ roots./p.mantle_radius rtol=2e-8
        k=first(roots)/p.mantle_radius;lambda=-p.Em*k^2
        x=zeros(ComplexF64,op.matrix_size)
        x[index[(1,section)]]=coefficients(r->sj(1,k*r),p.N,p.ricb,1.)
        x[index[(1,mantle)]]=coefficients(r->sj(1,k*r),p.N,1.,p.mantle_radius)
        # j₁(kr)/(r/ri) is entire in s=(r/ri)², including the centre.
        function regular_core(t)
            s=(t+1)/2;term=k*p.ricb/3;value=term
            for n in 1:30
                term *= -k^2*p.ricb^2*s/(2n*(2n+3));value+=term
            end
            value
        end
        x[index[(1,core)]]=coefficients(regular_core,p.N,-1.,1.)
        @test norm((A-lambda*B)[rows,:]*x)/norm(x)<2e-8
    end
end

@testset "Mantle magnetic decay agrees with independent layered Bessel roots" begin
    for ratio in (1.,3.)
        op=operator(mantle_diffusivity_ratio=ratio);p=op.params
        A,B,_,_=assemble_mhd_matrices(op);index=M._mhd_index_map(op)
        for (section,mantle) in ((:f,:fm),(:g,:gm))
            rows=vcat(collect(index[(1,section)]),collect(index[(1,mantle)]))
            expected=positive_roots(k->shell_determinant(k,1,section,p.ricb,p.mantle_radius,ratio))
            @test decay_rates(A[rows,rows],B[rows,rows],p.Em) ≈ expected rtol=2e-8
        end
    end
end

@testset "Mantle electric matching uses diffusivity and motional field" begin
    for ratio in (1.,3.),bg in (axial,dipole)
        op=operator(N=12,bco=0,B0_type=bg,mantle_diffusivity_ratio=ratio);p=op.params
        A,B,interior,_=assemble_mhd_matrices(op);index=M._mhd_index_map(op)
        rows=setdiff(vcat(collect(index[(2,:g)]),collect(index[(2,:gm)])),interior)
        x=zeros(ComplexF64,op.matrix_size)
        # With u=0, choose g(1)=gm(1)=1 and both insulating endpoint
        # values zero. Physical E_t continuity is ηf(g′+g)=ηm(gm′+gm).
        mantle_width=p.mantle_radius-1
        slope=ratio*(1-1/mantle_width)-1
        quadratic=-(1-slope*(1-p.ricb))/(1-p.ricb)^2
        x[index[(2,:g)]]=coefficients(r->1+slope*(r-1)+quadratic*(r-1)^2,p.N,p.ricb,1.)
        x[index[(2,:gm)]]=coefficients(r->(p.mantle_radius-r)/mantle_width,p.N,1.,p.mantle_radius)
        @test norm((A*x)[rows])<2e-10
        # Uφ(1,θ)=sinθ/(2√π) and B0r(1,θ)=cosθ. Hence the
        # mantle g₂₀′(1)=1/(3ηm) supplies the same tangential electric field.
        fill!(x,0)
        x[index[(1,:v)]]=coefficients(r->r-p.ricb*(r-1)^2/(1-p.ricb)^2,p.N,p.ricb,1.)
        c=1/(3p.Em*ratio)
        x[index[(2,:gm)]]=coefficients(r->c*(r-1)*(p.mantle_radius-r)/mantle_width,p.N,1.,p.mantle_radius)
        @test norm((A*x)[rows])<2e-10
        x[index[(1,:v)]].=0
        @test norm((A*x)[rows])>.1
    end
end

@testset "Mantle storage, DOF and distributed coefficient assembly" begin
    for T in (Float32,Float64),core in (0,1)
        op=operator(T;N=8,m=1,bci_magnetic=core);p=op.params
        index=M._mhd_index_map(op)
        @test p isa MHDParams{T}
        @test p.mantle_radius isa T
        expected=(length(op.ll_u)+length(op.ll_v)+length(op.ll_h)+
            (2+core)*(length(op.ll_f)+length(op.ll_g)))*(p.N+1)
        @test op.matrix_size==expected==M._mhd_total_dof(p)[1]
        ranges=sort(collect(values(index));by=first)
        @test reduce(vcat,collect.(ranges))==collect(1:expected)
        @test all(haskey(index,(l,:fm)) for l in op.ll_f)
        @test all(haskey(index,(l,:gm)) for l in op.ll_g)
        full=M._assemble_mhd_coo(op);n=full.n
        @test eltype(full.A_vals)==Complex{T}
        @test eltype(full.B_vals)==Complex{T}
        A=sparse(full.A_rows,full.A_cols,full.A_vals,n,n)
        B=sparse(full.B_rows,full.B_cols,full.B_vals,n,n)
        Ar=Int[];Ac=Int[];Av=Complex{T}[];Br=Int[];Bc=Int[];Bv=Complex{T}[]
        cuts=[0,fld(n,3),fld(2n,3),n]
        for part in 1:3
            owned=cuts[part]+1:cuts[part+1]
            localcoo=M._assemble_mhd_coo(op;owned_julia_rows=owned)
            @test all(in(owned),localcoo.A_rows)
            @test all(in(owned),localcoo.B_rows)
            append!(Ar,localcoo.A_rows);append!(Ac,localcoo.A_cols);append!(Av,localcoo.A_vals)
            append!(Br,localcoo.B_rows);append!(Bc,localcoo.B_cols);append!(Bv,localcoo.B_vals)
        end
        @test sparse(Ar,Ac,Av,n,n)==A
        @test sparse(Br,Bc,Bv,n,n)==B
        M.apply_velocity_boundary_conditions!(A,B,op)
        M.apply_temperature_boundary_conditions!(A,B,op)
        M.apply_magnetic_boundary_conditions!(A,B,op,:f)
        M.apply_magnetic_boundary_conditions!(A,B,op,:g)
        assembledA,assembledB,interior,_=assemble_mhd_matrices(op)
        @test A ≈ assembledA
        @test B == assembledB
        @test iszero(B[setdiff(1:n,interior),:])
    end
end

@testset "Public conducting-mantle solve and field reconstruction" begin
    op=operator(N=8);index=M._mhd_index_map(op);p=op.params
    x=zeros(ComplexF64,op.matrix_size)
    x[index[(2,:gm)]]=coefficients(r->r-1,p.N,1.,p.mantle_radius)
    Br,Btheta,Bphi,r,g=perturbation_magnetic(x,op;region=:mantle,Nr=5)
    @test first(r)≈1.
    @test last(r)≈p.mantle_radius
    @test iszero(Br)
    @test iszero(Btheta)
    @test Bphi ≈ 3(r.-1)*transpose(g.sinθ.*g.cosθ)./(2sqrt(pi)) atol=1e-11
    result=quiet(()->solve(MHDProblem(p);backend=:dense,nev=2,boundary_check=:none))
    @test length(result.eigenvalues)==2
    @test all(isfinite,result.eigenvalues)
    @test size(result.eigenvectors,1)==op.matrix_size
end

end
