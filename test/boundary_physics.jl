module BoundaryPhysicsChecks
using Test, Magrathea, LinearAlgebra, SparseArrays, Logging
# Evaluate a Chebyshev series and its derivatives by differentiating the
# three-term polynomial recurrence (independent of production endpoint rows).
function jet(c, r; ri=.35, ro=1.)
    x=(2r-ri-ro)/(ro-ri); a=(1.,0.,0.); b=(x,1.,0.)
    v=c[1]*a[1]; d=zero(v); dd=zero(v)
    if length(c)>1
        v+=c[2]*b[1]; d+=c[2]
    end
    for n in 2:length(c)-1
        q=(2x*b[1]-a[1],2b[1]+2x*b[2]-a[2],4b[2]+2x*b[3]-a[3])
        v+=c[n+1]*q[1];d+=c[n+1]*q[2];dd+=c[n+1]*q[3]
        a,b=b,q
    end
    (v,2d/(ro-ri),4dd/(ro-ri)^2)
end
function corejet(c,l,ri)
    a,d,dd=jet(c,1.;ri=-1.,ro=1.)
    a,(l*a+4d)/ri,(l*(l-1)*a+(8l+4)*d+16dd)/ri^2
end
function harmonic(l,m,μ)
    s=sqrt(1-μ^2); prev=0.; curr=(-1)^m*prod(1:2:2m-1)*s^m
    for k in m+1:l
        prev,curr=curr,((2k-1)*μ*curr-(k+m-1)*prev)/(k-m)
    end
    f=sqrt(Float64(factorial(big(l-m))/factorial(big(l+m)))/(4π))
    y=f*curr; h=f*(l*μ*curr-(l+m)*prev)/s; v=im*m*y/s
    y,h,v
end
function boundary_fields(op,x,r; core=false)
    p=op.params;idx=Magrathea._mhd_index_map(op)
    n=2p.lmax+6;gw=eigen(SymTridiagonal(zeros(n),[i/sqrt(4i^2-1) for i in 1:n-1]));μ=gw.values;w=2gw.vectors[1,:].^2
    U=zeros(ComplexF64,n,3);B=zero(U);J=zero(U)
    sections=core ? ((:fi,:p,op.ll_f),(:gi,:t,op.ll_g)) : ((:f,:p,op.ll_f),(:g,:t,op.ll_g),(:u,:up,op.ll_u),(:v,:ut,op.ll_v))
    for (section,kind,ls) in sections, l in ls
        coeff=x[idx[(l,section)]]
        f,d,dd=core ? corejet(coeff,l,p.ricb) : jet(coeff,r;ri=p.ricb)
        q=l*(l+1);lap=dd+2d/r-q*f/r^2
        for j in 1:n
            y,h,v=harmonic(l,p.m,μ[j])
            if kind in (:p,:up)
                target=kind==:p ? B : U
                target[j,1]+=q*f/r*y;target[j,2]+=(d+f/r)*h;target[j,3]+=(d+f/r)*v
                if kind==:p
                    J[j,2]-=lap*v;J[j,3]+=lap*h
                end
            else
                target=kind==:t ? B : U
                target[j,2]+=f*v;target[j,3]-=f*h
                if kind==:t
                    J[j,1]+=q*f/r*y;J[j,2]+=(d+f/r)*h;J[j,3]+=(d+f/r)*v
                end
            end
        end
    end
    br=p.B0_type==dipole ? μ/r^3 : μ
    bt=p.B0_type==dipole ? sqrt.(1 .-μ.^2)/(2r^3) : -sqrt.(1 .-μ.^2)
    E=hcat(p.Em.*J[:,2].-U[:,3].*br, p.Em.*J[:,3].-U[:,1].*bt.+U[:,2].*br)
    (U=U,B=B,J=J,E=E,μ=μ,w=w)
end
function project_E(E,d,op)
    values=ComplexF64[]
    for (ls,tor) in ((op.ll_g,false),(op.ll_f,true)),l in ls
        h=[harmonic(l,op.params.m,μ)[2] for μ in d.μ]
        v=[harmonic(l,op.params.m,μ)[3] for μ in d.μ]
        tor && ((h,v)=(-v,h))
        push!(values,2π*(2l+1)/(l*(l+1))*sum(d.w.*(conj.(h).*E[:,1].+conj.(v).*E[:,2])))
    end
    norm(values)
end
function eigenmodes(op)
    A,B,_,_=assemble_mhd_matrices(op)
    bc=findall(i->iszero(B[i,:]),axes(B,1));interior=setdiff(axes(B,1),bc)
    C=Matrix(A[bc,:]);C./=maximum(abs,C;dims=2)
    R=nullspace(C);e=eigen(Matrix(A[interior,:])*R,Matrix(B[interior,:])*R)
    ind=sortperm(abs.(e.values.+.1))[1:3]
    e.values[ind],R*e.vectors[:,ind]
end

function coeff(r,values;ri=minimum(r),ro=maximum(r))
    x=(2 .*r.-ri.-ro)./(ro-ri)
    V=[cos(k*acos(clamp(t,-1.,1.))) for t in x,k in 0:length(r)-1]
    V\values
end
function op_for(;kwargs...)
    defaults=(;E=.01,Pr=1.,Pm=1.,Ra=1.,Le=.05,ricb=.35,m=1,lmax=3,N=12,symm=1,B0_type=axial)
    MHDStabilityOperator(MHDParams(;merge(defaults,(;kwargs...))...))
end

@testset "Analytical wall EMF agrees with independent vector projection" begin
    for T in (Float32,Float64),bg in (axial,dipole),m in (0,1,3)
        op=op_for(E=T(.01),Pr=one(T),Pm=one(T),Ra=one(T),Le=T(.05),B0_amplitude=zero(T),
                  ricb=T(.1),B0_type=bg,m=m,lmax=8,N=8,symm=0)
        n=32;gw=eigen(SymTridiagonal(zeros(n),[i/sqrt(4i^2-1) for i in 1:n-1]));μ=gw.values;w=2gw.vectors[1,:].^2
        err=0.;forbidden=true
        for s in (:u,:v),lo in max(1,m):8,li in max(1,m):8
            h=[harmonic(lo,m,z)[2] for z in μ];v=[harmonic(lo,m,z)[3] for z in μ]
            hi=[harmonic(li,m,z)[2] for z in μ];vi=[harmonic(li,m,z)[3] for z in μ]
            θ,φ=s==:u ? (-vi,hi) : (hi,vi)
            br=bg==dipole ? μ/Float64(op.params.ricb)^3 : μ
            reference=2π*(2lo+1)/(lo*(lo+1))*sum(w.*br.*(conj.(h).*θ.+conj.(v).*φ))
            c=Magrathea._mhd_wall_emf(op,lo,li,s,op.params.ricb)
            err=max(err,abs(c-reference)/max(1.,abs(reference)))
            allowed=s==:u ? lo==li : abs(lo-li)==1
            allowed || (forbidden &= iszero(c))
        end
        @test err < (T==Float32 ? 5e-7 : 2e-10)
        @test forbidden
    end
end

@testset "Mechanical and thermal wall conditions in native potential conventions" begin
    # Independent polynomial jets -> radial velocity, tangential velocity and
    # spherical strain. Both collocation and coefficient representations are tested.
    for ri in (.2,.55),inner in (0,1),outer in (0,1),ti in (0,1),to in (0,1)
        op=op_for(ricb=ri,bci=inner,bco=outer,bci_thermal=ti,bco_thermal=to)
        rows,entries=Magrathea._compute_mhd_bc(op);index=Magrathea._mhd_index_map(op)
        C=sparse(first.(entries),getindex.(entries,2),last.(entries),op.matrix_size,op.matrix_size)
        _,_,layout=Magrathea.assemble_mhd_galerkin(op)
        for (section,ls) in ((:u,op.ll_u),(:v,op.ll_v),(:h,op.ll_h)),l in ls
            b=index[(l,section)];bc=sort!(collect(intersect(rows,Set(b))))
            F=Matrix(C[bc,b]); F./=maximum(abs,F;dims=2)
            for R in (nullspace(F),layout.R[(section,l)])
                errors=Float64[]
                for (r,mechanical,thermal) in ((ri,inner,ti),(1.,outer,to)),c in eachcol(R)
                    f,d,dd=jet(c,r;ri=ri);q=l*(l+1)
                    if section==:u
                        push!(errors,abs(q*f/r))
                        push!(errors,abs(mechanical==1 ? d+f/r : dd+(q-2)*f/r^2))
                    elseif section==:v
                        push!(errors,abs(mechanical==1 ? f : d-f/r))
                    else
                        push!(errors,abs(thermal==0 ? f : d))
                    end
                end
                @test maximum(errors)<2e-9
            end
        end
    end
    for bc in (:no_slip,:stress_free),thermal in (:fixed_temperature,:fixed_flux)
        p=OnsetParams(E=.01,Pr=1.,Ra=1.,χ=.35,m=1,lmax=3,Nr=14,
                      mechanical_bc=bc,thermal_bc=thermal)
        op=LinearStabilityOperator(p)
        for (l,field) in keys(op.index_map)
            F=Magrathea._constraint_subblock(op,l,field);F./=maximum(abs,F;dims=2)
            R=nullspace(F);errors=Float64[]
            for v in eachcol(R),r in (.35,1.)
                f,d,dd=jet(coeff(op.r,v),r)
                if field==:P
                    push!(errors,abs(l*(l+1)*f/r),abs(bc==:no_slip ? d+f/r : dd+(l*(l+1)-2)*f/r^2))
                elseif field==:T
                    push!(errors,abs(bc==:no_slip ? f : d-f/r))
                else
                    push!(errors,abs(thermal==:fixed_temperature ? f : d))
                end
            end
            @test maximum(errors)<2e-8
        end
    end
end

@testset "Mean-flow walls in 2D and 3D from independent radial interpolation" begin
    cd=ChebyshevDiffn(24,[.35,1.],4)
    for m in (0,2),mechanical in (:no_slip,:stress_free)
        bs=nonaxisymmetric_basic_state(cd,.35,.01,100.,1.,6,m,Dict((2,m)=>.01);mechanical_bc=mechanical)
        flow=bs.flow;errors=Float64[];scale=maximum(norm(v) for v in values(flow.p))+maximum(norm(v) for v in values(flow.t))
        for (key,p) in flow.p,r in (.35,1.)
            l=key[1];P,dP,ddP=jet(coeff(flow.r,p),r)
            T,dT,_=jet(coeff(flow.r,flow.t[key]),r)
            # Unit-vector potentials in mean flow: ur=qP/r², uS=P'/r, uT=T/r.
            push!(errors,abs(l*(l+1)*P/r^2))
            if mechanical==:no_slip
                push!(errors,abs(dP/r),abs(T/r))
            else
                push!(errors,abs(ddP/r-2dP/r^2+l*(l+1)*P/r^3),abs(dT/r-2T/r^2))
            end
        end
        @test maximum(errors)/scale<2e-8
    end
end

@testset "Manufactured thermal boundary values and radial flux signs" begin
    # A smooth polynomial solves Δ_l T=F. Test all mixed/flux combinations,
    # both radial grid orderings, and both boundaries against its exact values.
    exact(r)=r^3+.2r^2-.3r+.7
    deriv(r)=3r^2+.4r-.3
    cd=ChebyshevDiffn(18,[.35,1.],2)
    for reverse_grid in (false,true),l in (0,2),inner in (:fixed_temperature,:fixed_flux),outer in (:fixed_temperature,:fixed_flux)
        perm=reverse_grid ? (18:-1:1) : (1:18)
        r=cd.x[perm];D1=Matrix(cd.D1[perm,perm]);D2=Matrix(cd.D2[perm,perm])
        rhs = 6 .* r .+ .4 .+ 2 .* deriv.(r) ./ r .- l*(l+1) .* exact.(r) ./ r.^2
        iv=inner==:fixed_temperature ? exact(.35) : deriv(.35)
        ov=outer==:fixed_temperature ? exact(1.) : deriv(1.)
        if l==0 && inner==outer==:fixed_flux
            @test_throws ArgumentError Magrathea.solve_poisson_mode(l,0,r,D2,D1,.35,1.,rhs;inner_bc=inner,outer_bc=outer,inner_value=iv,outer_value=ov)
        else
            t,dt=Magrathea.solve_poisson_mode(l,0,r,D2,D1,.35,1.,rhs;inner_bc=inner,outer_bc=outer,inner_value=iv,outer_value=ov)
            @test norm(t-exact.(r),Inf)<2e-10
            @test norm(dt-deriv.(r),Inf)<2e-9
        end
    end
    for side in (:inner_bc,:outer_bc)
        @test_throws ArgumentError Magrathea.solve_poisson_mode(2,0,cd.x,Matrix(cd.D2),Matrix(cd.D1),.35,1.,zero(cd.x);(;side=>:invalid)...)
    end
end

@testset "Magnetic insulating matching and unsupported recombination" begin
    for l in (1,3,7),r in (.35,1.)
        # An external vacuum potential decays as r^(-l-1), the cavity field
        # grows as r^l in the code's radius-vector potential convention.
        f,d = r==1. ? (r^(-l-1),-(l+1)*r^(-l-2)) : (r^l,l*r^(l-1))
        @test abs(r==1. ? (l+1)*f+r*d : l*f-r*d)<1e-12
    end
    for inner in (0,2),outer in (0,2),l in (1,3)
        R=Magrathea.recomb_magnetic_poloidal(Float64,12,l,.35,1.;bci=inner,bco=outer)
        errors=Float64[]
        for c in eachcol(R)
            fi,di,_=jet(c,.35);fo,do_,_=jet(c,1.)
            push!(errors,abs(inner==0 ? l*fi-.35di : fi),abs(outer==0 ? (l+1)*fo+do_ : fo))
        end
        @test maximum(errors)<1e-10
    end
    for (inner,outer) in ((1,0),(0,1),(9,0))
        @test_throws ArgumentError Magrathea.recomb_magnetic_poloidal(Float64,12,2,.35,1.;bci=inner,bco=outer)
    end
end

@testset "Tangential electric field of computed eigenmodes converges at walls" begin
    for bg in (axial,dipole),inner in (1,2),mechanical in (0,1)
        errors=Float64[]
        for N in (12,bg==axial ? 24 : 48)
            op=op_for(B0_type=bg,N=N,bci=mechanical,bco=mechanical,bci_magnetic=inner,bco_magnetic=2)
            _,V=eigenmodes(op);emax=0.;bmax=0.;umax=0.
            for x in eachcol(V),r in (.35,1.)
                d=boundary_fields(op,x,r);E=d.E
                if r==.35 && inner==1
                    core=boundary_fields(op,x,r;core=true)
                    bmax=max(bmax,norm(d.B-core.B)/norm(x))
                    E=E-core.E
                else
                    bmax=max(bmax,norm(d.B[:,1])/norm(x))
                end
                umax=max(umax,norm(d.U[:,1])/norm(x))
                mechanical==1 && (umax=max(umax,norm(d.U)/norm(x)))
                emax=max(emax,project_E(E,d,op)/norm(x))
            end
            @test bmax<1e-8
            @test umax<1e-8
            push!(errors,emax)
        end
        @test errors[2]<.01errors[1]
        @test errors[2]<(bg==axial ? 1e-7 : 2e-6)
    end
end
end # module
