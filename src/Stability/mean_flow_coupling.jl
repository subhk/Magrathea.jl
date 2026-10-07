# Linearization about an arbitrary mean state. Work with physical vectors:
#   F = U × curl(u) + u × curl(U),
#   g = -U·grad(Θ) - u·grad(Tbar).
# F differs from -(U·grad)u-(u·grad)U only by grad(U·u), which the
# pressure-eliminating projection removes. This includes all metric/shear terms.
# With a mean field B̄ = B₀ + b̄ (imposed plus induced, current J̄ = curl b̄), the
# perturbation field b adds the Lorentz force Le²(curl(b) × B̄ + J̄ × b) to F, and
# the induction equation λb = curl(V) + E_m∇²b has the EMF V = u × B̄ + U × b.
#
# The onset radial matrices use u = curl curl(r P rhat) + curl(r T rhat)
# and complex harmonics Y_lm/sqrt(2l+1). This differs from both the public
# no-factorial mean-temperature coefficients and the native mean-flow potentials.

"""Complex orthonormal Y, ∂θY and (∂φY)/sinθ at φ=0, including signed m."""
function _coupling_harmonic(g::SHGrid{T}, l, m) where T
    a=abs(m); phase=m<0 && isodd(a) ? -one(T) : one(T)
    y=T[phase*g.Q[a][l-a+1,j] for j in eachindex(g.μ)]
    h=T[phase*_sh_dQdθ(g,l,a,j) for j in eachindex(g.μ)]
    v=Complex{T}[im*m*phase*_sh_Q_over_sin(g,l,a,j) for j in eachindex(g.μ)]  # pole-safe
    (y,h,v)
end

# Convert real cosine/sine coefficients to the coefficient of complex Y_lm.
# Temperature and legacy components have the public no-factorial normalization;
# native vector potentials are already real orthonormal.
function _coupling_profile(d, l, m, r_source, r; orthonormal=false)
    T=eltype(r); a=abs(m)
    A=get(d,(l,a),nothing); B=a==0 ? nothing : get(d,(l,-a),nothing)
    A===nothing && B===nothing && return nothing
    at(v)=v===nothing ? zeros(T,length(r)) :
        (r==r_source ? v : T[_mean_barycentric(r_source,v,x) for x in r])
    av=at(A)
    c=a==0 ? Complex{T}.(av) : (m>0 ? (av .- im.*at(B))./sqrt(T(2)) :
        (isodd(a) ? -one(T) : one(T)).*(av .+ im.*at(B))./sqrt(T(2)))
    orthonormal || (c .*= _sh_nf_to_orth_factor(l,a,T))
    c
end

_coupling_dict(d::Dict{Int,Vector{T}}) where T = Dict((l,0)=>v for (l,v) in d)
_coupling_dict(d::Dict{Tuple{Int,Int},Vector{T}}) where T = d

"""Add Fourier component `m` of the solenoidal field of potentials `f` (velocity or
magnetic) to `V`, and of its curl to `C` unless `C === nothing`."""
function _add_solenoidal_component!(V, C, f, r, g, m)
    for l in max(1,abs(m)):f.lmax
        p=_coupling_profile(f.p,l,m,f.r,r;orthonormal=true)
        p===nothing && continue
        dp=_coupling_profile(f.dp,l,m,f.r,r;orthonormal=true)
        t=_coupling_profile(f.t,l,m,f.r,r;orthonormal=true)
        y,h,v=_coupling_harmonic(g,l,m); q=l*(l+1)
        V[1] .+= (q.*p./r.^2)*transpose(y)
        V[2] .+= (dp./r)*transpose(h) .- (t./r)*transpose(v)
        V[3] .+= (dp./r)*transpose(v) .+ (t./r)*transpose(h)
        C===nothing && continue
        d2p=_coupling_profile(f.d2p,l,m,f.r,r;orthonormal=true)
        dt=_coupling_profile(f.dt,l,m,f.r,r;orthonormal=true)
        lap=(d2p .- q.*p./r.^2)./r
        C[1] .-= (q.*t./r.^2)*transpose(y)
        C[2] .-= (dt./r)*transpose(h) .+ lap*transpose(v)
        C[3] .+= lap*transpose(h) .- (dt./r)*transpose(v)
    end
end

"""The imposed field as native potentials on `r`: `p₁₀ = r h(r)/c` with
`B₀ = ∇×∇×(h cosθ 𝐫)` and `c = Y₁₀/cosθ` for the real orthonormal harmonic."""
function _background_field_flow(B0_type::BackgroundField, r::AbstractVector{T}) where T
    c=_normalized_legendre_table(0,1,T[1])[2,1]
    p,dp,d2p = B0_type == axial ? (r.^2 ./ 2, r, one.(r)) :
               B0_type == dipole ? (inv.(2 .* r), -inv.(2 .* r.^2), inv.(r.^3)) :
               throw(ArgumentError("No imposed field for B0_type=$B0_type"))
    key=(1,0); z=zero(collect(r))
    SolenoidalMeanFlow(1,0,collect(r),Dict(key=>p./c),Dict(key=>z),Dict(key=>dp./c),
                       Dict(key=>d2p./c),Dict(key=>z))
end

"""Mean field of a basic state (the induced field `b̄`), or `nothing`."""
_mean_field(bs) = bs !== nothing && hasproperty(bs, :field) ? bs.field : nothing

"""One Fourier component of the physical mean velocity, vorticity, ∇T, magnetic field
and current. `bs === nothing` is the motionless conductive state, whose only coupling
is through the imposed field `B0_type` (current free, m = 0)."""
function _coupling_mean_fields(bs, r::Vector{T}, g, m; B0_type=no_field) where T
    dims=(length(r),length(g.μ)); CT=Complex{T}
    U=ntuple(_->zeros(CT,dims),3); W=ntuple(_->zeros(CT,dims),3)
    G=ntuple(_->zeros(CT,dims),3)
    B=ntuple(_->zeros(CT,dims),3); J=ntuple(_->zeros(CT,dims),3)
    m==0 && B0_type != no_field && _add_solenoidal_component!(B,nothing,
        _background_field_flow(B0_type,r),r,g,0)
    bs===nothing && return U,W,G,B,J
    b=_mean_field(bs)
    b===nothing || _add_solenoidal_component!(B,J,b,r,g,m)
    td=_coupling_dict(bs.theta_coeffs); dtd=_coupling_dict(bs.dtheta_dr_coeffs)
    for l in abs(m):bs.lmax_bs
        c=_coupling_profile(td,l,m,bs.r,r); c===nothing && continue
        dc=_coupling_profile(dtd,l,m,bs.r,r)
        dc===nothing && (dc=ChebyshevDiffn(length(r),[first(r),last(r)],1).D1*c)
        y,h,v=_coupling_harmonic(g,l,m)
        G[1] .+= dc*transpose(y)
        G[2] .+= (c./r)*transpose(h)
        G[3] .+= (c./r)*transpose(v)
    end
    f=bs.flow
    if f!==nothing
        _add_solenoidal_component!(U,W,f,r,g,m)
    else
        # Custom states may supply scalar component expansions without potentials.
        # Differentiate those actual expansions, including the spherical metrics.
        ds=map(_coupling_dict,(bs.ur_coeffs,bs.utheta_coeffs,bs.uphi_coeffs))
        drs=map(_coupling_dict,(bs.dur_dr_coeffs,bs.dutheta_dr_coeffs,bs.duphi_dr_coeffs))
        dR=ntuple(_->zeros(CT,dims),3); dH=ntuple(_->zeros(CT,dims),3)
        for l in abs(m):bs.lmax_bs, a in 1:3
            c=_coupling_profile(ds[a],l,m,bs.r,r); c===nothing && continue
            dc=_coupling_profile(drs[a],l,m,bs.r,r)
            dc===nothing && (dc=ChebyshevDiffn(length(r),[first(r),last(r)],1).D1*c)
            y,h,v=_coupling_harmonic(g,l,m)
            U[a] .+= c*transpose(y); dR[a] .+= dc*transpose(y)
            dH[a] .+= c*transpose(h)
        end
        s=sqrt.(1 .-g.μ.^2); cotθ=g.μ./s
        W[1] .= (dH[3] .+ U[3].*transpose(cotθ) .- im*m.*U[2]./transpose(s))./r
        W[2] .= im*m.*U[1]./(r*transpose(s)) .- dR[3] .- U[3]./r
        W[3] .= dR[2] .+ U[2]./r .- dH[1]./r
    end
    U,W,G,B,J
end

"""Radial blocks of the linearized physical equations, before boundary constraints.

`bs === nothing` stands for the motionless conductive state: only the imposed field
then couples, since the conduction gradient is part of the radial assembly."""
function _mean_state_blocks(bs, source, target, m_from, m_to)
    r=source.r; T=eltype(r); CT=Complex{T}; n=length(r)
    r==target.r || throw(ArgumentError("Coupled modes require a common radial grid"))
    params=target.params; magnetic=_has_magnetic(params)
    b̄=_mean_field(bs)
    L=max(maximum(first(k) for k in keys(source.index_map)),
          maximum(first(k) for k in keys(target.index_map)),
          bs===nothing ? 0 : bs.lmax_bs,
          bs===nothing || bs.flow===nothing ? 0 : bs.flow.lmax,
          b̄===nothing ? 0 : b̄.lmax, magnetic ? 1 : 0)
    mb=m_to-m_from
    blocks=Dict{Tuple{Int,Symbol,Int,Symbol},Matrix{CT}}()
    abs(mb)>L && return blocks
    g=sh_grid(L,max(abs(m_from),abs(m_to),abs(mb)),T)
    U,W,G,B,J=_coupling_mean_fields(bs,r,g,mb;B0_type=magnetic ? params.B0_type : no_field)
    Le2=magnetic ? params.Le^2 : zero(T)
    D=(Matrix{T}(I,n,n),Matrix(source.cd.D1),Matrix(source.cd.D2),Matrix(source.cd.D3))
    weight=r.^(target.params.use_sparse_weighting ? 3 : 2)
    Z=zeros(CT,n,length(g.μ))
    cross(a,b)=(a[2].*b[3].-a[3].*b[2], a[3].*b[1].-a[1].*b[3], a[1].*b[2].-a[2].*b[1])
    for (li,field) in sort!(collect(keys(source.index_map)))
        y,h,v=_coupling_harmonic(g,li,m_from)
        scale=inv(sqrt(T(2li+1))); y.*=scale; h.*=scale; v.*=scale
        Y=ones(T,n)*transpose(y); H=ones(T,n)*transpose(h); V=ones(T,n)*transpose(v)
        q=li*(li+1)
        # The vector field of a unit potential (u for P/T, b for F/G) and its curl,
        # by the order of the radial derivative they act on.
        if field===:P || field===:F
            vec=((q.*Y./r,H./r,V./r),(Z,H,V),(Z,Z,Z))
            curl=((Z,q.*V./r.^2,-q.*H./r.^2),
                  (Z,-2 .*V./r,2 .*H./r),(Z,-V,H))
        elseif field===:T || field===:G
            vec=((Z,V,-H),(Z,Z,Z))
            curl=((q.*Y./r,H./r,V./r),(Z,H,V))
        end
        forces=(); heat=(); emf=()
        if field===:Θ
            bs===nothing || (heat=(-U[2].*H./r .- U[3].*V./r, -U[1].*Y))
        elseif field===:P || field===:T
            if bs!==nothing
                forces=map(vec,curl) do u,w
                    (U[2].*w[3].-U[3].*w[2].+u[2].*W[3].-u[3].*W[2],
                     U[3].*w[1].-U[1].*w[3].+u[3].*W[1].-u[1].*W[3],
                     U[1].*w[2].-U[2].*w[1].+u[1].*W[2].-u[2].*W[1])
                end
                heat=map(u->-(u[1].*G[1].+u[2].*G[2].+u[3].*G[3]),vec)
            end
            magnetic && (emf=map(u->cross(u,B),vec))
        else
            # Lorentz force on the flow and the EMF of mean-flow advection of b.
            forces=map(vec,curl) do b,j
                f=cross(j,B)
                b̄===nothing || (f=f.+cross(J,b))
                Le2.*f
            end
            bs===nothing || (emf=map(b->cross(U,b),vec))
        end
        for (lo,out) in sort!(collect(keys(target.index_map)))
            terms=out===:Θ ? heat : (out===:P || out===:T) ? forces : emf
            isempty(terms) && continue
            yo,ho,vo=_coupling_harmonic(g,lo,m_to)
            # Divide by the norm of Y/sqrt(2l+1), including the 2π φ integral.
            w=T(2π)*sqrt(T(2lo+1)).*g.w
            py=w.*conj.(yo); ph=w.*conj.(ho); pv=w.*conj.(vo)
            block=zeros(CT,n,n)
            if out===:Θ
                for k in eachindex(terms)
                    a=terms[k]*py
                    block .+= (weight.*a).*D[k]
                end
            else
                qo=lo*(lo+1)
                for k in eachindex(terms)
                    fr,fh,fv=terms[k]
                    if out===:P || out===:G
                        a=fr*py
                        b=(fh*ph+fv*pv)./qo
                        # Apply the product rule before discretization. D*diag(b)*Dk
                        # aliases the highest perturbation polynomial even for a
                        # linear b (and fails the exact rigid-rotation identity).
                        # The G rows carry +𝐫·∇×∇×V where the P rows carry -𝐫·∇×∇×F.
                        rb=r.*b
                        sgn=out===:P ? -qo : qo
                        block .+= (sgn.*r.^3).*((a-D[2]*rb).*D[k] - rb.*D[k+1])
                    else
                        c=(-fh*pv+fv*ph)./qo
                        block .+= (qo .* r.^2 .* c).*D[k]
                    end
                end
            end
            # Preserve small physical amplitudes; only exactly zero blocks are
            # discarded below, without an amplitude-dependent cutoff.
            blocks[(lo,out,li,field)]=block
        end
    end
    filter!(kv->any(!iszero,last(kv)),blocks)
    blocks
end

function _mean_state_matrix(bs,source,target,m_from,m_to)
    T=eltype(source.r); C=zeros(Complex{T},target.total_dof,source.total_dof)
    for ((lo,fo,li,fi),block) in _mean_state_blocks(bs,source,target,m_from,m_to)
        C[target.index_map[(lo,fo)],source.index_map[(li,fi)]] .+= block
    end
    C
end
