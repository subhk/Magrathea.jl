# Linearization about an arbitrary mean state. Work with physical vectors:
#   F = U × curl(u) + u × curl(U),
#   g = -U·grad(Θ) - u·grad(Tbar).
# F differs from -(U·grad)u-(u·grad)U only by grad(U·u), which the
# pressure-eliminating projection removes. This includes all metric/shear terms.
#
# The onset radial matrices use u = curl curl(r P rhat) + curl(r T rhat)
# and complex harmonics Y_lm/sqrt(2l+1). This differs from both the public
# no-factorial mean-temperature coefficients and the native mean-flow potentials.

"""Complex orthonormal Y, ∂θY and (∂φY)/sinθ at φ=0, including signed m."""
function _coupling_harmonic(g::SHGrid{T}, l, m) where T
    a=abs(m); phase=m<0 && isodd(a) ? -one(T) : one(T)
    y=T[phase*g.N[a][l-a+1]*g.P[a][l-a+1,j] for j in eachindex(g.μ)]
    h=T[phase*g.N[a][l-a+1]*_sh_dPdθ(g,l,a,j) for j in eachindex(g.μ)]
    v=Complex{T}[im*m*y[j]/_sh_sinθ(g,j) for j in eachindex(g.μ)]
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

"""One Fourier component of the physical mean velocity, vorticity and ∇T."""
function _coupling_mean_fields(bs, r::Vector{T}, g, m) where T
    dims=(length(r),length(g.μ)); CT=Complex{T}
    U=ntuple(_->zeros(CT,dims),3); W=ntuple(_->zeros(CT,dims),3)
    G=ntuple(_->zeros(CT,dims),3)
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
        for l in max(1,abs(m)):f.lmax
            p=_coupling_profile(f.p,l,m,f.r,r;orthonormal=true)
            p===nothing && continue
            dp=_coupling_profile(f.dp,l,m,f.r,r;orthonormal=true)
            d2p=_coupling_profile(f.d2p,l,m,f.r,r;orthonormal=true)
            t=_coupling_profile(f.t,l,m,f.r,r;orthonormal=true)
            dt=_coupling_profile(f.dt,l,m,f.r,r;orthonormal=true)
            y,h,v=_coupling_harmonic(g,l,m); q=l*(l+1)
            U[1] .+= (q.*p./r.^2)*transpose(y)
            U[2] .+= (dp./r)*transpose(h) .- (t./r)*transpose(v)
            U[3] .+= (dp./r)*transpose(v) .+ (t./r)*transpose(h)
            lap=(d2p .- q.*p./r.^2)./r
            W[1] .-= (q.*t./r.^2)*transpose(y)
            W[2] .-= (dt./r)*transpose(h) .+ lap*transpose(v)
            W[3] .+= lap*transpose(h) .- (dt./r)*transpose(v)
        end
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
    U,W,G
end

"""Radial blocks of the linearized physical equations, before boundary constraints."""
function _mean_state_blocks(bs, source, target, m_from, m_to)
    r=source.r; T=eltype(r); CT=Complex{T}; n=length(r)
    r==target.r || throw(ArgumentError("Coupled modes require a common radial grid"))
    L=max(maximum(first(k) for k in keys(source.index_map)),
          maximum(first(k) for k in keys(target.index_map)),bs.lmax_bs,
          bs.flow===nothing ? 0 : bs.flow.lmax)
    mb=m_to-m_from
    blocks=Dict{Tuple{Int,Symbol,Int,Symbol},Matrix{CT}}()
    abs(mb)>L && return blocks
    g=sh_grid(L,max(abs(m_from),abs(m_to),abs(mb)),T)
    U,W,G=_coupling_mean_fields(bs,r,g,mb)
    D=(Matrix{T}(I,n,n),Matrix(source.cd.D1),Matrix(source.cd.D2),Matrix(source.cd.D3))
    weight=r.^(target.params.use_sparse_weighting ? 3 : 2)
    Z=zeros(CT,n,length(g.μ))
    for (li,field) in sort!(collect(keys(source.index_map)))
        y,h,v=_coupling_harmonic(g,li,m_from)
        scale=inv(sqrt(T(2li+1))); y.*=scale; h.*=scale; v.*=scale
        Y=ones(T,n)*transpose(y); H=ones(T,n)*transpose(h); V=ones(T,n)*transpose(v)
        q=li*(li+1)
        if field===:Θ
            heat=(-U[2].*H./r .- U[3].*V./r, -U[1].*Y)
            forces=()
        else
            if field===:P
                vel=((q.*Y./r,H./r,V./r),(Z,H,V),(Z,Z,Z))
                curl=((Z,q.*V./r.^2,-q.*H./r.^2),
                      (Z,-2 .*V./r,2 .*H./r),(Z,-V,H))
            else
                vel=((Z,V,-H),(Z,Z,Z))
                curl=((q.*Y./r,H./r,V./r),(Z,H,V))
            end
            forces=map(vel,curl) do u,w
                (U[2].*w[3].-U[3].*w[2].+u[2].*W[3].-u[3].*W[2],
                 U[3].*w[1].-U[1].*w[3].+u[3].*W[1].-u[1].*W[3],
                 U[1].*w[2].-U[2].*w[1].+u[1].*W[2].-u[2].*W[1])
            end
            heat=map(u->-(u[1].*G[1].+u[2].*G[2].+u[3].*G[3]),vel)
        end
        for (lo,out) in sort!(collect(keys(target.index_map)))
            field===:Θ && out!==:Θ && continue
            yo,ho,vo=_coupling_harmonic(g,lo,m_to)
            # Divide by the norm of Y/sqrt(2l+1), including the 2π φ integral.
            w=T(2π)*sqrt(T(2lo+1)).*g.w
            py=w.*conj.(yo); ph=w.*conj.(ho); pv=w.*conj.(vo)
            block=zeros(CT,n,n)
            if out===:Θ
                for k in eachindex(heat)
                    a=heat[k]*py
                    block .+= (weight.*a).*D[k]
                end
            else
                qo=lo*(lo+1)
                for k in eachindex(forces)
                    fr,fh,fv=forces[k]
                    if out===:P
                        a=fr*py
                        b=(fh*ph+fv*pv)./qo
                        # Apply the product rule before discretization. D*diag(b)*Dk
                        # aliases the highest perturbation polynomial even for a
                        # linear b (and fails the exact rigid-rotation identity).
                        rb=r.*b
                        block .+= (-qo.*r.^3).*((a-D[2]*rb).*D[k] - rb.*D[k+1])
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
