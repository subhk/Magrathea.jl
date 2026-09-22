"""
Solenoidal mean velocity in real orthonormal vector spherical harmonics.

For q=l(l+1), u_r=q*p/r², u_h=(p'/r)∇hY+(t/r)r̂×∇hY.
The potentials, rather than scalar projections of the tangential components,
are the authoritative velocity representation.
"""
struct SolenoidalMeanFlow{T<:Real}
    lmax::Int
    mmax::Int
    r::Vector{T}
    p::Dict{Tuple{Int,Int},Vector{T}}
    t::Dict{Tuple{Int,Int},Vector{T}}
    dp::Dict{Tuple{Int,Int},Vector{T}}
    d2p::Dict{Tuple{Int,Int},Vector{T}}
    dt::Dict{Tuple{Int,Int},Vector{T}}
end

const _MEAN_CORIOLIS_CACHE = Dict{Tuple{Int,Int,DataType},Any}()

# Project 2 ẑ×u onto (Y r̂, ∇hY, r̂×∇hY). The tangential basis has norm²=q.
# Evaluating the actual vector cross product also handles cosine/sine phases.
function _mean_coriolis(lmax::Int, am::Int, ::Type{T}) where T
    get!(_MEAN_CORIOLIS_CACHE, (lmax, am, T)) do
        modes = [(l,m) for m in (am == 0 ? (0,) : (am,-am)) for l in max(1,am):lmax]
        g = sh_grid(lmax, am, T)
        n = length(modes)
        Y = zeros(T, length(g.μ)*length(g.φ), n)
        Hθ = similar(Y); Hφ = similar(Y)
        w = T[]; s = T[]; c = T[]
        for k in eachindex(g.φ), j in eachindex(g.μ)
            push!(w, g.w[j]*2T(π)/length(g.φ))
            push!(s, _sh_sinθ(g,j)); push!(c,g.μ[j])
            h = j+(k-1)*length(g.μ)
            for (a,(l,m)) in enumerate(modes)
                Y[h,a]=_sh_Y(g,l,m,j,k)
                Hθ[h,a]=_sh_dYθ(g,l,m,j,k)
                Hφ[h,a]=_sh_dYφ_over_sin(g,l,m,j,k)
            end
        end
        Z = zero(Y)
        basis = ((Y,Z,Z),(Z,Hθ,Hφ),(Z,-Hφ,Hθ))
        C = zeros(T,3n,3n)
        for a in 1:3, b in 1:3
            ar,at,ap = basis[a]; br,bt,bp = basis[b]
            block = ar' * ((-2 .* w .* s) .* bp) +
                    at' * ((-2 .* w .* c) .* bp) +
                    ap' * ((2 .* w .* s) .* br + (2 .* w .* c) .* bt)
            if a != 1
                for (i,(l,m)) in enumerate(modes)
                    block[i,:] ./= l*(l+1)
                end
            end
            C[(a-1)*n+1:a*n,(b-1)*n+1:b*n] .= block
        end
        (modes,C)
    end
end

# Nodal integration weights on an ascending Chebyshev grid.
function _mean_radial_weights(r::Vector{T}) where T
    n=length(r); x=(2 .* r .- first(r) .- last(r))./(last(r)-first(r))
    V=T[cos(k*acos(clamp(z,-one(T),one(T)))) for z in x, k in 0:n-1]
    moments=T[iseven(k) ? 2/(1-k*k) : 0 for k in 0:n-1]
    (V' \ moments) .* ((last(r)-first(r))/2)
end

"""
Solve 2 ẑ×u = -∇π + β r T r̂ + E∇²u, ∇·u=0, with β=Ra E²/[Pr(1-χ)³].
Ra is shell-gap based; radius and time use r_o and Ω⁻¹. Inertia is neglected.
Both boundaries are impermeable and no-slip or stress-free. The stress-free
axisymmetric solid-rotation nullspace is fixed by zero axial angular momentum.
Input temperature coefficients use the public no-factorial normalization.
"""
function _steady_mean_flow(theta, r::Vector{T}, D1, D2, E, Ra, Pr, lmax, mmax;
                           mechanical_bc=:no_slip) where T
    E > 0 || throw(ArgumentError("The viscous mean-flow solve requires E > 0"))
    Pr > 0 || throw(ArgumentError("Pr must be positive"))
    mechanical_bc in (:no_slip,:stress_free) || throw(ArgumentError("Invalid mechanical_bc"))
    length(r)>=6 || throw(ArgumentError("The mean-flow solve requires at least 6 radial nodes"))
    lmax>=0 && 0<=mmax<=lmax || throw(ArgumentError("Require lmax ≥ 0 and 0 ≤ mmax ≤ lmax"))
    first(r)>0 && last(r)>first(r) || throw(ArgumentError("Expected ascending shell radii"))
    for ((l,m),v) in theta
        length(v)==length(r) || throw(DimensionMismatch("Temperature profile length must match the radial grid"))
        all(isfinite,v) || throw(ArgumentError("Temperature profiles must be finite"))
        if any(!iszero,v)
            abs(m)<=l<=lmax && abs(m)<=mmax || throw(ArgumentError("Temperature mode ($l,$m) is outside the retained harmonic range"))
        end
    end
    N=length(r); D=Matrix{T}(D1); D²=Matrix{T}(D2); D4=D²*D²
    β=T(Ra*E^2/(Pr*(last(r)-first(r))^3))
    θ=_sh_rescale(theta,+1)
    p=Dict{Tuple{Int,Int},Vector{T}}(); t=empty(p)
    for am in 0:mmax
        any(l>0 && abs(m)==am && any(!iszero,v) for ((l,m),v) in θ) || continue
        modes,C=_mean_coriolis(lmax,am,T); n=length(modes)
        A=zeros(T,2n*N,2n*N); f=zeros(T,2n*N)
        for (a,(l,m)) in enumerate(modes)
            rows=(a-1)*N+1:a*N; rt=n*N .+ rows
            f[rows] .= β .* r .* get(θ,(l,m),zeros(T,N))
            q=T(l*(l+1))
            A[rows,rows] .+= E .* (D4 - 2q .* (D² ./ r.^2) +
                4q .* (D ./ r.^3) + Diagonal(q*(q-6) ./ r.^4))
            A[rt,rt] .-= E .* ((D²-Diagonal(q ./ r.^2)) ./ r)
            for (b,(L,M)) in enumerate(modes)
                cols=(b-1)*N+1:b*N; ct=n*N .+ cols
                R=Diagonal(T(L*(L+1)) ./ r.^2); S=D ./ r; U=Diagonal(one(T) ./ r)
                CRp=C[a,b]*R+C[a,n+b]*S
                CSp=C[n+a,b]*R+C[n+a,n+b]*S
                A[rows,cols] .+= CRp-D*(r .* CSp)
                A[rows,ct] .+= C[a,2n+b]*U-D*(r .* (C[n+a,2n+b]*U))
                A[rt,cols] .+= C[2n+a,b]*R+C[2n+a,n+b]*S
                A[rt,ct] .+= C[2n+a,2n+b]*U
            end
            # Four conditions on p and two on t. Boundary rows replace equations.
            for (row,node,kind) in ((first(rows),1,:p),(first(rows)+1,1,:dp),
                                    (last(rows)-1,N,:dp),(last(rows),N,:p),
                                    (first(rt),1,:t),(last(rt),N,:t))
                A[row,:] .= 0; f[row]=0
                if kind==:p
                    A[row,rows[node]]=1
                elseif kind==:dp
                    A[row,rows] .= mechanical_bc==:no_slip ? D[node,:] : D²[node,:]-2D[node,:]/r[node]
                elseif mechanical_bc==:no_slip
                    A[row,rt[node]]=1
                else
                    A[row,rt] .= D[node,:]
                    A[row,rt[node]] -= 2/r[node]
                end
            end
        end
        if mechanical_bc==:stress_free && am==0
            # Bordered solve preserves both stress conditions; the multiplier
            # is the net axial torque and vanishes for radial thermal forcing.
            gauge=zeros(T,2n*N); torque=copy(gauge)
            a=findfirst(==((1,0)),modes); rt=n*N .+ ((a-1)*N+1:a*N)
            gauge[rt] .= _mean_radial_weights(r).*r.^2
            torque[rt[2:end-1]] .= r[2:end-1]
            A=[A torque; gauge' zero(T)]; f=[f;zero(T)]
        end
        scales=maximum(abs,A;dims=2)
        x=(A ./ scales) \ (f ./ vec(scales))
        all(isfinite,x) || error("Non-finite steady mean-flow solution")
        for (a,key) in enumerate(modes)
            rows=(a-1)*N+1:a*N
            p[key]=x[rows]; t[key]=x[n*N .+ rows]
        end
    end
    SolenoidalMeanFlow(lmax,mmax,r,p,t,Dict(k=>D*v for (k,v) in p),
        Dict(k=>D²*v for (k,v) in p),Dict(k=>D*v for (k,v) in t))
end

# Implicit advection-diffusion at fixed velocity, in the orthonormal scalar
# basis. Solving transport implicitly avoids the diffusion-only Picard update
# becoming unstable as the Peclet number increases.
function _mean_temperature_step(flow::SolenoidalMeanFlow{T},D1,D2,κ,bc) where T
    g=sh_grid(flow.lmax,flow.mmax,T); r=flow.r; N=length(r)
    modes=[(l,m) for m in -g.mmax:g.mmax for l in abs(m):g.lmax]
    n=length(modes); ng=length(g.μ)*length(g.φ)
    Y=zeros(T,ng,n); Hθ=similar(Y); Hφ=similar(Y); w=zeros(T,ng)
    for k in eachindex(g.φ),j in eachindex(g.μ)
        h=j+(k-1)*length(g.μ); w[h]=g.w[j]*2T(π)/length(g.φ)
        for (a,(l,m)) in enumerate(modes)
            Y[h,a]=_sh_Y(g,l,m,j,k); Hθ[h,a]=_sh_dYθ(g,l,m,j,k)
            Hφ[h,a]=_sh_dYφ_over_sin(g,l,m,j,k)
        end
    end
    A=zeros(T,n*N,n*N); f=zeros(T,n*N)
    for (a,(l,m)) in enumerate(modes)
        ix=(a-1)*N+1:a*N
        A[ix,ix] .= κ.*(D2+2 .* D1 ./ r-Diagonal(l*(l+1) ./ r.^2))
    end
    for i in 2:N-1
        ur,uθ,uφ=_mean_flow_grid(flow,i,g)
        Ar=Y' * ((w.*vec(ur)).*Y)
        Ah=Y' * ((w.*vec(uθ)).*Hθ+(w.*vec(uφ)).*Hφ) ./ r[i]
        for a in 1:n,b in 1:n
            row=(a-1)*N+i; cols=(b-1)*N+1:b*N
            A[row,cols] .-= Ar[a,b].*D1[i,:]
            A[row,cols[i]] -= Ah[a,b]
        end
    end
    for (a,key) in enumerate(modes)
        ix=(a-1)*N+1:a*N
        inner,outer,kind=bc[key]; factor=_sh_nf_to_orth_factor(key...,T)
        A[first(ix),:] .= 0; A[first(ix),first(ix)]=1; f[first(ix)]=inner*factor
        A[last(ix),:] .= 0; f[last(ix)]=outer*factor
        if kind==:fixed_temperature
            A[last(ix),last(ix)]=1
        else
            A[last(ix),ix] .= D1[end,:]
        end
    end
    scales=maximum(abs,A;dims=2)
    x=(A ./ scales) \ (f ./ vec(scales))
    all(isfinite,x) || error("Non-finite mean temperature solution")
    Dict(key=>x[(a-1)*N+1:a*N]./_sh_nf_to_orth_factor(key...,T) for (a,key) in enumerate(modes))
end

function _mean_barycentric(r,v,x)
    k=findfirst(==(x),r); k===nothing || return v[k]
    first(r)<=x<=last(r) || throw(ArgumentError("Radius is outside the shell"))
    weights=[(iseven(i) ? -one(x) : one(x))*(i in (1,length(r)) ? 0.5 : 1)/(x-r[i]) for i in eachindex(r)]
    dot(weights,v)/sum(weights)
end

"""
    mean_flow_velocity(bs, r, θ, φ=0)

Evaluate `(ur, utheta, uphi)` from a constructed basic state's divergence-free
vector harmonics, with spectral radial interpolation. θ is colatitude. Scalar
component coefficient dictionaries are compatibility projections; use this
function for physical fields, continuity checks, and convergence studies.
"""
function mean_flow_velocity(bs, r::Real, θ::Real, φ::Real=0)
    first(bs.r)<=r<=last(bs.r) || throw(ArgumentError("Radius is outside the shell"))
    0<=θ<=π || throw(ArgumentError("Colatitude must satisfy 0 ≤ θ ≤ π"))
    flow=bs.flow
    if flow===nothing
        if all(all(iszero,v) for d in (bs.ur_coeffs,bs.utheta_coeffs,bs.uphi_coeffs) for v in values(d))
            z=zero(eltype(bs.r))
            return (ur=z,utheta=z,uphi=z)
        end
        throw(ArgumentError("This custom state has no vector-harmonic mean flow"))
    end
    T=eltype(flow.r); μ=T[cos(θ)]; L=flow.lmax; M=flow.mmax
    g=SHGrid{T}(L,M,μ,T[2],T[φ],
        Dict(a=>_associated_legendre_table(a,L,μ) for a in 0:min(M+1,L)),
        Dict(a=>_normalization_table(T,a,L) for a in 0:M))
    at(d)=Dict(k=>T[_mean_barycentric(flow.r,v,T(r))] for (k,v) in d)
    f=SolenoidalMeanFlow(L,M,T[r],at(flow.p),at(flow.t),at(flow.dp),at(flow.d2p),at(flow.dt))
    ur,uθ,uφ=_mean_flow_grid(f,1,g)
    (ur=only(ur),utheta=only(uθ),uphi=only(uφ))
end

"""Evaluate the vector-harmonic velocity on an SH grid at radial node i."""
function _mean_flow_grid(flow::SolenoidalMeanFlow{T},i,g; derivative=false) where T
    R=Dict{Tuple{Int,Int},T}(); S=empty(R); U=empty(R); r=flow.r[i]
    for (key,p) in flow.p
        l,m=key; q=l*(l+1)
        R[key]=derivative ? q*(flow.dp[key][i]/r^2-2p[i]/r^3) : q*p[i]/r^2
        S[key]=derivative ? flow.d2p[key][i]/r-flow.dp[key][i]/r^2 : flow.dp[key][i]/r
        U[key]=derivative ? flow.dt[key][i]/r-flow.t[key][i]/r^2 : flow.t[key][i]/r
    end
    ur=sh_synthesize(R,g)
    uθ=sh_synthesize(S,g;Yf=_sh_dYθ)-sh_synthesize(U,g;Yf=_sh_dYφ_over_sin)
    uφ=sh_synthesize(S,g;Yf=_sh_dYφ_over_sin)+sh_synthesize(U,g;Yf=_sh_dYθ)
    (ur,uθ,uφ)
end

# Compatibility projections for existing scalar-coefficient consumers. Never
# feed these tangential projections back into the solenoidal momentum solve.
function _mean_flow_components(flow::SolenoidalMeanFlow{T}) where T
    g=sh_grid(flow.lmax,flow.mmax,T); N=length(flow.r)
    active_m=Set(abs(m) for (l,m) in keys(flow.p))
    fields=ntuple(_->Dict((l,m)=>zeros(T,N) for m in -g.mmax:g.mmax for l in abs(m):g.lmax),6)
    isempty(flow.p) && return fields
    for i in 1:N
        for (d,v) in zip(fields,(_mean_flow_grid(flow,i,g)...,_mean_flow_grid(flow,i,g;derivative=true)...))
            for (k,a) in sh_analyze(v,g)
                abs(k[2]) in active_m || continue
                d[k][i]=a/_sh_nf_to_orth_factor(k...,T)
            end
        end
    end
    fields
end

function _mean_flow_advection(theta,dtheta,flow::SolenoidalMeanFlow{T}) where T
    g=sh_grid(flow.lmax,flow.mmax,T); N=length(flow.r)
    θ=_sh_rescale(theta,+1); dθ=_sh_rescale(dtheta,+1)
    out=Dict((l,m)=>zeros(T,N) for m in -g.mmax:g.mmax for l in abs(m):g.lmax)
    for i in 1:N
        a=Dict(k=>v[i] for (k,v) in θ); da=Dict(k=>v[i] for (k,v) in dθ)
        ur,uθ,uφ=_mean_flow_grid(flow,i,g)
        adv=ur.*sh_synthesize(da,g)+(uθ.*sh_synthesize(a,g;Yf=_sh_dYθ)+
            uφ.*sh_synthesize(a,g;Yf=_sh_dYφ_over_sin))./flow.r[i]
        for (k,v) in sh_analyze(adv,g)
            out[k][i]=v/_sh_nf_to_orth_factor(k...,T)
        end
    end
    out
end
