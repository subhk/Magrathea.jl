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

# Julia 1.10 builds `Dict{Any,Any}` from an empty comprehension whose element type
# it cannot infer; convert such dictionaries to the field types.
SolenoidalMeanFlow(lmax,mmax,r::Vector{T},p,t,dp,d2p,dt) where T<:Real =
    SolenoidalMeanFlow{T}(lmax,mmax,r,p,t,dp,d2p,dt)

# Module-level memo caches (`_MEAN_CORIOLIS_CACHE`, `_SH_GRID_CACHE`,
# `_GAUNT_CACHE`) are shared by all tasks and threads. Look up under the lock,
# build outside it (builds are expensive and may consult other caches), then
# insert with `get!` under the lock: racing callers all receive the first
# stored value, and no lock is held while another is acquired.
struct _CacheMiss end

function _locked_get!(build, cache::AbstractDict, cache_lock::ReentrantLock, key)
    hit = @lock cache_lock get(cache, key, _CacheMiss())
    hit isa _CacheMiss || return hit
    value = build()
    return @lock cache_lock get!(cache, key, value)
end

const _MEAN_CORIOLIS_CACHE = Dict{Tuple{Int,Int,DataType},Any}()
const _MEAN_CORIOLIS_CACHE_LOCK = ReentrantLock()

# Project 2 ẑ×u onto (Y r̂, ∇hY, r̂×∇hY). The tangential basis has norm²=q.
# Evaluating the actual vector cross product also handles cosine/sine phases.
function _mean_coriolis(lmax::Int, am::Int, ::Type{T}) where T
    _locked_get!(_MEAN_CORIOLIS_CACHE, _MEAN_CORIOLIS_CACHE_LOCK, (lmax, am, T)) do
        modes = [(l,m) for m in (am == 0 ? (0,) : (am,-am)) for l in max(1,am):lmax]
        n = length(modes)
        (; Y, Hθ, Hφ, w, sinθ, cosθ) = _sh_basis_samples(sh_grid(lmax, am, T), modes)
        s = sinθ; c = cosθ
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

const _MEAN_FIELD_CROSS_CACHE = Dict{Tuple{Int,Int,DataType,BackgroundField},Any}()
const _MEAN_FIELD_CROSS_CACHE_LOCK = ReentrantLock()

"""
    _mean_field_cross(lmax, am, T, B0_type) -> (modes, M)

Project `V×b` onto `(Y r̂, ∇hY, r̂×∇hY)`, as `_mean_coriolis` does for `2ẑ×u`, where
`B₀ = _background_radial_factor(B0_type, r) · V(θ)` splits the imposed field into a
radial factor and an angular profile `V = (V_r, V_θ, 0)`: `(cosθ, -sinθ)` for the
axial field ẑ and `(cosθ, sinθ/2)` for the dipole. For `axial`, `2M` is the Coriolis
matrix.
"""
function _mean_field_cross(lmax::Int, am::Int, ::Type{T}, B0_type::BackgroundField) where T
    _locked_get!(_MEAN_FIELD_CROSS_CACHE, _MEAN_FIELD_CROSS_CACHE_LOCK, (lmax, am, T, B0_type)) do
        modes = [(l,m) for m in (am == 0 ? (0,) : (am,-am)) for l in max(1,am):lmax]
        n = length(modes)
        (; Y, Hθ, Hφ, w, sinθ, cosθ) = _sh_basis_samples(sh_grid(lmax, am, T), modes)
        Vr = cosθ
        Vθ = B0_type == axial ? -sinθ : B0_type == dipole ? sinθ ./ 2 :
            throw(ArgumentError("No imposed field for B0_type=$B0_type"))
        Z = zero(Y)
        basis = ((Y,Z,Z),(Z,Hθ,Hφ),(Z,-Hφ,Hθ))
        M = zeros(T,3n,3n)
        for a in 1:3, b in 1:3
            ar,at,ap = basis[a]; br,bt,bp = basis[b]
            # V×b = (V_θ b_φ, -V_r b_φ, V_r b_θ - V_θ b_r)
            block = ar' * ((w .* Vθ) .* bp) + at' * ((-w .* Vr) .* bp) +
                    ap' * ((w .* Vr) .* bt - (w .* Vθ) .* br)
            if a != 1
                for (i,(l,m)) in enumerate(modes)
                    block[i,:] ./= l*(l+1)
                end
            end
            M[(a-1)*n+1:a*n,(b-1)*n+1:b*n] .= block
        end
        (modes,M)
    end
end

"""Radial factor of the imposed field: 1 for the axial field, `r⁻³` for the dipole."""
_background_radial_factor(B0_type::BackgroundField, r) = B0_type == dipole ? inv.(r.^3) : one.(r)

# Nodal integration weights on an ascending Chebyshev grid.
function _mean_radial_weights(r::Vector{T}) where T
    n=length(r); x=(2 .* r .- first(r) .- last(r))./(last(r)-first(r))
    V=T[cos(k*acos(clamp(z,-one(T),one(T)))) for z in x, k in 0:n-1]
    moments=T[iseven(k) ? 2/(1-k*k) : 0 for k in 0:n-1]
    (V' \ moments) .* ((last(r)-first(r))/2)
end

"""
Solve 2 ẑ×u = -∇π + β r T r̂ + E∇²u, ∇·u=0, with β=Ra E²/[Pr(1-χ)³].
Ra is shell-gap based; radius and time use r_o and Ω⁻¹. Optional `inertia`
adds a frozen projection of u×curl(u) to the right-hand side for Picard updates.
Both boundaries are impermeable and no-slip or stress-free. The stress-free
axisymmetric solid-rotation nullspace is fixed by zero axial angular momentum.
Input temperature coefficients use the public no-factorial normalization.
"""
function _steady_mean_flow(theta, r::Vector{T}, D1, D2, E, Ra, Pr, lmax, mmax;
                           mechanical_bc=:no_slip, inertia=nothing,
                           systems=Dict{Int,Any}()) where T
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
    N=length(r); D=Matrix{T}(D1); D²=Matrix{T}(D2)
    β=T(Ra*E^2/(Pr*(last(r)-first(r))^3))
    θ=_sh_rescale(theta,+1)
    p=Dict{Tuple{Int,Int},Vector{T}}(); t=empty(p)
    for am in 0:mmax
        active=any(l>0 && abs(m)==am && any(!iszero,v) for ((l,m),v) in θ)
        if inertia!==nothing
            active |= any(abs(m)==am && any(!iszero,v) for d in (inertia.p,inertia.t) for ((l,m),v) in d)
        end
        active || continue
        system=get!(systems,am) do
            _mean_momentum_system(r,D,D²,E,lmax,am,mechanical_bc)
        end
        x=_mean_momentum_solve(system,_mean_momentum_rhs(system,θ,inertia,r,β))
        all(isfinite,x) || error("Non-finite steady mean-flow solution")
        n=length(system.modes)
        for (a,key) in enumerate(system.modes)
            rows=(a-1)*N+1:a*N
            p[key]=x[rows]; t[key]=x[n*N .+ rows]
        end
    end
    SolenoidalMeanFlow(lmax,mmax,r,p,t,Dict(k=>D*v for (k,v) in p),
        Dict(k=>D²*v for (k,v) in p),Dict(k=>D*v for (k,v) in t))
end

"""
    _steady_mean_mhd(theta, r, D1, D2, E, Ra, Pr, lmax, mmax; magnetic, mechanical_bc,
                     forcing=nothing, induction=nothing, systems) -> (flow, field)

The MHD version of `_steady_mean_flow`: solve the steady momentum balance together with
the induction equation of the induced field b̄, coupled through the imposed field
`magnetic.B0_type` (the Lorentz force Le²(∇×b̄)×B₀ and the EMF U×B₀; see
`_add_mean_magnetic_terms!`), so the magnetic braking and induction are implicit.
`forcing` adds frozen momentum forcing (inertia and Le²J̄×b̄) and `induction` the
frozen EMF U×b̄, both from `_mean_cross_projection`. `magnetic` is
`(Le, Em, B0_type, magnetic_bc)` with `Em = E/Pm`.
"""
function _steady_mean_mhd(theta, r::Vector{T}, D1, D2, E, Ra, Pr, lmax, mmax; magnetic,
                          mechanical_bc=:no_slip, forcing=nothing, induction=nothing,
                          systems=Dict{Int,Any}()) where T
    N=length(r); D=Matrix{T}(D1); D²=Matrix{T}(D2)
    β=T(Ra*E^2/(Pr*(last(r)-first(r))^3))
    θ=_sh_rescale(theta,+1)
    p=Dict{Tuple{Int,Int},Vector{T}}(); t=empty(p); f=empty(p); g=empty(p)
    active_in(d,am)=d!==nothing && any(abs(m)==am && any(!iszero,v) for x in (d.p,d.t) for ((l,m),v) in x)
    for am in 0:mmax
        active=any(l>0 && abs(m)==am && any(!iszero,v) for ((l,m),v) in θ) ||
               active_in(forcing,am) || active_in(induction,am)
        active || continue
        system=get!(systems,am) do
            _mean_momentum_system(r,D,D²,E,lmax,am,mechanical_bc;magnetic=magnetic)
        end
        x=_mean_momentum_solve(system,_mean_momentum_rhs(system,θ,forcing,r,β;induction=induction))
        all(isfinite,x) || error("Non-finite steady MHD mean state")
        n=length(system.modes)
        for (a,key) in enumerate(system.modes)
            rows=(a-1)*N+1:a*N
            p[key]=x[rows]; t[key]=x[n*N .+ rows]; f[key]=x[2n*N .+ rows]; g[key]=x[3n*N .+ rows]
        end
    end
    potentials(a,b)=SolenoidalMeanFlow(lmax,mmax,r,a,b,Dict(k=>D*v for (k,v) in a),
        Dict(k=>D²*v for (k,v) in a),Dict(k=>D*v for (k,v) in b))
    potentials(p,t),potentials(f,g)
end

"""Magnetic configuration `(Le, Em, B0_type, magnetic_bc)` of a mean-state constructor
after validation, or `nothing` without an imposed field."""
function _mean_magnetic(E, B0_type, Le, Pm, magnetic_bc, mechanical_bc)
    _check_magnetic_options(Pm, Le, B0_type, magnetic_bc, mechanical_bc)
    B0_type == no_field && return nothing
    return (Le=Le, Em=E/Pm, B0_type=B0_type, magnetic_bc=magnetic_bc)
end

"""The `magnetic` record a basic state keeps: what its stability problems must match."""
_magnetic_record(magnetic, Pm) = magnetic === nothing ? nothing :
    (B0_type=magnetic.B0_type, Le=magnetic.Le, Pm=Pm, magnetic_bc=magnetic.magnetic_bc)

"""Steady flow of a temperature state and, with an imposed field, its induced field."""
function _mean_flow_and_field(theta, r, D1, D2, E, Ra, Pr, lmax, mmax; mechanical_bc, magnetic)
    magnetic === nothing && return _steady_mean_flow(theta, r, D1, D2, E, Ra, Pr, lmax, mmax;
                                                     mechanical_bc=mechanical_bc), nothing
    return _steady_mean_mhd(theta, r, D1, D2, E, Ra, Pr, lmax, mmax; magnetic=magnetic,
                            mechanical_bc=mechanical_bc)
end

# Each fixed-temperature/Picard update has the same Stokes matrix. Reuse its
# row-scaled factorization throughout a nonlinear solve, without a global cache.
function _mean_momentum_system(r::Vector{T},D,D²,E,lmax,am,mechanical_bc;
                               magnetic=nothing) where T
    N=length(r); D4=D²*D²
    modes,C=_mean_coriolis(lmax,am,T); n=length(modes)
    nfields=magnetic===nothing ? 2 : 4
    A=zeros(T,nfields*n*N,nfields*n*N)
    for (a,(l,m)) in enumerate(modes)
        rows=(a-1)*N+1:a*N; rt=n*N .+ rows
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
            A[row,:] .= 0
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
    magnetic===nothing || _add_mean_magnetic_terms!(A,r,D,D²,lmax,modes,am,magnetic)
    if mechanical_bc==:stress_free && am==0
        # Bordered solve preserves both stress conditions; the multiplier
        # is the net axial torque and vanishes for radial thermal forcing.
        gauge=zeros(T,nfields*n*N); torque=copy(gauge)
        a=findfirst(==((1,0)),modes); rt=n*N .+ ((a-1)*N+1:a*N)
        gauge[rt] .= _mean_radial_weights(r).*r.^2
        torque[rt[2:end-1]] .= r[2:end-1]
        A=[A torque; gauge' zero(T)]
    end
    scales=vec(maximum(abs,A;dims=2))
    physical=Int[]
    for a in 1:n
        append!(physical,(a-1)*N .+ (3:N-2))
        append!(physical,n*N+(a-1)*N .+ (2:N-1))
        if magnetic!==nothing
            append!(physical,2n*N+(a-1)*N .+ (2:N-1))
            append!(physical,3n*N+(a-1)*N .+ (2:N-1))
        end
    end
    (modes=modes,matrix=A,scales=scales,factor=lu(A./scales),physical=physical)
end

"""
Add the imposed-field terms of the MHD mean state to the momentum system `A`, whose
unknowns are p, t, f, g in blocks of modes (f, g: native potentials of the induced
field b̄, b_r = q f/r², b_h = (f'/r)∇hY + (g/r) r̂×∇hY). The momentum rows gain the
Lorentz force Le² J×B₀ of the induced current J = ∇×b̄, on the side of the other
forces. The induction rows are the radial component and r times the toroidal
component of E_m∇²b̄ + ∇×W = 0:

    E_m (D² - q/r²) f / r = V(W),    E_m (D² - q/r²) g = R(W) - ∂r(r S(W)),

with the EMF W = U×B₀ (its radial, spheroidal and toroidal coefficients R, S, V) on
this side; `U×b̄` stays on the right. Insulating walls match a potential field,
r f' - (ℓ+1) f = 0 inside the inner wall and r f' + ℓ f = 0 outside the outer one, with
g = 0; a perfect conductor next to a no-slip wall has f = 0 and g' = 0.
"""
function _add_mean_magnetic_terms!(A,r::Vector{T},D,D²,lmax,modes,am,magnetic) where T
    N=length(r); n=length(modes)
    _,M=_mean_field_cross(lmax,am,T,magnetic.B0_type)
    Bf=Diagonal(_background_radial_factor(magnetic.B0_type,r))
    Le2=T(magnetic.Le)^2; Em=T(magnetic.Em)
    block(field,a)=(field-1)*n*N+(a-1)*N+1:(field-1)*n*N+a*N
    inner,outer=_magnetic_walls(magnetic.magnetic_bc)
    for (a,(l,m)) in enumerate(modes)
        q=T(l*(l+1))
        Rp,Rt,Rf,Rg=block(1,a),block(2,a),block(3,a),block(4,a)
        A[Rf,Rf] .+= Em .* ((D²-Diagonal(q ./ r.^2)) ./ r)
        A[Rg,Rg] .+= Em .* (D²-Diagonal(q ./ r.^2))
        for (b,(L,_)) in enumerate(modes)
            Q=T(L*(L+1))
            Cp,Ct,Cf,Cg=block(1,b),block(2,b),block(3,b),block(4,b)
            # Radial, spheroidal and toroidal parts of the basis fields as operators
            # on their potentials: u (or b) of p (f) and t (g), and J = ∇×b of f and g.
            R=Diagonal(Q ./ r.^2); S=D ./ r; U=Diagonal(one(T) ./ r)
            JV=(D²-Diagonal(Q ./ r.^2)) ./ r
            # Lorentz force -Le² B₀×J: J_V of f, and J_R = -R, J_S = -S of g.
            proj(i,X)=Le2 .* (Bf*X)
            Lf=(proj(a,M[a,2n+b] .* JV),proj(a,M[n+a,2n+b] .* JV),proj(a,M[2n+a,2n+b] .* JV))
            Lg=(proj(a,-(M[a,b] .* R+M[a,n+b] .* S)),proj(a,-(M[n+a,b] .* R+M[n+a,n+b] .* S)),
                proj(a,-(M[2n+a,b] .* R+M[2n+a,n+b] .* S)))
            A[Rp,Cf] .+= Lf[1]-D*(r .* Lf[2]); A[Rt,Cf] .+= Lf[3]
            A[Rp,Cg] .+= Lg[1]-D*(r .* Lg[2]); A[Rt,Cg] .+= Lg[3]
            # EMF W = -B₀×U of the flow, moved to this side of the induction rows.
            Wp=(Bf*(M[a,b] .* R+M[a,n+b] .* S),Bf*(M[n+a,b] .* R+M[n+a,n+b] .* S),
                Bf*(M[2n+a,b] .* R+M[2n+a,n+b] .* S))
            Wt=(Bf*(M[a,2n+b] .* U),Bf*(M[n+a,2n+b] .* U),Bf*(M[2n+a,2n+b] .* U))
            A[Rf,Cp] .+= Wp[3]; A[Rf,Ct] .+= Wt[3]
            A[Rg,Cp] .+= Wp[1]-D*(r .* Wp[2]); A[Rg,Ct] .+= Wt[1]-D*(r .* Wt[2])
        end
        # The momentum wall rows keep only their velocity conditions.
        for row in (first(Rp),first(Rp)+1,last(Rp)-1,last(Rp),first(Rt),last(Rt))
            A[row,2n*N+1:4n*N] .= 0
        end
        for (row,node,wall) in ((first(Rf),1,inner),(last(Rf),N,outer))
            A[row,:] .= 0
            if wall===:insulating
                A[row,Rf] .= r[node] .* D[node,:]
                A[row,Rf[node]] += node==1 ? -(l+1) : l
            else
                A[row,Rf[node]]=1
            end
        end
        for (row,node,wall) in ((first(Rg),1,inner),(last(Rg),N,outer))
            A[row,:] .= 0
            wall===:insulating ? (A[row,Rg[node]]=1) : (A[row,Rg] .= D[node,:])
        end
    end
    A
end

function _mean_momentum_rhs(system,theta,inertia,r,β;induction=nothing)
    T=eltype(r); N=length(r); n=length(system.modes)
    f=zeros(T,size(system.matrix,1))
    for (a,key) in enumerate(system.modes)
        rows=(a-1)*N+1:a*N; rt=n*N .+ rows
        haskey(theta,key) && (f[rows] .= β.*r.*theta[key])
        if inertia!==nothing
            f[rows] .+= inertia.p[key]
            f[rt] .+= inertia.t[key]
        end
        f[[first(rows),first(rows)+1,last(rows)-1,last(rows),first(rt),last(rt)]] .= 0
        # EMF U×b̄ of the induced field: V(W) on the f rows, R(W) - ∂r(rS(W)) on g.
        if induction!==nothing
            rf=2n*N .+ rows; rg=3n*N .+ rows
            f[rf] .+= induction.t[key]; f[rg] .+= induction.p[key]
            f[[first(rf),last(rf),first(rg),last(rg)]] .= 0
        end
    end
    f
end

function _mean_momentum_solve(system,f)
    x=system.factor \ (f./system.scales)
    # Wall rows such as the stress-free condition are O(N⁴) larger than the
    # interior equations, so one refinement step brings them from LU round-off
    # down to the tolerance the stability operator checks.
    x.+=system.factor \ ((f.-system.matrix*x)./system.scales)
    x
end

"""Poloidal potentials driven by buoyancy alone: column `(b-1)N+j` holds the
profiles of every mode of `system`, stacked by mode, for a unit orthonormal
temperature coefficient of mode `b` at node `j`. Wall nodes carry boundary
conditions instead of buoyancy, as in `_mean_momentum_rhs`."""
function _mean_buoyancy_response(system,r,β)
    T=eltype(r); N=length(r); n=length(system.modes)
    F=zeros(T,size(system.matrix,1),n*N)
    for a in 1:n, j in 3:N-2
        F[(a-1)*N+j,(a-1)*N+j]=β*r[j]
    end
    _mean_momentum_solve(system,F)[1:n*N,:]
end


# Wall conditions of a mean-temperature mode: one symbol for the outer wall with a
# fixed inner temperature, or an (inner, outer) pair.
_mean_bc_kinds(kind::Symbol) = (:fixed_temperature, kind)
_mean_bc_kinds(kinds::Tuple{Symbol,Symbol}) = kinds

# Implicit advection-diffusion at fixed velocity, in the orthonormal scalar
# basis. Solving transport implicitly avoids the diffusion-only Picard update
# becoming unstable as the Peclet number increases.
#
# Optional `feedback` = (gradient, responses, forced) also makes the advection of
# the spherical mean temperature, u_r ∂r T₀₀, implicit: u_r comes from the poloidal
# flow that the new temperature drives, p = forced + G θ, with `responses[am]` =
# (modes, G) from `_mean_buoyancy_response`, `forced` the flow driven by the
# frozen inertia, and `gradient` = ∂r T₀₀ of the previous iterate. Lagging that
# linear buoyancy feedback instead limits a Picard iteration to Rayleigh numbers
# well below onset, whatever the forcing amplitude.
function _mean_temperature_step(flow::SolenoidalMeanFlow{T},D1,D2,κ,bc;feedback=nothing) where T
    g=sh_grid(flow.lmax,flow.mmax,T); r=flow.r; N=length(r)
    modes=[(l,m) for m in -g.mmax:g.mmax for l in abs(m):g.lmax]
    n=length(modes)
    (; Y, Hθ, Hφ, w) = _sh_basis_samples(g, modes)
    A=zeros(T,n*N,n*N); f=zeros(T,n*N)
    for (a,(l,m)) in enumerate(modes)
        ix=(a-1)*N+1:a*N
        A[ix,ix] .= κ.*(D2+2 .* D1 ./ r-Diagonal(l*(l+1) ./ r.^2))
    end
    b00=feedback===nothing ? 0 : findfirst(==((0,0)),modes)
    for i in 2:N-1
        ur,uθ,uφ=_mean_flow_grid(flow,i,g)
        Ar=Y' * ((w.*vec(ur)).*Y)
        Ah=Y' * ((w.*vec(uθ)).*Hθ+(w.*vec(uφ)).*Hφ) ./ r[i]
        for a in 1:n,b in 1:n
            row=(a-1)*N+i; cols=(b-1)*N+1:b*N
            # With feedback, u_r ∂r T₀₀ enters below with the implicit velocity.
            b==b00 && a!=b00 || (A[row,cols] .-= Ar[a,b].*D1[i,:])
            A[row,cols[i]] -= Ah[a,b]
        end
    end
    if feedback!==nothing
        for (am,(rmodes,G)) in feedback.responses
            h=[findfirst(==(k),modes) for k in rmodes]
            # Projection of u_r Y₀₀ onto each Y: diagonal for an orthonormal basis.
            K=Y[:,h]' * ((w.*Y[:,b00]).*Y[:,h])
            for a in eachindex(h), c in eachindex(h), i in 2:N-1
                coef=K[a,c]*rmodes[c][1]*(rmodes[c][1]+1)/r[i]^2*feedback.gradient[i]
                iszero(coef) && continue
                row=(h[a]-1)*N+i; grow=(c-1)*N+i
                for b in eachindex(h)
                    A[row,(h[b]-1)*N+1:h[b]*N] .-= coef.*G[grow,(b-1)*N+1:b*N]
                end
                haskey(feedback.forced,rmodes[c]) && (f[row]+=coef*feedback.forced[rmodes[c]][i])
            end
        end
    end
    for (a,key) in enumerate(modes)
        ix=(a-1)*N+1:a*N
        inner,outer,kind=bc[key]; factor=_sh_nf_to_orth_factor(key...,T)
        for (row,wall_kind,value) in ((first(ix),first(_mean_bc_kinds(kind)),inner),
                                      (last(ix),last(_mean_bc_kinds(kind)),outer))
            A[row,:] .= 0; f[row]=value*factor
            if wall_kind==:fixed_temperature
                A[row,row]=1
            else
                A[row,ix] .= D1[row-first(ix)+1,:]
            end
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
    # Chebyshev–Gauss–Lobatto weights; one(x)/2 keeps Float32 data in Float32.
    weights=[(iseven(i) ? -one(x) : one(x))*(i in (1,length(r)) ? one(x)/2 : one(x))/(x-r[i]) for i in eachindex(r)]
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
    g=SHGrid{T}(L,M,μ,T[2],T[φ])
    at(d)=Dict(k=>T[_mean_barycentric(flow.r,v,T(r))] for (k,v) in d)
    f=SolenoidalMeanFlow(L,M,T[r],at(flow.p),at(flow.t),at(flow.dp),at(flow.d2p),at(flow.dt))
    ur,uθ,uφ=_mean_flow_grid(f,1,g)
    (ur=only(ur),utheta=only(uθ),uphi=only(uφ))
end

"""
    mean_temperature(bs, r, θ, φ=0)

Evaluate the temperature ``\\bar T(r,θ,φ)`` of a `BasicState` or `BasicState3D`,
with spectral radial interpolation. θ is colatitude. With
[`mean_flow_velocity`](@ref) this gives the physical fields of a constructed state,
for example on meridional or equatorial sections.
"""
function mean_temperature(bs, r::Real, θ::Real, φ::Real=0)
    ascending=first(bs.r)<=last(bs.r)
    rgrid=ascending ? bs.r : reverse(bs.r)
    first(rgrid)<=r<=last(rgrid) || throw(ArgumentError("Radius is outside the shell"))
    0<=θ<=π || throw(ArgumentError("Colatitude must satisfy 0 ≤ θ ≤ π"))
    T=eltype(bs.r)
    # Stored coefficients use the public no-factorial normalization; axisymmetric
    # states key them by ℓ alone.
    c=Dict{Tuple{Int,Int},T}()
    for (key,v) in bs.theta_coeffs
        l,m=key isa Integer ? (Int(key),0) : key
        c[(l,m)]=_sh_nf_to_orth_factor(l,m,T)*_mean_barycentric(rgrid,ascending ? v : reverse(v),T(r))
    end
    isempty(c) && return zero(T)
    g=SHGrid{T}(maximum(first,keys(c)),maximum(k->abs(last(k)),keys(c)),T[cos(θ)],T[2],T[φ])
    only(sh_synthesize(c,g))
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

# Shared body of the legacy component-projection wrappers (solve_meridional_*!,
# solve_thermal_wind_*!): solve the viscous mean flow once and copy selected
# scalar component projections into the caller's dictionaries. `targets` maps
# component names (:ur, :utheta, :uphi, :dur, :dutheta, :duphi) to destination
# dictionaries; `keep` filters (l,m) keys and `key` maps them to destination
# keys. With `reset`, destinations are emptied first.
function _project_mean_flow!(targets::NamedTuple, theta, r, D1, D2, E, Ra, Pr, lmax, mmax;
                             mechanical_bc=:no_slip, keep=Returns(true), key=identity,
                             reset=true)
    flow=_steady_mean_flow(theta,r,D1,D2,E,Ra,Pr,lmax,mmax;mechanical_bc=mechanical_bc)
    ur,utheta,uphi,dur,dutheta,duphi=_mean_flow_components(flow)
    fields=(ur=ur,utheta=utheta,uphi=uphi,dur=dur,dutheta=dutheta,duphi=duphi)
    for (name,target) in pairs(targets)
        reset && empty!(target)
        for (k,v) in fields[name]
            keep(k) && (target[key(k)]=v)
        end
    end
    flow
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
