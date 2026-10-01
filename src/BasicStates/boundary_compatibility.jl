# Check stationary-wall compatibility from the physical mean velocity, not
# from the constructor that produced it. Zero flow satisfies either wall type.

function _mean_bc_derivative_rows(grid)
    r = grid.r
    # Match _mean_barycentric exactly, including on rounded or custom grids.
    # Its fixed CGL weights define a rational interpolant when nodes depart
    # from CGL locations; general polynomial weights would check another field.
    T = eltype(r)
    w = T[(isodd(i) ? one(T) : -one(T)) *
          (i in (1,length(r)) ? one(T)/2 : one(T)) for i in eachindex(r)]
    D = zeros(eltype(r), 2, length(r))
    for (wall, i) in enumerate((1, length(r)))
        for j in eachindex(r)
            j == i && continue
            D[wall,j] = w[j] / (w[i] * (r[i] - r[j]))
        end
        D[wall,i] = -sum(D[wall,:])
    end
    all(isfinite,D) || throw(ArgumentError("Basic-state wall derivatives are not representable"))
    D
end

function _mean_bc_accept(residual, scale, roundoff, precision, bc, wall, channel)
    # A small physical floor accommodates traces from an ill-conditioned
    # collocation solve without letting one large velocity channel mask another.
    # The derivative term is local; it is not an N^4 multiple of the full flow.
    physical_floor = 4sqrt(precision) * scale
    isfinite(roundoff) && roundoff <= physical_floor || throw(ArgumentError(
        "Cannot validate basic-state $(channel) at the $(wall) wall: wall derivative " *
        "roundoff $(roundoff) exceeds the allowed physical uncertainty $(physical_floor). " *
        "Increase numerical precision or use a better-conditioned radial grid."))
    tolerance = physical_floor + roundoff
    all(isfinite,(residual,scale,roundoff,tolerance)) && residual <= tolerance || throw(ArgumentError(
        "Basic-state velocity is incompatible with stationary $(bc) at the $(wall) wall: " *
        "$(channel) residual $(residual) exceeds numerical tolerance $(tolerance) " *
        "(channel velocity scale $(scale)). Construct a mean flow satisfying the requested " *
        "mechanical boundary conditions; if these were used already, increase numerical " *
        "precision or resolution."))
    nothing
end

function _check_native_mean_bc(walls, bs, ::Type{T}, precision) where T
    f = bs.flow
    grid = _resolution_grid(f.r,T)
    _resolution_same_shell(T.(bs.r),grid.r,16precision) || throw(ArgumentError(
        "Native velocity and basic-state radii must occupy the same shell"))
    p = _resolution_coefficients(f.p,grid,T;lmax=f.lmax,mmax=f.mmax)
    t = _resolution_coefficients(f.t,grid,T;lmax=f.lmax,mmax=f.mmax)
    dp = _resolution_coefficients(f.dp,grid,T;lmax=f.lmax,mmax=f.mmax)
    keys(p)==keys(t)==keys(dp) || throw(ArgumentError(
        "Native mean flow must supply matching p, t and dp harmonic modes"))
    isempty(p) && return nothing
    r=grid.r; gap=last(r)-first(r); D=_mean_bc_derivative_rows(grid)
    # The supplied arrays define the represented field. Source precision only
    # sets the small physical floor above; differentiating their interpolants
    # is performed in T and must not admit source-eps times N² strain errors.
    arithmetic_precision=eps(T)
    scales=zeros(T,3,length(r)); residual=zeros(T,3,2); errors=zero(residual)
    for (key,v) in p
        q=T(key[1]*(key[1]+1)); sq=sqrt(q); dv=dp[key]; tv=t[key]
        scales[1,:] .= hypot.(scales[1,:],q.*v./r.^2)
        scales[2,:] .= hypot.(scales[2,:],sq.*dv./r)
        scales[3,:] .= hypot.(scales[3,:],sq.*tv./r)
        for (wall,i) in enumerate((1,length(r)))
            R=q*v[i]/r[i]^2
            residual[1,wall] = hypot(residual[1,wall],R)
            if walls[wall]===:no_slip
                S=sq*dv[i]/r[i]; V=sq*tv[i]/r[i]
                errS=errV=zero(T)
            else
                # Differentiate the dp/t interpolants used by the physical
                # velocity evaluator. This checks strain without trusting a
                # supplied d2p that could disagree with the velocity itself.
                S=gap*sq*(dot(D[wall,:],dv)/r[i]-2dv[i]/r[i]^2+q*v[i]/r[i]^3)
                V=gap*sq*(dot(D[wall,:],tv)/r[i]-2tv[i]/r[i]^2)
                errS=64arithmetic_precision*gap*sq*(dot(abs.(D[wall,:]),abs.(dv))/r[i]+
                    2abs(dv[i])/r[i]^2+q*abs(v[i])/r[i]^3)
                errV=64arithmetic_precision*gap*sq*(dot(abs.(D[wall,:]),abs.(tv))/r[i]+2abs(tv[i])/r[i]^2)
            end
            residual[2,wall]=hypot(residual[2,wall],S); residual[3,wall]=hypot(residual[3,wall],V)
            errors[2,wall]=hypot(errors[2,wall],errS); errors[3,wall]=hypot(errors[3,wall],errV)
        end
    end
    channel_scales=vec(maximum(scales;dims=2))
    for (wall,name) in enumerate((:inner,:outer)), channel in 1:3
        label=channel==1 ? "radial velocity" : walls[wall]===:no_slip ?
            (channel==2 ? "poloidal velocity" : "toroidal velocity") :
            (channel==2 ? "poloidal tangential strain × shell gap" : "toroidal tangential strain × shell gap")
        _mean_bc_accept(residual[channel,wall],channel_scales[channel],
            errors[channel,wall],precision,walls[wall],name,label)
    end
    nothing
end

function _check_component_mean_bc(walls, bs, grid, ::Type{T}, precision) where T
    r=grid.r; N=length(r); L=bs.lmax_bs; M=bs isa BasicState ? 0 : bs.mmax_bs
    convert(d)=_resolution_coefficients(d,grid,T;scalar=true,lmax=L,mmax=M)
    fields=map(convert,(bs.ur_coeffs,bs.utheta_coeffs,bs.uphi_coeffs))
    D=_mean_bc_derivative_rows(grid); gap=last(r)-first(r); arithmetic_precision=eps(T)
    scales=[maximum(norm(T[v[i] for v in values(d)]) for i in 1:N) for d in fields]
    for (wall,name) in enumerate((:inner,:outer))
        i=wall==1 ? 1 : N
        trace(d)=Dict{Tuple{Int,Int},T}(k=>v[i] for (k,v) in d)
        radial=trace(fields[1])
        _mean_bc_accept(norm(collect(values(radial))),scales[1],zero(T),precision,
            walls[wall],name,"radial velocity")
        if walls[wall]===:no_slip
            for channel in 2:3
                _mean_bc_accept(norm(collect(values(trace(fields[channel])))),scales[channel],zero(T),precision,
                    walls[wall],name,channel==2 ? "latitudinal velocity" : "azimuthal velocity")
            end
        else
            g=sh_grid(L,M,T)
            for channel in 2:3
                d=fields[channel]
                # Check the represented velocity itself. Stale derivative
                # dictionaries must not hide an incompatible wall profile.
                coeffs=Dict{Tuple{Int,Int},T}(k=>dot(D[wall,:],v)-v[i]/r[i] for (k,v) in d)
                angular=sh_synthesize(radial,g;Yf=channel==2 ? _sh_dYθ : _sh_dYφ_over_sin)/r[i]
                traction=sh_synthesize(coeffs,g)+angular
                residual=gap*norm(sqrt.(g.w).*traction)*sqrt(2T(pi)/length(g.φ))
                error=64arithmetic_precision*gap*norm(T[dot(abs.(D[wall,:]),abs.(v))+abs(v[i])/r[i] for v in values(d)])
                _mean_bc_accept(residual,scales[channel],error,precision,walls[wall],name,
                    channel==2 ? "latitudinal tangential strain × shell gap" : "azimuthal tangential strain × shell gap")
            end
        end
    end
    nothing
end

"""Require actual mean velocity/strain to satisfy stationary mechanical walls.
Residuals use separate physical velocity channels and local derivative roundoff;
constructor labels and scalar projections of a native flow are not consulted."""
function _check_basic_state_mechanical_bc(bc, bs)
    bs===nothing && return nothing
    walls=bc isa Symbol ? (bc,bc) : bc
    walls isa Tuple{Symbol,Symbol} && all(in((:no_slip,:stress_free)),walls) ||
        throw(ArgumentError("Invalid mean-flow mechanical boundary conditions $(repr(bc))"))
    bs isa Union{BasicState,BasicState3D} || throw(ArgumentError("Expected a BasicState or BasicState3D"))
    bs.Nr==length(bs.r) || throw(DimensionMismatch("Basic-state Nr does not match its radial grid"))
    T=promote_type(Float64,float(eltype(bs.r)))
    precision=T(eps(float(eltype(bs.r))))
    grid=_resolution_grid(bs.r,T)
    bs.flow===nothing ? _check_component_mean_bc(walls,bs,grid,T,precision) :
        _check_native_mean_bc(walls,bs,T,precision)
    nothing
end
