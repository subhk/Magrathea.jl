"""
    mean_flow_resolution(coarse, fine; rtol=1e-3, atol=0)

Compare two basic states on the same spherical shell in the physical volume
`L²` norm, using native velocity potentials and the union of their harmonics.
Radial grids and angular truncations may differ; missing harmonics are zero.

Returns `(converged, velocity=(total, radial, poloidal, toroidal),
temperature=(total, anomaly))`. Each metric contains `converged`, `error`
(the norm of `coarse - fine`), `scale` (the fine-state norm), `tolerance`
(`atol + rtol*scale`), and `relative_error`. The poloidal and toroidal velocity
metrics measure the two horizontal vector-harmonic channels; radial velocity
is checked separately. Temperature anomaly excludes the spherical monopole.
All six metrics must pass, so a strong zonal flow or mean temperature cannot
hide an unresolved weaker component. A zero reference norm has relative error
zero for an exact match and `Inf` otherwise; `atol` still applies.

The quadrature uses both states' radial interpolants and preserves their
promoted floating-point precision. A nonzero velocity requires native
`SolenoidalMeanFlow` potentials; legacy scalar component projections are not
used. This comparison does not establish convergence from a single grid pair:
refine radial and angular resolution independently and check successive pairs.
No ordering of the two resolutions is required; `fine` defines the reference.
"""
function mean_flow_resolution(coarse::Union{BasicState,BasicState3D},
                              fine::Union{BasicState,BasicState3D};
                              rtol::Real=1e-3, atol::Real=0)
    isfinite(rtol) && rtol >= 0 || throw(ArgumentError("rtol must be finite and nonnegative"))
    isfinite(atol) && atol >= 0 || throw(ArgumentError("atol must be finite and nonnegative"))
    T = float(promote_type(eltype(coarse.r), eltype(fine.r)))
    a = _resolution_state(coarse, T)
    b = _resolution_state(fine, T)
    shell_tol = 16max(eps(float(eltype(coarse.r))), eps(float(eltype(fine.r))))
    _resolution_same_shell(a.grid.r, b.grid.r, shell_tol) || throw(ArgumentError(
        "The states must occupy the same spherical shell"))

    modes = sort!(collect(union(keys(a.p), keys(a.dp), keys(a.t),
                               keys(b.p), keys(b.dp), keys(b.t))))
    thermal_modes = sort!(collect(union(keys(a.temperature), keys(b.temperature))))
    nsource = maximum(length(g.r) for g in (a.grid,b.grid,a.flow_grid,b.flow_grid))
    previous = nothing
    energies = nothing
    nquad = max(16, nsource)
    # Log-radius quadrature resolves the inner-wall 1/r factors even in thick
    # shells. Successive rules check integration error independently of the
    # user's spatial-convergence tolerance.
    for attempt in 1:8
        energies = _resolution_integrals(a,b,modes,thermal_modes,nquad,T)
        if previous !== nothing && _resolution_quadrature_agrees(previous,energies,nsource,T)
            break
        end
        attempt == 8 && error("Mean-flow norm quadrature did not converge; check the radial grids and data")
        previous = energies
        nquad *= 2
    end
    errors, scales = energies
    metric(indices) = _resolution_metric(sqrt(sum(errors[indices])),
        sqrt(sum(scales[indices])), rtol, atol)
    velocity = (total=metric(1:3), radial=metric(1:1),
                poloidal=metric(2:2), toroidal=metric(3:3))
    temperature = (total=metric(4:5), anomaly=metric(5:5))
    converged = all(x.converged for x in values(velocity)) &&
                all(x.converged for x in values(temperature))
    return (; converged, velocity, temperature)
end

function _resolution_metric(error::T, scale::T, rtol, atol) where T
    tolerance = atol + rtol * scale
    relative_error = iszero(scale) ? (iszero(error) ? zero(T) : T(Inf)) : error / scale
    (; converged=error <= tolerance, error, scale, tolerance, relative_error)
end

_resolution_same_shell(a,b,tol) = isapprox(first(a),first(b); rtol=tol,atol=0) &&
                                  isapprox(last(a),last(b); rtol=tol,atol=0)

function _resolution_grid(radial, ::Type{T}) where T
    length(radial) >= 2 || throw(ArgumentError("A radial grid must contain at least two points"))
    r = T.(radial)
    all(isfinite,r) && first(r)>0 && all(>(zero(T)),diff(r)) || throw(ArgumentError(
        "Radial grids must be finite, positive, and strictly increasing"))
    # General barycentric weights also handle rounded Float32 grids and custom
    # polynomial grids. Logarithms avoid overflowing products of node gaps.
    logs = T[-sum(log(abs(r[j]-r[k])) for k in eachindex(r) if k!=j) for j in eachindex(r)]
    offset = maximum(logs)
    weights = T[(isodd(length(r)-j) ? -one(T) : one(T))*exp(logs[j]-offset) for j in eachindex(r)]
    all(x->isfinite(x) && !iszero(x),weights) || throw(ArgumentError("Radial interpolation weights are not representable"))
    (; r, weights)
end

function _resolution_coefficients(d, grid, ::Type{T}; scalar=false, lmax, mmax) where T
    out = Dict{Tuple{Int,Int},Vector{T}}()
    for (key,v) in d
        lm = key isa Integer ? (Int(key),0) : key
        lm isa Tuple{Int,Int} || throw(ArgumentError("Harmonic keys must be l or (l,m)"))
        l,m = lm
        (scalar ? 0 : 1) <= l <= lmax && abs(m)<=min(l,mmax) || throw(ArgumentError(
            "Harmonic ($l,$m) is outside the declared resolution"))
        length(v)==length(grid.r) || throw(DimensionMismatch("Coefficient ($l,$m) does not match its radial grid"))
        all(isfinite,v) || throw(ArgumentError("Harmonic coefficients must be finite"))
        values = T.(v)
        if scalar
            # Convert public no-factorial coefficients to real orthonormal SH
            # without forming an overflowing factorial or normalization factor.
            for k in (l-abs(m)+1):(l+abs(m))
                values .*= sqrt(T(k))
            end
        end
        all(isfinite,values) || throw(ArgumentError("Normalized coefficients are not representable"))
        out[lm] = values
    end
    out
end

function _resolution_state(bs, ::Type{T}) where T
    bs.Nr==length(bs.r) || throw(DimensionMismatch("Basic-state Nr does not match its radial grid"))
    grid = _resolution_grid(bs.r,T)
    mmax = bs isa BasicState ? 0 : bs.mmax_bs
    0 <= mmax <= bs.lmax_bs || throw(ArgumentError("Invalid basic-state harmonic resolution"))
    temperature = _resolution_coefficients(bs.theta_coeffs,grid,T;
        scalar=true,lmax=bs.lmax_bs,mmax=mmax)
    flow = bs.flow
    if flow === nothing
        for d in (bs.ur_coeffs,bs.utheta_coeffs,bs.uphi_coeffs)
            legacy = _resolution_coefficients(d,grid,T;scalar=true,lmax=bs.lmax_bs,mmax=mmax)
            any(v->any(!iszero,v),values(legacy)) && throw(ArgumentError(
                "Comparing nonzero velocities requires native mean-flow potentials"))
        end
        empty = Dict{Tuple{Int,Int},Vector{T}}()
        return (; grid, temperature, flow_grid=grid, p=empty, dp=empty, t=empty)
    end
    0 <= flow.mmax <= flow.lmax || throw(ArgumentError("Invalid native-flow harmonic resolution"))
    flow_grid = _resolution_grid(flow.r,T)
    _resolution_same_shell(grid.r,flow_grid.r,16eps(T)) || throw(ArgumentError(
        "Native velocity and temperature must occupy the same shell"))
    p = _resolution_coefficients(flow.p,flow_grid,T;lmax=flow.lmax,mmax=flow.mmax)
    dp = _resolution_coefficients(flow.dp,flow_grid,T;lmax=flow.lmax,mmax=flow.mmax)
    t = _resolution_coefficients(flow.t,flow_grid,T;lmax=flow.lmax,mmax=flow.mmax)
    for (key,v) in p
        haskey(dp,key) || all(iszero,v) || throw(ArgumentError("Missing radial derivative for poloidal mode $key"))
    end
    (; grid, temperature, flow_grid, p, dp, t)
end

function _resolution_interpolation(grid, r)
    T = eltype(r)
    M = zeros(T,length(r),length(grid.r))
    for (i,x) in enumerate(r)
        node = findfirst(==(x),grid.r)
        if node !== nothing
            M[i,node] = one(T)
        else
            row = view(M,i,:)
            row .= grid.weights ./ (x .- grid.r)
            row ./= sum(row)
        end
    end
    all(isfinite,M) || throw(ArgumentError("The radial grid is too ill-conditioned for interpolation"))
    M
end

# Generic-precision Gauss-Legendre rule: do not route BigFloat data through the
# Float64-only plotting-grid helper.
function _resolution_gauss(n, ::Type{T}) where T
    x = zeros(T,n); w = similar(x)
    for i in 1:div(n+1,2)
        z = cos(T(π)*(T(i)-T(1)/4)/(T(n)+T(1)/2))
        derivative = zero(T)
        for iteration in 1:100
            p,pold = one(T),zero(T)
            for k in 1:n
                p,pold = ((2k-1)*z*p-(k-1)*pold)/k,p
            end
            derivative = n*(z*p-pold)/(z*z-one(T))
            correction = p/derivative
            z -= correction
            abs(correction)<=4eps(T) && break
            iteration==100 && error("Radial quadrature root did not converge")
        end
        # Re-evaluate the derivative at the accepted root for its weight.
        p,pold = one(T),zero(T)
        for k in 1:n
            p,pold = ((2k-1)*z*p-(k-1)*pold)/k,p
        end
        derivative = n*(z*p-pold)/(z*z-one(T))
        x[i]=-z; x[n+1-i]=z
        w[i]=2/((one(T)-z*z)*derivative^2); w[n+1-i]=w[i]
    end
    x,w
end

function _resolution_integrals(a,b,modes,thermal_modes,nquad,::Type{T}) where T
    x,w = _resolution_gauss(nquad,T)
    inner,outer = first(b.grid.r),last(b.grid.r)
    stretch = log(outer/inner)/2
    r = inner .* exp.((x .+ one(T)).*stretch)
    volume_weights = w .* stretch .* r.^3
    Ma = _resolution_interpolation(a.flow_grid,r)
    Mb = _resolution_interpolation(b.flow_grid,r)
    Ta = _resolution_interpolation(a.grid,r)
    Tb = _resolution_interpolation(b.grid,r)
    errors = zeros(T,5); scales = zeros(T,5)
    profile(M,d,key) = haskey(d,key) ? M*d[key] : zeros(T,nquad)
    for key in modes
        l,m = key; q = T(l)*T(l+1)
        for (i,field,factor) in ((1,:p,q ./ r.^2), (2,:dp,sqrt(q) ./ r), (3,:t,sqrt(q) ./ r))
            va = profile(Ma,getproperty(a,field),key) .* factor
            vb = profile(Mb,getproperty(b,field),key) .* factor
            errors[i] += dot(volume_weights,abs2.(va .- vb))
            scales[i] += dot(volume_weights,abs2.(vb))
        end
    end
    for key in thermal_modes
        i = key[1]==0 ? 4 : 5
        va = profile(Ta,a.temperature,key); vb = profile(Tb,b.temperature,key)
        errors[i] += dot(volume_weights,abs2.(va .- vb))
        scales[i] += dot(volume_weights,abs2.(vb))
    end
    all(isfinite,errors) && all(isfinite,scales) || throw(ArgumentError(
        "Physical squared norms exceed the coefficient precision's representable range"))
    errors,scales
end

function _resolution_quadrature_agrees(previous,current,nsource,::Type{T}) where T
    e0,s0 = previous; e1,s1 = current
    tol = 64eps(T)
    for i in eachindex(e0)
        abs(s1[i]-s0[i]) <= tol*max(s0[i],s1[i]) || return false
        # Interpolation perturbs the difference by δ*||fine||. Its squared
        # norm therefore has both a cross term 2δ*||difference||*||fine|| and
        # a δ² term. Keeping only δ² cannot converge for two close but unequal
        # states, because additional quadrature points do not remove roundoff.
        δ = T(32)*T(nsource)*eps(T)
        reference = max(s0[i],s1[i])
        difference = max(e0[i],e1[i])
        floor = (2δ*sqrt(reference))*sqrt(difference)+δ^2*reference
        abs(e1[i]-e0[i]) <= tol*max(e0[i],e1[i])+floor || return false
    end
    true
end
