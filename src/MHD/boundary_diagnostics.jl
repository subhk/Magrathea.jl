"""
    magnetic_boundary_residuals(op::MHDStabilityOperator, eigenvectors;
                                rtol=1e-6, atol=0, Nθ=nothing)

Check the physical magnetic boundary conditions of full MHD coefficient vectors
(one vector, or one vector per matrix column). This evaluates the fields and
their radial derivatives directly, independently of the assembled boundary rows.
It also retains angular electric-field components outside the trial space, which
can reveal underresolution in either a tau or a weak Galerkin solution.

The report contains `applicable`, `checked`, `passed`, `per_mode`, `maximum`, and
the tolerances and quadrature size. Each mode has `checked`, `passed`, and wall
reports `inner`, `outer`, and optionally `mantle_outer`. Wall channels are
`normal_field`, `tangential_field`, and `tangential_electric`; an inapplicable
channel is `nothing`. Each metric contains `residual`, `scale`,
`relative_residual`, `tolerance`, and `passed`. `maximum` identifies the mode with
the largest relative residual in each wall/channel, and its `passed` flag checks
that channel in every checked mode.

Residuals are surface L2 norms of Bnormal, the tangential field jump, or the
tangential electric field/jump `E = Em*curl(b) - u×B0`. Insulating exteriors use
the unique regular/decaying vacuum field with matching Bnormal. Conducting core
and mantle interfaces use the package's equal-permeability model, so continuity
of tangential H is equivalent to continuity of tangential B.

The scale includes both wall terms and an independent bulk RMS field reference;
in particular, a vanishing wall current does not make roundoff a unit relative
error. No arbitrary amplitude floor is added, and multiplying a vector by a
constant preserves its relative residuals. `atol` is in the absolute surface-norm
units of each channel. An identically zero vector is unchecked and fails; a
nonzero thermal-only vector can satisfy the magnetic conditions trivially.
No-field problems and empty/zero-row distributed eigenvector matrices are
unchecked. These diagnostics do not replace convergence with radial and angular
refinement or impose extra equations on the weak formulation.
"""
function magnetic_boundary_residuals(op::MHDStabilityOperator,
                                     eigenvectors::AbstractVecOrMat;
                                     rtol::Real=1e-6, atol::Real=0,
                                     Nθ::Union{Int,Nothing}=nothing)
    isfinite(rtol) && rtol >= 0 || throw(ArgumentError("rtol must be finite and nonnegative"))
    isfinite(atol) && atol >= 0 || throw(ArgumentError("atol must be finite and nonnegative"))
    V = eigenvectors isa AbstractVector ? reshape(eigenvectors, :, 1) : eigenvectors
    nθ = Nθ === nothing ? 2op.params.lmax + 6 : Nθ
    nθ >= op.params.lmax + 2 || throw(ArgumentError(
        "Nθ must be at least lmax + 2 to resolve the full boundary electric field"))
    applicable = !isempty(op.ll_f) || !isempty(op.ll_g)
    if !applicable || isempty(V)
        return (; applicable, checked=false, passed=false, rtol, atol, Nθ=nθ,
                  per_mode=NamedTuple[], maximum=nothing)
    end
    size(V, 1) == op.matrix_size || throw(DimensionMismatch(
        "Magnetic boundary diagnostics require $(op.matrix_size) full coefficients per vector"))
    all(isfinite, V) || throw(ArgumentError("Eigenvectors must contain only finite coefficients"))
    T = float(promote_type(typeof(op.params.E), typeof(real(zero(eltype(V))))))
    angular = _mhd_boundary_angles(op, nθ, T)
    index = _mhd_index_map(op)
    reports = [_mhd_boundary_mode(op, view(V, :, k), index, angular, rtol, atol)
               for k in axes(V, 2)]
    checked = all(r -> r.checked, reports)
    return (; applicable, checked, passed=checked && all(r -> r.passed, reports),
              rtol, atol, Nθ=nθ, per_mode=reports,
              maximum=_mhd_boundary_maximum(reports))
end

function _mhd_boundary_angles(op, n, ::Type{T}) where T
    μ, w = _resolution_gauss(n, T)
    m = op.params.m; L = op.params.lmax
    # Neighboring orders provide dθ without differentiating sampled fields.
    q = Dict(a => _normalized_legendre_table(a, L, μ)
             for a in max(0, m-1):min(L, m+1))
    sinθ = sqrt.(one(T) .- μ.^2)
    basis = Dict{Int,NamedTuple}()
    for l in max(1, m):L
        y = q[m][l-m+1, :] ./ sqrt(T(2l+1))
        h = if m == 0
            sqrt(T(l*(l+1))) .* q[1][l, :] ./ sqrt(T(2l+1))
        else
            qp = m < l ? sqrt(T((l-m)*(l+m+1))) .* q[m+1][l-m, :] : zeros(T,n)
            qm = sqrt(T((l+m)*(l-m+1))) .* q[m-1][l-m+2, :]
            (qp .- qm) ./ (2sqrt(T(2l+1)))
        end
        v = (Complex{T}(0, m) .* y) ./ sinθ
        basis[l] = (; y, h, v)
    end
    return (; μ, w, basis)
end

# Differentiate the Chebyshev polynomial recurrence, rather than applying tau
# endpoint functionals or differentiating an undersampled reconstructed field.
function _mhd_boundary_jet(coeff, r, ri, ro; core_l=nothing)
    x = core_l === nothing ? 2(r-ri)/(ro-ri)-1 : 2(r/ri)^2-1
    a, b = (one(x), zero(x), zero(x)), (x, one(x), zero(x))
    f, d, dd = coeff[1], zero(coeff[1]), zero(coeff[1])
    if length(coeff) > 1
        f += coeff[2]*x
        d += coeff[2]
    end
    for n in 2:length(coeff)-1
        c = (2x*b[1]-a[1], 2b[1]+2x*b[2]-a[2], 4b[2]+2x*b[3]-a[3])
        f += coeff[n+1]*c[1]; d += coeff[n+1]*c[2]; dd += coeff[n+1]*c[3]
        a, b = b, c
    end
    if core_l === nothing
        scale = 2/(ro-ri)
        return f, scale*d, scale^2*dd
    end
    l = core_l; prefactor = (r/ri)^l
    return (prefactor*f,
            prefactor*(l*f/r + 4r*d/ri^2),
            prefactor*(l*(l-1)*f/r^2 + (8l+4)*d/ri^2 + 16r^2*dd/ri^4))
end

function _mhd_boundary_region(op, region)
    p = op.params; T = typeof(p.E)
    region === :fluid && return (; pol=:f, tor=:g, ri=p.ricb, ro=one(T), η=p.Em, core=false)
    region === :core && return (; pol=:fi, tor=:gi, ri=zero(T), ro=p.ricb, η=p.Em, core=true)
    region === :mantle && return (; pol=:fm, tor=:gm, ri=one(T), ro=p.mantle_radius,
                                   η=p.Em*p.mantle_diffusivity_ratio, core=false)
    throw(ArgumentError("Unknown magnetic region $region"))
end

function _mhd_boundary_radial(full, index, section, l, r, region)
    coeff = view(full, index[(l, section)])
    if region.core
        return _mhd_boundary_jet(coeff, r, region.ro, region.ro; core_l=l)
    end
    return _mhd_boundary_jet(coeff, r, region.ri, region.ro)
end

function _mhd_boundary_fields(op, full, index, angular, r, region_name)
    region = _mhd_boundary_region(op, region_name)
    n = length(angular.μ); T = eltype(angular.μ)
    B = zeros(Complex{T}, n, 3); J = zero(B); U = zero(B)
    sections = [(region.pol, :p, op.ll_f), (region.tor, :t, op.ll_g)]
    region_name === :fluid && append!(sections, [(:u, :up, op.ll_u), (:v, :ut, op.ll_v)])
    for (section, kind, ls) in sections, l in ls
        f, d, dd = _mhd_boundary_radial(full, index, section, l, r, region)
        q = l*(l+1); y, h, v = angular.basis[l]
        if kind in (:p, :up)
            field = kind === :p ? B : U
            field[:,1] .+= (q*f/r) .* y
            field[:,2] .+= (d+f/r) .* h
            field[:,3] .+= (d+f/r) .* v
            if kind === :p
                lap = dd+2d/r-q*f/r^2
                J[:,2] .-= lap .* v
                J[:,3] .+= lap .* h
            end
        else
            field = kind === :t ? B : U
            field[:,2] .+= f .* v
            field[:,3] .-= f .* h
            if kind === :t
                J[:,1] .+= (q*f/r) .* y
                J[:,2] .+= (d+f/r) .* h
                J[:,3] .+= (d+f/r) .* v
            end
        end
    end
    current = region.η .* J[:,2:3]
    motional = zeros(Complex{T}, n, 2)
    if region_name === :fluid
        br, bt = _mhd_boundary_background(op.params, r, angular.μ)
        motional[:,1] .= -U[:,3] .* br
        motional[:,2] .= -U[:,1] .* bt .+ U[:,2] .* br
    end
    return (; B, J, U, current, motional, E=current+motional)
end

function _mhd_boundary_background(p, r, μ)
    s = sqrt.(one(eltype(μ)) .- μ.^2)
    p.B0_type === axial && return μ, -s
    p.B0_type === dipole && return μ ./ r^3, s ./ (2r^3)
    return zero(μ), zero(μ)
end

function _mhd_boundary_vacuum(op, full, index, angular, r, region_name, side)
    region = _mhd_boundary_region(op, region_name)
    B = zeros(Complex{eltype(angular.μ)}, length(angular.μ), 3)
    for l in op.ll_f
        f = first(_mhd_boundary_radial(full, index, region.pol, l, r, region))
        y, h, v = angular.basis[l]
        slope = side === :inner ? l+1 : -l
        B[:,1] .+= (l*(l+1)*f/r) .* y
        B[:,2] .+= (slope*f/r) .* h
        B[:,3] .+= (slope*f/r) .* v
    end
    return B
end

# Physical volume RMS amplitudes provide a scale even when every exact wall
# term vanishes. Orthogonality integrates the full vector harmonic field.
function _mhd_boundary_bulk(op, full, index, region_name, ::Type{T}) where T
    region = _mhd_boundary_region(op, region_name)
    nq = region.core ? 2op.params.N + op.params.lmax + 3 : op.params.N + 3
    x, weights = _resolution_gauss(nq, T)
    radii = region.ri .+ (x .+ one(T)) .* ((region.ro-region.ri)/2)
    weights .*= (region.ro-region.ri)/2
    magnetic = zero(T); velocity = zero(T)
    sections = [(region.pol, :p, op.ll_f), (region.tor, :t, op.ll_g)]
    region_name === :fluid && append!(sections, [(:u, :up, op.ll_u), (:v, :ut, op.ll_v)])
    for (section, kind, ls) in sections, l in ls, (j,r) in enumerate(radii)
        f, d, _ = _mhd_boundary_radial(full, index, section, l, r, region)
        q = T(l*(l+1))
        magnitude = kind in (:p,:up) ? hypot(sqrt(q)*abs(f),abs(r*d+f)) : abs(r*f)
        contribution = sqrt(weights[j]*q/T(2l+1))*magnitude
        if kind in (:p,:t)
            magnetic = hypot(magnetic,contribution)
        else
            velocity = hypot(velocity,contribution)
        end
    end
    volume = T(4)*T(π)/3*(region.ro^3-region.ri^3)
    return (; B=magnetic/sqrt(volume), U=velocity/sqrt(volume),
              length=region.ro-region.ri, η=region.η)
end

_mhd_boundary_norm(field, angular, r) =
    sqrt(2oftype(r,π))*r*norm(sqrt.(angular.w) .* field)

function _mhd_boundary_metric(residual, scale, amplitude, rtol, atol)
    relative_residual = iszero(scale) ? (iszero(residual) ? zero(scale) : oftype(scale,Inf)) : residual/scale
    # Compare in the rescaled vector to avoid under/overflow when the caller
    # supplies an arbitrarily normalized eigenvector.
    passed = isfinite(residual) && isfinite(scale) && isfinite(relative_residual) &&
             residual <= rtol*scale + atol/amplitude
    return (; residual=residual*amplitude, scale=scale*amplitude, relative_residual,
              tolerance=atol+rtol*(scale*amplitude), passed)
end

function _mhd_boundary_wall(op, full, index, angular, r, region_name, side, bc,
                            other_name, bulk, amplitude, rtol, atol)
    fields = _mhd_boundary_fields(op, full, index, angular, r, region_name)
    surface = sqrt(4oftype(r,π))*r
    own = bulk[region_name]
    bscale = max(_mhd_boundary_norm(fields.B,angular,r), surface*own.B)
    # The imposed fields have max|B0| = 1 (axial), or r^-3 (dipole).
    background = op.params.B0_type === dipole ? inv(r^3) : one(r)
    escale = max(_mhd_boundary_norm(fields.current,angular,r),
                 _mhd_boundary_norm(fields.motional,angular,r),
                 surface*(own.η*own.B/own.length + own.U*background))
    normal_field = tangential_field = tangential_electric = nothing
    if bc == 0
        vacuum = _mhd_boundary_vacuum(op,full,index,angular,r,region_name,side)
        bscale = max(bscale, _mhd_boundary_norm(vacuum,angular,r))
        tangential_field = _mhd_boundary_metric(
            _mhd_boundary_norm(fields.B[:,2:3]-vacuum[:,2:3],angular,r),
            bscale,amplitude,rtol,atol)
    elseif bc == 2
        normal_field = _mhd_boundary_metric(_mhd_boundary_norm(fields.B[:,1],angular,r),
            bscale,amplitude,rtol,atol)
        tangential_electric = _mhd_boundary_metric(_mhd_boundary_norm(fields.E,angular,r),
            escale,amplitude,rtol,atol)
    else
        other = _mhd_boundary_fields(op,full,index,angular,r,other_name)
        ref = bulk[other_name]
        bscale = max(bscale,_mhd_boundary_norm(other.B,angular,r),surface*ref.B)
        escale = max(escale,_mhd_boundary_norm(other.current,angular,r),surface*ref.η*ref.B/ref.length)
        normal_field = _mhd_boundary_metric(_mhd_boundary_norm(fields.B[:,1]-other.B[:,1],angular,r),
            bscale,amplitude,rtol,atol)
        tangential_field = _mhd_boundary_metric(_mhd_boundary_norm(fields.B[:,2:3]-other.B[:,2:3],angular,r),
            bscale,amplitude,rtol,atol)
        tangential_electric = _mhd_boundary_metric(_mhd_boundary_norm(fields.E-other.E,angular,r),
            escale,amplitude,rtol,atol)
    end
    passed = all(c -> c === nothing || c.passed, (normal_field,tangential_field,tangential_electric))
    kind = bc == 0 ? :insulating : bc == 2 ? :perfect_conductor : :conducting_interface
    return (; kind, radius=r, passed, normal_field, tangential_field, tangential_electric)
end

function _mhd_boundary_mode(op, vector, index, angular, rtol, atol)
    # A complex modulus can overflow even when both stored components are
    # finite. Componentwise scaling keeps such a vector representable here.
    amplitude = maximum(v -> max(abs(real(v)),abs(imag(v))), vector)
    if iszero(amplitude)
        return (; checked=false, passed=false, inner=nothing, outer=nothing, mantle_outer=nothing)
    end
    T = eltype(angular.μ)
    full = Complex{T}.(vector ./ amplitude)
    p = op.params
    regions = [:fluid]
    p.bci_magnetic == 1 && push!(regions,:core)
    p.bco_magnetic == 1 && push!(regions,:mantle)
    bulk = Dict(region => _mhd_boundary_bulk(op,full,index,region,T) for region in regions)
    inner = _mhd_boundary_wall(op,full,index,angular,T(p.ricb),:fluid,:inner,p.bci_magnetic,
                              :core,bulk,amplitude,rtol,atol)
    outer = _mhd_boundary_wall(op,full,index,angular,one(T),:fluid,:outer,p.bco_magnetic,
                              :mantle,bulk,amplitude,rtol,atol)
    mantle_outer = p.bco_magnetic == 1 ?
        _mhd_boundary_wall(op,full,index,angular,T(p.mantle_radius),:mantle,:outer,0,
                           nothing,bulk,amplitude,rtol,atol) : nothing
    passed = inner.passed && outer.passed && (mantle_outer === nothing || mantle_outer.passed)
    return (; checked=true, passed, inner, outer, mantle_outer)
end

function _mhd_boundary_maximum(reports)
    walls = (:inner,:outer,:mantle_outer)
    channels = (:normal_field,:tangential_field,:tangential_electric)
    return NamedTuple{walls}(map(walls) do wall
        present = [(k,getproperty(r,wall)) for (k,r) in enumerate(reports)
                   if getproperty(r,wall) !== nothing]
        isempty(present) && return nothing
        NamedTuple{channels}(map(channels) do channel
            metrics = [(k,getproperty(w,channel)) for (k,w) in present
                       if getproperty(w,channel) !== nothing]
            isempty(metrics) && return nothing
            k, worst = metrics[argmax([c.relative_residual for (_,c) in metrics])]
            merge(worst,(;mode=k,passed=all(c.passed for (_,c) in metrics)))
        end)
    end)
end
