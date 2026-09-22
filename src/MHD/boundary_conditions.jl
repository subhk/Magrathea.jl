# Unknowns are Chebyshev coefficients; fluid residuals use derivative-order
# ultraspherical bases. Tau constraints replace their highest coefficients,
# never spatial endpoint rows.

"""Diffusion in a stationary core of the same diffusivity/permeability as the shell.

The regular potential is (r/ricb)^l a(x), x=2(r/ricb)^2-1. Its radial
Laplacian is (r/ricb)^l/ricb^2 [8(x+1)a'' + (8l+12)a'].
"""
function _mhd_core_operators(op::MHDStabilityOperator{T}, l::Int) where T
    p = op.params; N = p.N
    radial(k, d) = sparse_radial_operator(k, d, N, -one(T), one(T))
    mass = sparse(one(T)I, N + 1, N + 1)
    diffusion = (p.Em / p.ricb^2) *
        (8 * (radial(1, 2) + radial(0, 2)) + (8l + 12) * radial(0, 1))
    return diffusion, mass
end

# Coefficient of the spheroidal part of B0r er×u_h. The wall is impermeable;
# its tangential electric field is Em curl(b)_h + B0r er×u_h. Project with
# exact degree-one harmonic identities in the native Y_lm/sqrt(2l+1)
# convention. Quadrature plus an absolute cutoff leaked forbidden couplings in
# Float32, particularly when a dipole's r^-3 factor is large at the inner wall.
function _mhd_wall_emf(op::MHDStabilityOperator{T}, lout, lin, section, radius) where T
    p = op.params
    p.B0_type == no_field && return zero(Complex{T})
    scale = p.B0_type == dipole ? inv(T(radius)^3) : one(T)
    qo = T(lout * (lout + 1))
    if section == :u
        lout == lin || return zero(Complex{T})
        return Complex{T}(0, -scale * T(p.m) / qo)
    elseif section == :v
        abs(lout - lin) == 1 || return zero(Complex{T})
        qi = T(lin * (lin + 1)); k = max(lout, lin)
        # ∫ cosθ ∇Yout*·∇Yin = (qo+qi-2)/2 ∫ cosθ Yout* Yin.
        c = sqrt(T(k^2 - p.m^2)) / T(2lin + 1)
        return Complex{T}(scale * (qo + qi - 2) * c / (2qo))
    end
    throw(ArgumentError("Velocity section must be :u or :v"))
end

"""Shared tau boundary rows and sparse A entries for serial and distributed MHD."""
function _compute_mhd_bc(op::MHDStabilityOperator{T}) where T
    p = op.params; N = p.N; ri = p.ricb; ro = one(T)
    imap = _mhd_index_map(op)
    rows = Set{Int}()
    entries = Tuple{Int,Int,Complex{T}}[]
    function add!(row, block, values)
        push!(rows, row)
        for (col, val) in zip(block, values)
            iszero(val) || push!(entries, (row, col, Complex{T}(val)))
        end
    end
    scale = T(2) / (ro - ri)
    val(side) = _chebyshev_boundary_values(N, side, T)
    deriv(side) = scale .* _chebyshev_boundary_derivative(N, side, T)
    second(side) = scale^2 .* _chebyshev_boundary_second_derivative(N, side, T)
    vo, vi = val(:outer), val(:inner)
    do_, di = deriv(:outer), deriv(:inner)
    for l in op.ll_u
        block = imap[(l, :u)]; lastrow = last(block)
        add!(lastrow - 3, block, vo)
        add!(lastrow - 2, block, p.bco == 1 ? do_ : second(:outer))
        add!(lastrow - 1, block, vi)
        add!(lastrow, block, p.bci == 1 ? di : second(:inner))
    end
    for l in op.ll_v
        block = imap[(l, :v)]
        add!(last(block) - 1, block, p.bco == 1 ? vo : vo .- ro .* do_)
        add!(last(block), block, p.bci == 1 ? vi : vi .- ri .* di)
    end
    for l in op.ll_h
        block = imap[(l, :h)]
        add!(last(block) - 1, block, p.bco_thermal == 0 ? vo : do_)
        add!(last(block), block, p.bci_thermal == 0 ? vi : di)
    end

    # Nonzero wall slip contributes to electric matching. For no-slip walls
    # u_h=0, so no extra entries are necessary.
    function add_emf!(row, l, side)
        radius = side == :inner ? ri : ro
        mechanical = side == :inner ? p.bci : p.bco
        mechanical == 1 && return
        v, d = side == :inner ? (vi, di) : (vo, do_)
        for (section, ls) in ((:u, op.ll_u), (:v, op.ll_v)), lin in ls
            c = _mhd_wall_emf(op, l, lin, section, radius)
            iszero(c) && continue
            add!(row, imap[(lin, section)], c .* (section == :u ? d .+ v ./ radius : v))
        end
    end

    for l in op.ll_f
        block = imap[(l, :f)]; outerrow = last(block) - 1; innerrow = last(block)
        add!(outerrow, block, p.bco_magnetic == 0 ? (l + 1) .* vo .+ ro .* do_ : vo)
        if p.bci_magnetic == 0
            add!(innerrow, block, l .* vi .- ri .* di)
        elseif p.bci_magnetic == 2
            # f=0 gives zero normal perturbation field. The induction equation
            # supplies the other electric condition; no extra diffusion row.
            add!(innerrow, block, vi)
        else
            core = imap[(l, :fi)]
            dc = (l .* vo .+ 4 .* _chebyshev_boundary_derivative(N, :outer, T)) ./ ri
            add!(innerrow, block, vi)
            add!(innerrow, core, -vo)
            add!(last(core), block, di)
            add!(last(core), core, -dc)
        end
    end
    for l in op.ll_g
        block = imap[(l, :g)]; outerrow = last(block) - 1; innerrow = last(block)
        if p.bco_magnetic == 0
            add!(outerrow, block, vo)
        else
            add!(outerrow, block, p.Em .* (do_ .+ vo ./ ro))
            add_emf!(outerrow, l, :outer)
        end
        if p.bci_magnetic == 0
            add!(innerrow, block, vi)
        elseif p.bci_magnetic == 2
            add!(innerrow, block, p.Em .* (di .+ vi ./ ri))
            add_emf!(innerrow, l, :inner)
        else
            core = imap[(l, :gi)]
            dc = (l .* vo .+ 4 .* _chebyshev_boundary_derivative(N, :outer, T)) ./ ri
            add!(innerrow, block, vi)
            add!(innerrow, core, -vo)
            add!(last(core), block, p.Em .* di)
            add!(last(core), core, -p.Em .* dc)
            add_emf!(last(core), l, :inner)
        end
    end
    return rows, entries
end

function _apply_mhd_boundary_conditions!(A, B, op, sections)
    rows, entries = _compute_mhd_bc(op)
    imap = _mhd_index_map(op)
    selected = Set(i for ((_, sec), block) in imap if sec in sections for i in block if i in rows)
    for row in selected
        _zero_row!(A, row)
        _zero_row!(B, row)
    end
    for (row, col, val) in entries
        row in selected && (A[row, col] += val)
    end
    return nothing
end

"""Apply velocity conditions to the highest residual coefficient rows."""
apply_velocity_boundary_conditions!(A, B, op::MHDStabilityOperator) =
    _apply_mhd_boundary_conditions!(A, B, op, (:u, :v))

"""Apply thermal conditions to the highest residual coefficient rows."""
apply_temperature_boundary_conditions!(A, B, op::MHDStabilityOperator) =
    _apply_mhd_boundary_conditions!(A, B, op, (:h,))
