# =============================================================================
#  MHD Matrix Assembly
#
#  Assembles the full MHD generalized eigenvalue problem:
#      A * x = λ * B * x
#
#  Where x = [u, v, f, g, h] contains:
#  - u: poloidal velocity perturbation
#  - v: toroidal velocity perturbation
#  - f: poloidal magnetic field perturbation
#  - g: toroidal magnetic field perturbation
#  - h: temperature perturbation
#
# =============================================================================

"""
MHD matrix assembly.
Must be included after MHD/types.jl and MHD/operator_functions.jl.
"""

# This file is included in Magrathea via MHD/MHD.jl
# All operator functions (operator_u, operator_coriolis_*, etc.) and is_dipole_case
# are available in the Magrathea module namespace.

# -----------------------------------------------------------------------------
# Inline operator construction for MHDStabilityOperator
# These extend the functions from SparseOperator.jl to work with MHDStabilityOperator
# -----------------------------------------------------------------------------

"""Return the poloidal-velocity mass operator for the MHD B matrix."""
function operator_u(op::MHDStabilityOperator{T}, l::Int) where {T}
    L = l * (l + 1)
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return L * (L * op.r4_D0_u - 2 * op.r5_D1_u - op.r6_D2_u)
    else
        return L * (L * op.r2_D0_u - 2 * op.r3_D1_u - op.r4_D2_u)
    end
end

"""Return the diagonal Coriolis contribution in the poloidal velocity equation."""
function operator_coriolis_diagonal(op::MHDStabilityOperator{T}, l::Int, m::Int) where {T}
    L = l * (l + 1)
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return 2im * m * (-L * op.r4_D0_u + 2 * op.r5_D1_u + op.r6_D2_u)
    else
        return 2im * m * (-L * op.r2_D0_u + 2 * op.r3_D1_u + op.r4_D2_u)
    end
end

"""Return the `l -> l +/- 1` Coriolis coupling from toroidal to poloidal velocity."""
function operator_coriolis_offdiag(op::MHDStabilityOperator{T}, l::Int, m::Int, offset::Int) where {T}
    dipole = is_dipole_case(op.params.B0_type, op.params.ricb)
    if offset == -1
        C = (l^2 - 1) * sqrt(T(l^2 - m^2)) / (2l - 1)
        if dipole
            mtx = 2 * C * ((l - 1) * op.r5_D0_u - op.r6_D1_u)
        else
            mtx = 2 * C * ((l - 1) * op.r3_D0_u - op.r4_D1_u)
        end
        return mtx, -1
    elseif offset == 1
        C = l * (l + 2) * sqrt(T((l + m + 1) * (l - m + 1))) / (2l + 3)
        if dipole
            mtx = 2 * C * (-(l + 2) * op.r5_D0_u - op.r6_D1_u)
        else
            mtx = 2 * C * (-(l + 2) * op.r3_D0_u - op.r4_D1_u)
        end
        return mtx, 1
    else
        error("offset must be ±1 for Coriolis off-diagonal")
    end
end

"""Return the viscous poloidal-velocity diffusion block."""
function operator_viscous_diffusion(op::MHDStabilityOperator{T}, l::Int, E::T) where {T}
    L = l * (l + 1)
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return E * L * (-L * (l + 2) * (l - 1) * op.r2_D0_u +
                        2 * L * op.r4_D2_u -
                        4 * op.r5_D3_u -
                        op.r6_D4_u)
    else
        return E * L * (-L * (l + 2) * (l - 1) * op.r0_D0_u +
                        2 * L * op.r2_D2_u -
                        4 * op.r3_D3_u -
                        op.r4_D4_u)
    end
end

"""Return the temperature-to-poloidal-velocity buoyancy coupling block."""
function operator_buoyancy(op::MHDStabilityOperator{T}, l::Int, Ra::T, Pr::T) where {T}
    # Convert gap-based Ra to internal Ra
    # Ra_internal = Ra_gap / gap^3 (gap = r_o - r_i = 1 - ricb when r_o = 1)
    E = op.params.E
    ricb = op.params.ricb
    gap = one(T) - ricb
    Ra_internal = Ra / gap^3

    # Beyonce factor = BV² = -Ra_internal * E² / Pr
    beyonce = -Ra_internal * E^2 / Pr
    L = l * (l + 1)
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return beyonce * L * op.r6_D0_u
    else
        return beyonce * L * op.r4_D0_u
    end
end

"""Return the reverse Coriolis coupling from poloidal to toroidal velocity rows."""
function operator_coriolis_v_to_u(op::MHDStabilityOperator{T}, l::Int, m::Int, offset::Int) where {T}
    dipole = is_dipole_case(op.params.B0_type, op.params.ricb)
    if offset == -1
        C = (l^2 - 1) * sqrt(T(l^2 - m^2)) / (2l - 1)
        if dipole
            return 2 * C * ((l - 1) * op.r4_D0_v - op.r5_D1_v)
        else
            return 2 * C * ((l - 1) * op.r1_D0_v - op.r2_D1_v)
        end
    elseif offset == 1
        C = l * (l + 2) * sqrt(T((l + m + 1) * (l - m + 1))) / (2l + 3)
        if dipole
            return 2 * C * (-(l + 2) * op.r4_D0_v - op.r5_D1_v)
        else
            return 2 * C * (-(l + 2) * op.r1_D0_v - op.r2_D1_v)
        end
    else
        error("offset must be ±1 for Coriolis v→u coupling")
    end
end

"""Return the toroidal-velocity mass operator for the MHD B matrix."""
function operator_u_toroidal(op::MHDStabilityOperator{T}, l::Int) where {T}
    L = l * (l + 1)
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return L * op.r5_D0_v
    else
        return L * op.r2_D0_v
    end
end

"""Return the diagonal Coriolis contribution in the toroidal velocity equation."""
function operator_coriolis_toroidal(op::MHDStabilityOperator{T}, l::Int, m::Int) where {T}
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return -2im * m * op.r5_D0_v
    else
        return -2im * m * op.r2_D0_v
    end
end

"""Return the viscous toroidal-velocity diffusion block."""
function operator_viscous_toroidal(op::MHDStabilityOperator{T}, l::Int, E::T) where {T}
    L = l * (l + 1)
    if is_dipole_case(op.params.B0_type, op.params.ricb)
        return E * L * (-L * op.r3_D0_v + 2 * op.r4_D1_v + op.r5_D2_v)
    else
        return E * L * (-L * op.r0_D0_v + 2 * op.r1_D1_v + op.r2_D2_v)
    end
end

"""Return the temperature mass operator for differential or internal heating."""
function operator_theta(op::MHDStabilityOperator{T}, l::Int) where {T}
    if op.params.heating == :differential
        return op.r3_D0_h
    else  # :internal
        return op.r2_D0_h
    end
end

"""Return the thermal diffusion block in the temperature equation."""
function operator_thermal_diffusion(op::MHDStabilityOperator{T}, l::Int, Etherm::T) where {T}
    L = l * (l + 1)
    if op.params.heating == :differential
        return Etherm * (-L * op.r1_D0_h + 2 * op.r2_D1_h + op.r3_D2_h)
    else  # :internal
        return Etherm * (-L * op.r0_D0_h + 2 * op.r1_D1_h + op.r2_D2_h)
    end
end

"""Return the poloidal-velocity-to-temperature advection block."""
function operator_thermal_advection(op::MHDStabilityOperator{T}, l::Int) where {T}
    L = l * (l + 1)
    if op.params.heating == :differential
        ricb = op.params.ricb
        gap = one(T) - ricb
        return L * op.r0_D0_h * (ricb / gap)
    else  # :internal
        return L * op.r2_D0_h
    end
end

"""Row layout of the MHD tau matrix as a Phase-1 index_map: keys `(ℓ, section)` for
sections `:u,:v,:f,:g,:h` in order, each a contiguous `(N+1)`-row range."""
function _mhd_index_map(op::MHDStabilityOperator)
    n_per_mode = op.params.N + 1
    im = Dict{Tuple{Int,Symbol}, UnitRange{Int}}()
    off = 0
    for (sec, ls) in ((:u, op.ll_u), (:v, op.ll_v), (:f, op.ll_f), (:g, op.ll_g), (:h, op.ll_h))
        for l in ls
            im[(l, sec)] = (off + 1):(off + n_per_mode)
            off += n_per_mode
        end
    end
    return im
end

"""Per-owned-row diagonal/off-diagonal nnz from COO triplets, for the owned PETSc row
block `[rstart, rend)` (0-based, half-open), diagonal band = same column range. `rows`
and `cols` are 1-based Julia indices. Returns `(d_nnz, o_nnz)`, length `rend-rstart`."""
function _owned_coo_nnz(rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer},
                        rstart::Int, rend::Int)
    nloc = rend - rstart
    d = zeros(Int, nloc); o = zeros(Int, nloc)
    @inbounds for k in eachindex(rows)
        r0 = rows[k] - 1
        if rstart <= r0 < rend
            c0 = cols[k] - 1
            i = r0 - rstart + 1
            (rstart <= c0 < rend) ? (d[i] += 1) : (o[i] += 1)
        end
    end
    return d, o
end

"""
    assemble_mhd_matrices(op::MHDStabilityOperator)

Assemble the full MHD matrices A and B for the generalized eigenvalue problem.

Returns: (A, B, interior_dofs, info)

# Matrix Structure

The matrix is organized in 5 sections:
1. Section u (poloidal velocity): 2curl Navier-Stokes with Lorentz force
2. Section v (toroidal velocity): 1curl Navier-Stokes with Lorentz force
3. Section f (poloidal B field): no-curl induction equation
4. Section g (toroidal B field): 1curl induction equation
5. Section h (temperature): heat equation with advection

# Couplings

Velocity → Velocity: Coriolis, viscous diffusion
Velocity → Magnetic: Induction (u,v → f,g)
Magnetic → Velocity: Lorentz force (f,g → u,v)
Magnetic → Magnetic: Magnetic diffusion
Velocity → Temperature: Thermal advection
Temperature → Velocity: Buoyancy
"""
function _assemble_mhd_coo(op::MHDStabilityOperator{T}; owned_julia_rows::Union{Nothing,UnitRange{Int}}=nothing) where {T}
    params = op.params
    E = params.E
    Pr = params.Pr
    Pm = params.Pm
    Ra = params.Ra
    Le = params.Le
    Etherm = params.Etherm
    Em = params.Em
    m = params.m
    N = params.N
    ricb = params.ricb

    n = op.matrix_size
    n_per_mode = N + 1

    nb_u = length(op.ll_u)
    nb_v = length(op.ll_v)
    nb_f = length(op.ll_f)
    nb_g = length(op.ll_g)
    nb_h = length(op.ll_h)

    section_info = String[]
    nb_u > 0 && push!(section_info, "u($nb_u)")
    nb_v > 0 && push!(section_info, "v($nb_v)")
    nb_f > 0 && push!(section_info, "f($nb_f)")
    nb_g > 0 && push!(section_info, "g($nb_g)")
    nb_h > 0 && push!(section_info, "h($nb_h)")
    @info "Assembling MHD sparse matrices" size="$n × $n" sections=join(section_info, ", ")

    # Use COO format for efficient assembly. Keep value storage tied to the
    # parameter precision; Coriolis terms still introduce complex values.
    A_rows = Int[]
    A_cols = Int[]
    A_vals = Complex{T}[]
    B_rows = Int[]
    B_cols = Int[]
    B_vals = Complex{T}[]

    # Helper function to add block to sparse matrix. Shift the COO indices as we
    # push instead of allocating `Is .+ offset` / `Js .+ offset` temporaries.
    function add_block!(rows, cols, vals, block, row_offset, col_offset)
        Is, Js, Vs = findnz(block)
        @inbounds for k in eachindex(Vs)
            grow = Is[k] + row_offset
            if owned_julia_rows === nothing || grow in owned_julia_rows
                push!(rows, grow); push!(cols, Js[k] + col_offset); push!(vals, Vs[k])
            end
        end
        return nothing
    end

    # =========================================================================
    # SECTION U: Poloidal Velocity (2curl Navier-Stokes + Lorentz)
    # =========================================================================
    @debug "Assembling section u (poloidal velocity)..."

    for (k, l) in enumerate(op.ll_u)
        row_base = (k - 1) * n_per_mode
        if owned_julia_rows !== nothing &&
           isempty(intersect((row_base+1):(row_base+n_per_mode), owned_julia_rows))
            continue
        end
        col_base = (k - 1) * n_per_mode

        # ---------------------------------------------------------------------
        # B matrix: Time derivative (inertia)
        # ---------------------------------------------------------------------
        u_op = operator_u(op, l)
        add_block!(B_rows, B_cols, B_vals, -u_op, row_base, col_base)

        # ---------------------------------------------------------------------
        # A matrix: RHS operators
        # ---------------------------------------------------------------------

        # Coriolis force (diagonal)
        cori_op = operator_coriolis_diagonal(op, l, m)
        add_block!(A_rows, A_cols, A_vals, cori_op, row_base, col_base)

        # Viscous diffusion (appears with a minus sign in Kore)
        visc_op = operator_viscous_diffusion(op, l, E)
        add_block!(A_rows, A_cols, A_vals, -visc_op, row_base, col_base)

        # Buoyancy (coupling from temperature)
        buoy_op = operator_buoyancy(op, l, Ra, Pr)
        temp_col_base = (nb_u + nb_v + nb_f + nb_g + k - 1) * n_per_mode
        add_block!(A_rows, A_cols, A_vals, buoy_op, row_base, temp_col_base)

        # Lorentz force from magnetic field (if Le > 0)
        if Le > 0
            # Coupling from poloidal magnetic field (bpol, section f)
            for offset in -2:2
                l_coupled = l + offset
                k_f = findfirst(==(l_coupled), op.ll_f)
                if k_f !== nothing
                    f_col_base = (nb_u + nb_v + k_f - 1) * n_per_mode
                    lorentz_bpol = operator_lorentz_poloidal_from_bpol(op, l, m, offset, Le)
                    add_block!(A_rows, A_cols, A_vals, lorentz_bpol, row_base, f_col_base)
                end
            end

            # Diagonal: toroidal B at same l (only if such mode exists)
            # For symm=±1, ll_u and ll_g have different parities, so diagonal coupling doesn't exist
            k_g = findfirst(==(l), op.ll_g)
            if k_g !== nothing
                lorentz_diag = operator_lorentz_poloidal_diagonal(op, l, Le)
                g_col_base = (nb_u + nb_v + nb_f + k_g - 1) * n_per_mode
                add_block!(A_rows, A_cols, A_vals, lorentz_diag, row_base, g_col_base)
            end

            # Off-diagonal: toroidal B at l±1
            for offset in (-1, 1)
                l_coupled = l + offset
                k_coupled = findfirst(==(l_coupled), op.ll_g)
                if k_coupled !== nothing
                    g_col_coupled = (nb_u + nb_v + nb_f + k_coupled - 1) * n_per_mode

                    lorentz_off = operator_lorentz_poloidal_offdiag(op, l, m, offset, Le)
                    add_block!(A_rows, A_cols, A_vals, lorentz_off, row_base, g_col_coupled)
                end
            end
        end

        # Coriolis off-diagonal: u ↔ v coupling
        for offset in (-1, 1)
            l_coupled = l + offset
            k_coupled = findfirst(==(l_coupled), op.ll_v)
            if k_coupled !== nothing
                v_col_coupled = (nb_u + k_coupled - 1) * n_per_mode

                cori_off, _ = operator_coriolis_offdiag(op, l, m, offset)
                add_block!(A_rows, A_cols, A_vals, cori_off, row_base, v_col_coupled)
            end
        end
    end

    # =========================================================================
    # SECTION V: Toroidal Velocity (1curl Navier-Stokes + Lorentz)
    # =========================================================================
    @debug "Assembling section v (toroidal velocity)..."

    for (k, l) in enumerate(op.ll_v)
        row_base = (nb_u + k - 1) * n_per_mode
        if owned_julia_rows !== nothing &&
           isempty(intersect((row_base+1):(row_base+n_per_mode), owned_julia_rows))
            continue
        end
        col_base = (nb_u + k - 1) * n_per_mode

        # ---------------------------------------------------------------------
        # B matrix: Time derivative
        # ---------------------------------------------------------------------
        v_op = operator_u_toroidal(op, l)
        add_block!(B_rows, B_cols, B_vals, -v_op, row_base, col_base)

        # ---------------------------------------------------------------------
        # A matrix: RHS operators
        # ---------------------------------------------------------------------

        # Coriolis (diagonal)
        cori_tor = operator_coriolis_toroidal(op, l, m)
        add_block!(A_rows, A_cols, A_vals, cori_tor, row_base, col_base)

        # Viscous diffusion (minus sign following Kore)
        visc_tor = operator_viscous_toroidal(op, l, E)
        add_block!(A_rows, A_cols, A_vals, -visc_tor, row_base, col_base)

        # Lorentz force from magnetic field (if Le > 0)
        if Le > 0
            # Coupling from poloidal magnetic field (section f, offsets l-1:l+1)
            for offset in -1:1
                l_src = l + offset
                idx_f = findfirst(==(l_src), op.ll_f)
                idx_f === nothing && continue
                f_col_base = (nb_u + nb_v + idx_f - 1) * n_per_mode
                lorentz_from_bpol = operator_lorentz_toroidal_from_bpol(op, l, m, offset, Le)
                add_block!(A_rows, A_cols, A_vals, lorentz_from_bpol, row_base, f_col_base)
            end

            # Coupling from toroidal magnetic field (section g, offsets l-2:l+2)
            for offset in -2:2
                l_src = l + offset
                idx_g = findfirst(==(l_src), op.ll_g)
                idx_g === nothing && continue
                g_col_base = (nb_u + nb_v + nb_f + idx_g - 1) * n_per_mode
                lorentz_from_btor = operator_lorentz_toroidal_from_btor(op, l, m, offset, Le)
                add_block!(A_rows, A_cols, A_vals, lorentz_from_btor, row_base, g_col_base)
            end
        end

        # Coriolis reverse coupling: v → u at l±1
        for offset in (-1, 1)
            l_coupled = l + offset
            k_coupled = findfirst(==(l_coupled), op.ll_u)
            if k_coupled !== nothing
                u_col_coupled = (k_coupled - 1) * n_per_mode

                cori_v_to_u = operator_coriolis_v_to_u(op, l, m, offset)
                add_block!(A_rows, A_cols, A_vals, cori_v_to_u, row_base, u_col_coupled)
            end
        end
    end

    # =========================================================================
    # SECTION F: Poloidal Magnetic Field (no-curl induction)
    # =========================================================================
    if nb_f > 0
        @debug "Assembling section f (poloidal magnetic field)..."

        for (k, l) in enumerate(op.ll_f)
            row_base = (nb_u + nb_v + k - 1) * n_per_mode
            if owned_julia_rows !== nothing &&
               isempty(intersect((row_base+1):(row_base+n_per_mode), owned_julia_rows))
                continue
            end
            col_base = (nb_u + nb_v + k - 1) * n_per_mode

            # -----------------------------------------------------------------
            # B matrix: Time derivative
            # -----------------------------------------------------------------
            b_pol = operator_b_poloidal(op, l)
            add_block!(B_rows, B_cols, B_vals, -b_pol, row_base, col_base)

            # -----------------------------------------------------------------
            # A matrix: RHS operators
            # -----------------------------------------------------------------

            # Magnetic diffusion. Enters A with a MINUS sign, like the (identical-form)
            # viscous operator (cf. velocity sections): the decoupled free-decay modes
            # must dissipate (Re<0). The previous +sign made them grow — see
            # test/mhd_galerkin.jl "magnetic free-decay". (Cross-check vs Kore advised.)
            mag_diff_pol = operator_magnetic_diffusion_poloidal(op, l, Em)
            add_block!(A_rows, A_cols, A_vals, -mag_diff_pol, row_base, col_base)

            # Induction from velocity field
            if Le > 0
                # From poloidal velocity u (offsets l-2 ... l+2)
                for offset in -2:2
                    l_src = l + offset
                    idx_u = findfirst(==(l_src), op.ll_u)
                    idx_u === nothing && continue

                    induct_from_u = operator_induction_poloidal_from_u(op, l, m, offset)
                    u_col_base = (idx_u - 1) * n_per_mode
                    add_block!(A_rows, A_cols, A_vals, induct_from_u, row_base, u_col_base)
                end

                # From toroidal velocity v (offsets l-1 ... l+1)
                for offset in -1:1
                    l_src = l + offset
                    idx_v = findfirst(==(l_src), op.ll_v)
                    idx_v === nothing && continue

                    induct_from_v = operator_induction_poloidal_from_v(op, l, m, offset)
                    v_col_base = (nb_u + idx_v - 1) * n_per_mode
                    add_block!(A_rows, A_cols, A_vals, induct_from_v, row_base, v_col_base)
                end
            end
        end
    end

    # =========================================================================
    # SECTION G: Toroidal Magnetic Field (1curl induction)
    # =========================================================================
    if nb_g > 0
        @debug "Assembling section g (toroidal magnetic field)..."

        for (k, l) in enumerate(op.ll_g)
            row_base = (nb_u + nb_v + nb_f + k - 1) * n_per_mode
            if owned_julia_rows !== nothing &&
               isempty(intersect((row_base+1):(row_base+n_per_mode), owned_julia_rows))
                continue
            end
            col_base = (nb_u + nb_v + nb_f + k - 1) * n_per_mode

            # -----------------------------------------------------------------
            # B matrix: Time derivative
            # -----------------------------------------------------------------
            b_tor = operator_b_toroidal(op, l)
            add_block!(B_rows, B_cols, B_vals, -b_tor, row_base, col_base)

            # -----------------------------------------------------------------
            # A matrix: RHS operators
            # -----------------------------------------------------------------

            # Magnetic diffusion — MINUS sign (dissipative free-decay); see poloidal note above.
            mag_diff_tor = operator_magnetic_diffusion_toroidal(op, l, Em)
            add_block!(A_rows, A_cols, A_vals, -mag_diff_tor, row_base, col_base)

            # Induction from velocity field (if Le > 0)
            if Le > 0
                # From toroidal velocity v (offsets l-2 ... l+2)
                for offset in -2:2
                    l_src = l + offset
                    idx_v = findfirst(==(l_src), op.ll_v)
                    idx_v === nothing && continue
                    v_col_base = (nb_u + idx_v - 1) * n_per_mode
                    induct_v_tor = operator_induction_toroidal_from_v(op, l, m, offset)
                    add_block!(A_rows, A_cols, A_vals, induct_v_tor, row_base, v_col_base)
                end

                # From poloidal velocity u (diagonal and off-diagonal)
                for offset in (-1, 0, 1)
                    l_coupled = l + offset
                    k_coupled = findfirst(==(l_coupled), op.ll_u)
                    if k_coupled !== nothing
                        u_col_coupled = (k_coupled - 1) * n_per_mode

                        induct_u_tor = operator_induction_toroidal_from_u(op, l, m, offset)
                        add_block!(A_rows, A_cols, A_vals, induct_u_tor, row_base, u_col_coupled)
                    end
                end
            end
        end
    end

    # =========================================================================
    # SECTION H: Temperature (same as hydrodynamic case)
    # =========================================================================
    @debug "Assembling section h (temperature)..."

    for (k, l) in enumerate(op.ll_h)
        row_base = (nb_u + nb_v + nb_f + nb_g + k - 1) * n_per_mode
        if owned_julia_rows !== nothing &&
           isempty(intersect((row_base+1):(row_base+n_per_mode), owned_julia_rows))
            continue
        end
        col_base = (nb_u + nb_v + nb_f + nb_g + k - 1) * n_per_mode

        # ---------------------------------------------------------------------
        # B matrix: Time derivative
        # ---------------------------------------------------------------------
        theta_op = operator_theta(op, l)
        add_block!(B_rows, B_cols, B_vals, theta_op, row_base, col_base)

        # ---------------------------------------------------------------------
        # A matrix: RHS operators
        # ---------------------------------------------------------------------

        # Thermal diffusion
        thermal_diff = operator_thermal_diffusion(op, l, Etherm)
        add_block!(A_rows, A_cols, A_vals, thermal_diff, row_base, col_base)

        # Thermal advection (from poloidal velocity)
        vel_col_base = (k - 1) * n_per_mode
        thermal_adv = operator_thermal_advection(op, l)
        add_block!(A_rows, A_cols, A_vals, thermal_adv, row_base, vel_col_base)
    end

    return (A_rows=A_rows, A_cols=A_cols, A_vals=A_vals,
            B_rows=B_rows, B_cols=B_cols, B_vals=B_vals,
            n=n, interior_dofs=Int[], info=Dict{String,Any}())
end

"""Assemble the MHD tau (A, B) sparse matrices with boundary conditions applied."""
function assemble_mhd_matrices(op::MHDStabilityOperator{T}) where {T}
    params = op.params
    N = params.N
    m = params.m

    nb_u = length(op.ll_u)
    nb_v = length(op.ll_v)
    nb_f = length(op.ll_f)
    nb_g = length(op.ll_g)
    nb_h = length(op.ll_h)

    c = _assemble_mhd_coo(op)
    n = c.n

    # =========================================================================
    # Apply boundary conditions at the COO stage, then build CSC once
    # =========================================================================
    # Tau BCs overwrite whole boundary rows across u/v/f/g/h. Applying them on the
    # assembled CSC forces O(nnz) structural insertions per row; instead drop the
    # operator entries on the boundary rows and append the BC functionals to the
    # triplets, so the single sparse() build carries the BCs with no churn.
    @debug "Applying boundary conditions (COO stage)..."
    bc_rows, bcA = _compute_mhd_bc(op)
    _filter_coo_rows!(c.A_rows, c.A_cols, c.A_vals, bc_rows)
    _filter_coo_rows!(c.B_rows, c.B_cols, c.B_vals, bc_rows)  # B boundary rows are zeroed
    @inbounds for (r, cc, v) in bcA
        push!(c.A_rows, r); push!(c.A_cols, cc); push!(c.A_vals, v)
    end

    @debug "Converting to CSC format..."
    A = sparse(c.A_rows, c.A_cols, c.A_vals, n, n)
    B = sparse(c.B_rows, c.B_cols, c.B_vals, n, n)

    @debug "Post-BC sparsity" A_nnz=nnz(A) B_nnz=nnz(B)

    # Identify interior DOFs
    B_diag = diag(B)
    interior_dofs = findall(i -> abs(B_diag[i]) > 1e-14, 1:n)
    @info "MHD assembly complete" interior_dofs=length(interior_dofs) total_dofs=n

    section_labels = String[]
    nb_u > 0 && push!(section_labels, "u")
    nb_v > 0 && push!(section_labels, "v")
    nb_f > 0 && push!(section_labels, "f")
    nb_g > 0 && push!(section_labels, "g")
    nb_h > 0 && push!(section_labels, "h")

    info = Dict(
        "method" => "MHD sparse ultraspherical",
        "N" => N,
        "lmax" => params.lmax,
        "m" => m,
        "nl_modes" => op.nl_modes,
        "matrix_size" => n,
        "sections" => join(section_labels, ", ")
    )

    return A, B, interior_dofs, info
end

# -----------------------------------------------------------------------------
# Boundary condition helpers
# -----------------------------------------------------------------------------

"""Overwrite MHD velocity tau rows with the selected poloidal and toroidal BCs."""
function apply_velocity_boundary_conditions!(A, B, op::MHDStabilityOperator{T}) where {T}
    # Apply boundary conditions to velocity fields (poloidal and toroidal)
    # Following the correct implementation from SparseOperator.jl
    params = op.params
    N = params.N
    n_per_mode = N + 1
    nb_u = length(op.ll_u)
    ro = one(params.E)

    # -------------------------------------------------------------------------
    # Poloidal velocity BCs (section u)
    # -------------------------------------------------------------------------
    for (k, l) in enumerate(op.ll_u)
        row_base = (k - 1) * n_per_mode

        # Outer boundary (r = ro = 1.0)
        if params.bco == 1
            # No-slip: u = 0, du/dr = 0
            apply_boundary_conditions!(A, B, [row_base + 1], :dirichlet, N,
                                       params.ricb, ro)
            apply_boundary_conditions!(A, B, [row_base + 2], :neumann, N,
                                       params.ricb, ro)
        else
            # Stress-free: u = 0, r·d²u/dr² = 0
            apply_boundary_conditions!(A, B, [row_base + 1], :dirichlet, N,
                                       params.ricb, ro)
            apply_boundary_conditions!(A, B, [row_base + 2], :neumann2, N,
                                       params.ricb, ro)
        end

        # Inner boundary (r = ri = ricb)
        if params.bci == 1
            # No-slip: u = 0, du/dr = 0
            apply_boundary_conditions!(A, B, [row_base + n_per_mode], :dirichlet, N,
                                       params.ricb, ro)
            apply_boundary_conditions!(A, B, [row_base + n_per_mode - 1], :neumann, N,
                                       params.ricb, ro)
        else
            # Stress-free: u = 0, r·d²u/dr² = 0
            apply_boundary_conditions!(A, B, [row_base + n_per_mode], :dirichlet, N,
                                       params.ricb, ro)
            apply_boundary_conditions!(A, B, [row_base + n_per_mode - 1], :neumann2, N,
                                       params.ricb, ro)
        end
    end

    # -------------------------------------------------------------------------
    # Toroidal velocity BCs (section v)
    # -------------------------------------------------------------------------
    scale = _radial_scale(params.ricb, ro)
    outer_vals = _chebyshev_boundary_values(N, :outer)
    inner_vals = _chebyshev_boundary_values(N, :inner)
    outer_deriv = _chebyshev_boundary_derivative(N, :outer)
    inner_deriv = _chebyshev_boundary_derivative(N, :inner)
    r_outer = _boundary_radius(params.ricb, ro, :outer)
    r_inner = _boundary_radius(params.ricb, ro, :inner)
    outer_row = @. -r_outer * scale * outer_deriv + outer_vals
    inner_row = @. -r_inner * scale * inner_deriv + inner_vals

    for (k, l) in enumerate(op.ll_v)
        row_base = (nb_u + k - 1) * n_per_mode

        # Outer boundary (r = ro = 1.0)
        if params.bco == 1
            # No-slip: v = 0
            apply_boundary_conditions!(A, B, [row_base + 1], :dirichlet, N,
                                       params.ricb, ro)
        else
            # Stress-free: -r·∂v/∂r + v = 0
            row = row_base + 1
            _zero_row!(A, row)
            _zero_row!(B, row)
            block_start = row_base + 1
            A[row, block_start:(block_start + N)] = Complex{T}.(outer_row)
        end

        # Inner boundary (r = ri = ricb)
        if params.bci == 1
            # No-slip: v = 0
            apply_boundary_conditions!(A, B, [row_base + n_per_mode], :dirichlet, N,
                                       params.ricb, ro)
        else
            # Stress-free: -r·∂v/∂r + v = 0
            row = row_base + n_per_mode
            _zero_row!(A, row)
            _zero_row!(B, row)
            block_start = row_base + 1
            A[row, block_start:(block_start + N)] = Complex{T}.(inner_row)
        end
    end
end

"""Overwrite MHD temperature tau rows with fixed-temperature or fixed-flux BCs."""
function apply_temperature_boundary_conditions!(A, B, op)
    # Apply boundary conditions to temperature field
    # Following the correct implementation from SparseOperator.jl
    params = op.params
    N = params.N
    n_per_mode = N + 1
    nb_u = length(op.ll_u)
    nb_v = length(op.ll_v)
    nb_f = length(op.ll_f)
    nb_g = length(op.ll_g)
    ro = one(params.E)

    # -------------------------------------------------------------------------
    # Temperature BCs (section h)
    # -------------------------------------------------------------------------
    for (k, l) in enumerate(op.ll_h)
        row_base = (nb_u + nb_v + nb_f + nb_g + k - 1) * n_per_mode

        # Outer boundary (r = ro = 1.0)
        if params.bco_thermal == 0
            # Fixed temperature: θ = 0
            apply_boundary_conditions!(A, B, [row_base + 1], :dirichlet, N,
                                       params.ricb, ro)
        else
            # Fixed flux: dθ/dr = 0
            apply_boundary_conditions!(A, B, [row_base + 1], :neumann, N,
                                       params.ricb, ro)
        end

        # Inner boundary (r = ri = ricb)
        if params.bci_thermal == 0
            # Fixed temperature: θ = 0
            apply_boundary_conditions!(A, B, [row_base + n_per_mode], :dirichlet, N,
                                       params.ricb, ro)
        else
            # Fixed flux: dθ/dr = 0
            apply_boundary_conditions!(A, B, [row_base + n_per_mode], :neumann, N,
                                       params.ricb, ro)
        end
    end
end

"""
    _compute_mhd_bc(op) -> (bc_rows::Set{Int}, bcA::Vector{Tuple{Int,Int,Complex{T}}})

Tau boundary-condition specification for the MHD operator as data: `bc_rows` are
overwritten by BCs (zeroed in B, replaced in A) and `bcA` are the (row, col, value)
entries of the replacement A rows. Mirrors apply_velocity/magnetic/temperature_
boundary_conditions! exactly (poloidal/toroidal velocity, the five magnetic
branches, fixed-T/flux), so BCs can be applied at the COO stage with no CSC churn.
"""
function _compute_mhd_bc(op::MHDStabilityOperator{T}) where {T}
    params = op.params
    N = params.N
    n_per_mode = N + 1
    ri = params.ricb
    ro = one(T)
    nb_u = length(op.ll_u); nb_v = length(op.ll_v)
    nb_f = length(op.ll_f); nb_g = length(op.ll_g)

    bc_rows = Set{Int}()
    bcA = Tuple{Int,Int,Complex{T}}[]

    push_row! = (row, bc_type) -> begin
        push!(bc_rows, row)
        rng, vals = _bc_row_values(bc_type, row, N, ri, ro, T)
        @inbounds for (j, c) in enumerate(rng)
            push!(bcA, (row, c, Complex{T}(vals[j])))
        end
    end
    # Explicit dense functional over a block (length N+1); block starts at row_base+1.
    push_block! = (row, row_base, vec) -> begin
        push!(bc_rows, row)
        @inbounds for i in 0:N
            push!(bcA, (row, row_base + 1 + i, Complex{T}(vec[i + 1])))
        end
    end

    # ---- Velocity: poloidal (section u) ----
    for (k, l) in enumerate(op.ll_u)
        row_base = (k - 1) * n_per_mode
        push_row!(row_base + 1, :dirichlet)
        push_row!(row_base + 2, params.bco == 1 ? :neumann : :neumann2)
        push_row!(row_base + n_per_mode, :dirichlet)
        push_row!(row_base + n_per_mode - 1, params.bci == 1 ? :neumann : :neumann2)
    end

    # ---- Velocity: toroidal (section v); stress-free uses explicit functionals ----
    scale = _radial_scale(ri, ro)
    outer_vals = _chebyshev_boundary_values(N, :outer)
    inner_vals = _chebyshev_boundary_values(N, :inner)
    outer_deriv = _chebyshev_boundary_derivative(N, :outer)
    inner_deriv = _chebyshev_boundary_derivative(N, :inner)
    r_outer = _boundary_radius(ri, ro, :outer)
    r_inner = _boundary_radius(ri, ro, :inner)
    outer_row = @. -r_outer * scale * outer_deriv + outer_vals
    inner_row = @. -r_inner * scale * inner_deriv + inner_vals
    for (k, l) in enumerate(op.ll_v)
        row_base = (nb_u + k - 1) * n_per_mode
        params.bco == 1 ? push_row!(row_base + 1, :dirichlet) :
                          push_block!(row_base + 1, row_base, outer_row)
        params.bci == 1 ? push_row!(row_base + n_per_mode, :dirichlet) :
                          push_block!(row_base + n_per_mode, row_base, inner_row)
    end

    # ---- Magnetic (sections f, g) ----
    if nb_f > 0 || nb_g > 0
        mscale = _radial_scale(ri, ro)
        mr_outer = T(_boundary_radius(ri, ro, :outer))
        mr_inner = T(_boundary_radius(ri, ro, :inner))
        mouter_vals = _chebyshev_boundary_values(N, :outer, T)
        minner_vals = _chebyshev_boundary_values(N, :inner, T)
        mouter_deriv = T(mscale) .* _chebyshev_boundary_derivative(N, :outer, T)
        minner_deriv = T(mscale) .* _chebyshev_boundary_derivative(N, :inner, T)
        minner_second = T(mscale)^2 .* _chebyshev_boundary_second_derivative(N, :inner, T)

        # Section f (poloidal magnetic)
        for (k, l) in enumerate(op.ll_f)
            row_base = (nb_u + nb_v + k - 1) * n_per_mode
            row_cmb = row_base + 1
            if params.bco_magnetic == 0   # insulating CMB: (l+1)f + ro f' = 0
                push_block!(row_cmb, row_base, (l + 1) .* mouter_vals .+ mr_outer .* mouter_deriv)
            else                           # perfectly conducting CMB: f = 0
                push_block!(row_cmb, row_base, mouter_vals)
            end
            row_icb = row_base + n_per_mode
            if params.bci_magnetic == 0    # insulating ICB: l f - ri f' = 0
                push_block!(row_icb, row_base, l .* minner_vals .- mr_inner .* minner_deriv)
            elseif params.bci_magnetic == 1
                freq = params.forcing_frequency
                Em = params.Em
                if Em <= 0
                    error("Conducting magnetic BC requires Em > 0")
                end
                if iszero(freq)            # steady limit: l f - ri f' = 0 (== insulating)
                    push_block!(row_icb, row_base, l .* minner_vals .- mr_inner .* minner_deriv)
                else                       # finite frequency: f' - k(j'/j) f = 0
                    k_wave = (1 - 1im) * sqrt(complex(freq) / (2 * Em))
                    dlog = spherical_bessel_j_logderiv(l, k_wave * ri)
                    push_block!(row_icb, row_base, minner_deriv .- (k_wave * dlog) .* minner_vals)
                end
            elseif params.bci_magnetic == 2  # perfect conductor ICB: two rows
                L = l * (l + 1)
                push_block!(row_icb, row_base, minner_vals)             # f = 0
                vt = (L / ri^2) .* minner_vals
                d1 = -(T(2) / ri) .* minner_deriv
                d2 = -minner_second
                push_block!(row_icb - 1, row_base, params.Em .* (vt .+ d1 .+ d2))
            else                            # simple conducting: f = 0
                push_block!(row_icb, row_base, minner_vals)
            end
        end

        # Section g (toroidal magnetic): g = 0 at CMB for all BC types
        for (k, l) in enumerate(op.ll_g)
            row_base = (nb_u + nb_v + nb_f + k - 1) * n_per_mode
            push_block!(row_base + 1, row_base, mouter_vals)
            row_icb = row_base + n_per_mode
            if params.bci_magnetic == 2     # perfect conductor: Em(-g' - g/ri) = 0
                vt = -(T(1) / ri) .* minner_vals
                d1 = -minner_deriv
                push_block!(row_icb, row_base, params.Em .* (vt .+ d1))
            else                            # insulating / conducting / default: g = 0
                push_block!(row_icb, row_base, minner_vals)
            end
        end
    end

    # ---- Temperature (section h) ----
    for (k, l) in enumerate(op.ll_h)
        row_base = (nb_u + nb_v + nb_f + nb_g + k - 1) * n_per_mode
        push_row!(row_base + 1, params.bco_thermal == 0 ? :dirichlet : :neumann)
        push_row!(row_base + n_per_mode, params.bci_thermal == 0 ? :dirichlet : :neumann)
    end

    return bc_rows, bcA
end
