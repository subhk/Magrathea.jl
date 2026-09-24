# =============================================================================
#  Tau-free ultraspherical-Galerkin radial operators.
#
#  Composes the banded ultraspherical primitives (derivative, conversion,
#  multiplication) up to a common C^(q) output basis WITHOUT the S_chain \ D
#  back-solve and WITHOUT tau boundary rows. Boundary conditions enter through
#  a recombined trial basis (see recombination builders below).
# =============================================================================

"""
    _diff_to_ultra(T, k, N, scale) -> SparseMatrixCSC

k-th radial-derivative operator mapping Chebyshev (C^(0)) coefficients to
C^(k) coefficients. Banded, no back-solve.
"""
function _diff_to_ultra(::Type{T}, k::Int, N::Int, scale::T) where {T<:Real}
    D = sparse(one(T)I, N + 1, N + 1)            # C^(0) identity
    for λ in 0:(k - 1)
        Dλ = ultraspherical_derivative(T, λ, N)  # C^(λ) -> C^(λ+1)
        D = (scale * Dλ) * D
    end
    return D                                      # C^(0) -> C^(k)
end

"""
    _convert_up(T, from, to, N) -> SparseMatrixCSC

Product of conversion matrices mapping C^(from) coefficients to C^(to).
"""
function _convert_up(::Type{T}, from::Int, to::Int, N::Int) where {T<:Real}
    S = sparse(one(T)I, N + 1, N + 1)
    for λ in from:(to - 1)
        S = ultraspherical_conversion(T, λ, N) * S
    end
    return S                                      # C^(from) -> C^(to)
end

"""
    banded_radial_term(T, power, deriv, q_out, N, ri, ro) -> SparseMatrixCSC

Banded representation of `r^power * d^deriv/dr^deriv`, as a map from Chebyshev
coefficients to C^(q_out) coefficients (q_out ≥ deriv). No tau rows, no back-solve.
Multiplication by r^power is applied in the C^(deriv) basis, then converted up.
"""
function banded_radial_term(::Type{T}, power::Int, deriv::Int, q_out::Int,
                            N::Int, ri::Real, ro::Real) where {T<:Real}
    @assert deriv <= q_out "q_out must be ≥ deriv"
    scale = T(_radial_scale(ri, ro))
    Dk = _diff_to_ultra(T, deriv, N, scale)              # C^(0) -> C^(deriv)
    if power != 0
        # multiplication_matrix's C^(λ) branch (here λ=deriv) expects the multiplier
        # expressed in C^(deriv) coefficients, not Chebyshev — convert r^power up.
        rc = _convert_up(T, 0, deriv, N) * chebyshev_coefficients(T, power, N + 1, ri, ro)
        M = multiplication_matrix(rc, T(deriv), N + 1; vector_parity=0)
        op = M * Dk
    else
        op = Dk
    end
    return _convert_up(T, deriv, q_out, N) * op          # C^(0) -> C^(q_out)
end

"""
    recomb_dirichlet(T, N) -> SparseMatrixCSC

Trial recombination for homogeneous Dirichlet at both ends (u(±1)=0).
Columns φ_k = T_k − T_{k+2}, k=0..N−2. Size (N+1)×(N−1).
"""
function recomb_dirichlet(::Type{T}, N::Int) where {T<:Real}
    rows = Int[]; cols = Int[]; vals = T[]
    for k in 0:(N - 2)
        push!(rows, k + 1);     push!(cols, k + 1); push!(vals, one(T))
        push!(rows, k + 2 + 1); push!(cols, k + 1); push!(vals, -one(T))
    end
    return sparse(rows, cols, vals, N + 1, N - 1)
end

"""
    recomb_neumann(T, N) -> SparseMatrixCSC

Trial recombination for homogeneous Neumann at both ends (u'(±1)=0).
Columns φ_k = T_k − (k²/(k+2)²) T_{k+2}, k=0..N−2. Size (N+1)×(N−1).
"""
function recomb_neumann(::Type{T}, N::Int) where {T<:Real}
    rows = Int[]; cols = Int[]; vals = T[]
    for k in 0:(N - 2)
        push!(rows, k + 1); push!(cols, k + 1); push!(vals, one(T))
        c = T(k)^2 / T(k + 2)^2
        push!(rows, k + 2 + 1); push!(cols, k + 1); push!(vals, -c)
    end
    return sparse(rows, cols, vals, N + 1, N - 1)
end

"""
    recomb_clamped(T, N) -> SparseMatrixCSC

Trial recombination for clamped BCs at both ends (u(±1)=u'(±1)=0), 4th order.
Columns φ_k = T_k − [2(k+2)/(k+3)] T_{k+2} + [(k+1)/(k+3)] T_{k+4}, k=0..N−4.
Size (N+1)×(N−3). (Shen 1995 Chebyshev biharmonic basis.)
"""
function recomb_clamped(::Type{T}, N::Int) where {T<:Real}
    rows = Int[]; cols = Int[]; vals = T[]
    for k in 0:(N - 4)
        a = T(2) * T(k + 2) / T(k + 3)
        b = T(k + 1) / T(k + 3)
        push!(rows, k + 1);     push!(cols, k + 1); push!(vals, one(T))
        push!(rows, k + 2 + 1); push!(cols, k + 1); push!(vals, -a)
        push!(rows, k + 4 + 1); push!(cols, k + 1); push!(vals, b)
    end
    return sparse(rows, cols, vals, N + 1, N - 3)
end

"""
    recomb_from_functionals(funcs) -> Matrix

Degree-local trial recombination for the q×(N+1) boundary functionals `funcs`:
column k is φ_k = T_k + Σ_{j=1}^{q} c_kj T_{k+j}, with coefficients chosen so that
every functional vanishes (Shen's construction). Returns (N+1)×(N+1−q). Each φ_k
stays near degree k, so Galerkin forms built from its derivatives keep the low modes
accurate at large N. An orthonormal nullspace basis instead mixes every degree into
every column, and those forms lose about N⁴·eps.
"""
function recomb_from_functionals(funcs::AbstractMatrix{T}) where {T}
    q, n = size(funcs)
    q == 0 && return Matrix{T}(I, n, n)
    R = zeros(T, n, n - q)
    for k in 1:(n - q)
        block = funcs[:, (k + 1):(k + q)]
        scaled = block ./ maximum(abs, block; dims=2)
        cond(scaled) < inv(sqrt(eps(real(float(T))))) || throw(ArgumentError(
            "Boundary functionals are nearly dependent on degrees $(k) to $(k + q - 1); " *
            "cannot build a degree-local recombination"))
        R[k, k] = one(T)
        R[(k + 1):(k + q), k] .= block \ (-funcs[:, k])
    end
    return R
end

"""
    galerkin_block(L_band, R_trial, M_test) -> Matrix

Project a banded C^(q) operator into Galerkin form: keep the leading `M_test`
rows (P_M restriction), right-multiply by the trial recombination `R_trial`.
"""
function galerkin_block(L_band::AbstractMatrix, R_trial::AbstractMatrix, M_test::Int)
    # Restrict to the leading `M_test` rows BEFORE multiplying: the product's
    # trailing rows are discarded anyway, so computing them is pure waste. This
    # multiplies an `M_test × N` block by `R_trial` instead of the full `(N+1) × N`.
    return Matrix(L_band[1:M_test, :] * R_trial)
end

"""
    recomb_poloidal_velocity(T, N, ri, ro; bci=1, bco=1)

Poloidal velocity trial basis with independent inner/outer mechanical BCs.
Each boundary imposes `u = 0`, plus `u' = 0` for no-slip (1) or `r·u'' = 0`
for stress-free (0), matching the tau functionals. Size (N+1)×(N−3).
"""
function recomb_poloidal_velocity(::Type{T}, N::Int, ri::Real, ro::Real;
                                   bci::Int=1, bco::Int=1) where {T<:Real}
    bci == bco == 1 && return recomb_clamped(T, N)
    scale = T(_radial_scale(ri, ro))
    funcs = Matrix{T}(undef, 4, N + 1)
    for (i, (b, bc)) in enumerate(((:outer, bco), (:inner, bci)))
        rb = T(_boundary_radius(ri, ro, b))
        funcs[2i - 1, :] = _chebyshev_boundary_values(N, b, T)
        funcs[2i, :] = bc == 1 ? scale .* _chebyshev_boundary_derivative(N, b, T) :
                                rb .* scale^2 .* _chebyshev_boundary_second_derivative(N, b, T)
    end
    return T.(recomb_from_functionals(funcs))
end

"""
    recomb_toroidal_velocity(T, N, ri, ro; bci=1, bco=1)

Toroidal velocity trial basis with independent inner/outer mechanical BCs:
`v = 0` for no-slip (1), `-r·v' + v = 0` for stress-free (0).
Matches the tau functionals. Size (N+1)×(N−1).
"""
function recomb_toroidal_velocity(::Type{T}, N::Int, ri::Real, ro::Real;
                                   bci::Int=1, bco::Int=1) where {T<:Real}
    bci == bco == 1 && return recomb_dirichlet(T, N)
    return T.(recomb_from_functionals(_toroidal_velocity_functionals(T, N, ri, ro, bci, bco)))
end

"""Toroidal velocity boundary functionals (outer, inner): `v` for no-slip (1),
`-r·v' + v` for stress-free (0)."""
function _toroidal_velocity_functionals(::Type{T}, N::Int, ri::Real, ro::Real,
                                        bci::Int, bco::Int) where {T<:Real}
    scale = T(_radial_scale(ri, ro))
    funcs = Matrix{T}(undef, 2, N + 1)
    for (i, (b, bc)) in enumerate(((:outer, bco), (:inner, bci)))
        rb    = T(_boundary_radius(ri, ro, b))
        vals  = _chebyshev_boundary_values(N, b, T)
        deriv = _chebyshev_boundary_derivative(N, b, T)
        funcs[i, :] = bc == 1 ? vals : -rb .* scale .* deriv .+ vals
    end
    return funcs
end

"""Poloidal trial basis with stress-free conditions at both boundaries."""
recomb_poloidal_stressfree(T::Type, N::Int, ri::Real, ro::Real) =
    recomb_poloidal_velocity(T, N, ri, ro; bci=0, bco=0)

"""Toroidal trial basis with stress-free conditions at both boundaries."""
recomb_toroidal_stressfree(T::Type, N::Int, ri::Real, ro::Real) =
    recomb_toroidal_velocity(T, N, ri, ro; bci=0, bco=0)

"""
    recomb_temperature(T, N, ri, ro; bci=0, bco=0)

Temperature trial basis with independent inner/outer thermal BCs:
fixed temperature (0) or fixed flux (1). Size (N+1)×(N−1).
"""
function recomb_temperature(::Type{T}, N::Int, ri::Real, ro::Real;
                              bci::Int=0, bco::Int=0) where {T<:Real}
    bci == bco == 0 && return recomb_dirichlet(T, N)
    bci == bco == 1 && return recomb_neumann(T, N)
    scale = T(_radial_scale(ri, ro))
    funcs = Matrix{T}(undef, 2, N + 1)
    for (i, (b, bc)) in enumerate(((:outer, bco), (:inner, bci)))
        funcs[i, :] = bc == 0 ? _chebyshev_boundary_values(N, b, T) :
                                scale .* _chebyshev_boundary_derivative(N, b, T)
    end
    return T.(recomb_from_functionals(funcs))
end

"""
    recomb_magnetic_poloidal(T, N, ℓ, ri, ro; bci=0, bco=0) -> Matrix

Trial recombination for the poloidal magnetic scalar `f` at degree `ℓ` (order q=2).
Insulating boundaries (`bci=bco=0`) impose the ℓ-dependent Robin conditions used by
`apply_magnetic_boundary_conditions!`: outer `(ℓ+1)·f + ro·f' = 0`, inner
`ℓ·f − ri·f' = 0`. Built as the nullspace of those functionals (ℓ-dependent ⇒ rebuilt
per ℓ). Perfect-conductor boundaries (`2`) impose `f = 0` for this potential.
A finite-conductivity core requires its own unknowns and interface equations;
it cannot be represented by this shell-only recombination and is rejected.
Size (N+1)×(N−1).
"""
function recomb_magnetic_poloidal(::Type{T}, N::Int, ℓ::Int, ri::Real, ro::Real;
                                  bci::Int=0, bco::Int=0) where {T<:Real}
    bci in (0, 2) && bco in (0, 2) || throw(ArgumentError(
        "Magnetic poloidal recombination supports insulating (0) or perfect-conductor (2) walls. Use MHD tau assembly for an evolving conducting core."))
    scale = T(_radial_scale(ri, ro))
    rb_o = T(_boundary_radius(ri, ro, :outer)); rb_i = T(_boundary_radius(ri, ro, :inner))
    val_o = _chebyshev_boundary_values(N, :outer, T)
    val_i = _chebyshev_boundary_values(N, :inner, T)
    der_o = scale .* _chebyshev_boundary_derivative(N, :outer, T)
    der_i = scale .* _chebyshev_boundary_derivative(N, :inner, T)
    f_out = bco == 0 ? ((ℓ + 1) .* val_o .+ rb_o .* der_o) : val_o
    f_in  = bci == 0 ? (ℓ .* val_i .- rb_i .* der_i)        : val_i
    return T.(recomb_from_functionals(vcat(f_out', f_in')))
end
