module MagratheaSlepcExt

using Magrathea
using SparseArrays
using PetscWrap
using SlepcWrap

# PetscWrap does `using MPI` internally, so its MPI module binding is reachable as
# PetscWrap.MPI. Aliasing it (rather than adding MPI as a separate weakdep) also
# guarantees we use the exact same MPI.Comm type PetscWrap's ccalls expect.
const MPI = PetscWrap.MPI

include("raw_petsc.jl")

const _INITIALIZED = Ref(false)

function _slepc_init!(opts::AbstractString="")
    if !_INITIALIZED[]
        SlepcInitialize(String(opts))
        _INITIALIZED[] = true
    end
    return nothing
end

function _slepc_finalize!()
    if _INITIALIZED[]
        SlepcFinalize()
        _INITIALIZED[] = false
    end
    return nothing
end

# Build a distributed MPIAIJ PETSc matrix from the full (replicated) Julia CSC,
# inserting only this rank's owned rows.
function _to_petsc_dist(M::SparseMatrixCSC, n::Int)
    mat = MatCreate(MPI.COMM_WORLD)
    MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, n, n)
    MatSetFromOptions(mat)
    rstart, rend = MatGetOwnershipRange(mat)            # 0-based, half-open
    d, o = Magrathea._petsc_owned_nnz(M, Int(rstart), Int(rend))
    PI = PetscWrap.PetscInt
    MatMPIAIJSetPreallocation(mat, PI(0), PI.(d), PI(0), PI.(o))
    rows = rowvals(M); vals = nonzeros(M)
    @inbounds for col in 1:size(M, 2)
        for k in nzrange(M, col)
            r0 = rows[k] - 1
            if rstart <= r0 < rend
                MatSetValue(mat, r0, col - 1, PetscScalar(vals[k]), INSERT_VALUES)
            end
        end
    end
    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY)
    return mat
end

# Build a distributed (possibly RECTANGULAR) `nrows×ncols` PETSc matrix from the full
# (replicated) Julia CSC, inserting only this rank's owned rows. Used for the sparse
# constraint-projection matrices S (n_reduced×n_full) and P (n_full×n_reduced), whose
# row/column ownership bands do not coincide, so the exact d/o split used by the square
# path does not apply. These matrices are built once and are tiny relative to the EPS
# factorization, so we use `MatSetUp` (type-agnostic default preallocation) plus
# `MAT_NEW_NONZERO_ALLOCATION_ERR=false` to allow dynamic allocation rather than compute
# an exact preallocation. (Perf detail only — correctness is unaffected.)
function _to_petsc_dist(M::SparseMatrixCSC, nrows::Int, ncols::Int)
    mat = MatCreate(MPI.COMM_WORLD)
    MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, nrows, ncols)
    MatSetFromOptions(mat)
    MatSetUp(mat)                                       # default preallocation (any type)
    PetscWrap.MatSetOption(mat, PetscWrap.MAT_NEW_NONZERO_ALLOCATION_ERR, false)
    rstart, rend = MatGetOwnershipRange(mat)            # 0-based, half-open (rows)
    rows = rowvals(M); vals = nonzeros(M)
    @inbounds for col in 1:size(M, 2)
        for k in nzrange(M, col)
            r0 = rows[k] - 1
            if rstart <= r0 < rend
                MatSetValue(mat, r0, col - 1, PetscScalar(vals[k]), INSERT_VALUES)
            end
        end
    end
    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY)
    return mat
end

"""
Distributed SLEPc solve of `A x = σ B x` over `MPI.COMM_WORLD`. Replicated Julia
assembly (each rank holds full `A`/`B`, inserts only owned rows). MUMPS shift-invert
comes from the option string set in `slepc_init!`. Returns the Magrathea contract
`(eigenvalues, eigenvectors, info)`: eigenvalues on all ranks; eigenvectors full
`n×nev` on rank 0, empty `n×0` on workers. Requires a complex-scalar PETSc build.
"""
function _slepc_solve(A::SparseMatrixCSC, B::SparseMatrixCSC;
                      nev::Int, sigma, which::Symbol, selection::Symbol,
                      tol::Float64, maxiter::Int, verbosity::Int=0)
    _INITIALIZED[] || error("call Magrathea.slepc_init!() once before a :slepc solve")
    PetscScalar <: Real &&
        error("PETSc/SLEPc must be built with complex scalars (--with-scalar-type=complex)")
    size(A) == size(B) || throw(DimensionMismatch("A and B must match"))
    n = size(A, 1)

    Amat = _to_petsc_dist(A, n)
    Bmat = _to_petsc_dist(B, n)

    return _eps_solve_and_gather(Amat, Bmat, n; nev=nev, sigma=sigma, which=which,
                                 selection=selection, tol=tol, maxiter=maxiter,
                                 verbosity=verbosity)
end

"""
Shared EPS shift-invert solve + rank-0 eigenvector gather, operating on already-built
distributed PETSc matrices `Amat`, `Bmat` (`n×n`). Owns the EPS lifecycle and destroys
`Amat`/`Bmat` before returning (NOT SlepcFinalize — that is the caller's explicit
lifecycle). Used by both `_slepc_solve` (replicated sparse path) and `_slepc_mhd_solve`
(distributed MHD path). Returns the Magrathea contract `(eigenvalues, eigenvectors, info)`:
eigenvalues identical on all ranks; eigenvectors full `n×nout` on rank 0, empty `n×0`
on workers.
"""
function _eps_solve_and_gather(Amat, Bmat, n::Int;
                              nev::Int, sigma, which::Symbol, selection::Symbol,
                              tol::Float64, maxiter::Int, verbosity::Int=0)
    target = sigma === nothing ?
        (which === :LR ? ComplexF64(10, 0) :
         which === :LI ? ComplexF64(0, 10) : ComplexF64(1, 0)) :
        ComplexF64(sigma)

    eps = EPSCreate(MPI.COMM_WORLD)
    EPSSetOperators(eps, Amat, Bmat)
    _eps_set_dimensions(eps, nev)
    EPSSetTarget(eps, PetscScalar(target))
    EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE)
    EPSSetFromOptions(eps)        # GNHEP + sinvert + MUMPS come from slepc_init! opts
    EPSSetUp(eps)
    EPSSolve(eps)

    nconv = EPSGetConverged(eps)
    nout = min(nconv, nev)
    nout == 0 && (EPSDestroy(eps); MatDestroy(Amat); MatDestroy(Bmat);
                  error("SLEPc returned no converged eigenpairs"))

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    vals = Vector{ComplexF64}(undef, nout)
    vecs = rank == 0 ? Matrix{ComplexF64}(undef, n, nout) : Matrix{ComplexF64}(undef, n, 0)
    vr, vi = MatCreateVecs(Amat)
    for j in 0:(nout - 1)
        vpr, vpi, vecr, veci = EPSGetEigenpair(eps, j, vr, vi)
        # Complex PETSc: EPSGetEigenpair returns the full eigenvalue in vpr (a
        # complex PetscScalar); vpi is the unused 0 imaginary slot. The 2-arg
        # ComplexF64(re, im) would coerce the complex vpr through Float64 -> InexactError.
        vals[j + 1] = ComplexF64(vpr)                 # collective: identical all ranks
        full = _vec_scatter_to_zero(vecr)             # length n on rank 0, else 0
        rank == 0 && (vecs[:, j + 1] .= full)
    end

    info = Dict{String,Any}("solver" => :slepc, "strategy" => :shift_invert,
        "target" => target, "nconv" => nconv, "selection" => selection,
        "ranks" => MPI.Comm_size(MPI.COMM_WORLD))

    EPSDestroy(eps); MatDestroy(Amat); MatDestroy(Bmat)   # NOT SlepcFinalize (explicit lifecycle)

    perm = _sort_indices_local(vals, selection)
    return vals[perm], (size(vecs, 2) == 0 ? vecs : vecs[:, perm]), info
end

function _sort_indices_local(ev::AbstractVector{<:Complex}, selection::Symbol)
    selection === :maxreal      ? sortperm(real.(ev); rev=true) :
    selection === :minabs       ? sortperm(abs.(ev)) :
    selection === :closest_real ? sortperm(abs.(real.(ev))) :
    error("Unknown selection strategy $(selection)")
end

# ---------------------------------------------------------------------------
# Distributed MHD assembly path
# ---------------------------------------------------------------------------

"""Create an empty distributed `n×n` MPIAIJ matrix with PETSc-decided local sizes,
returning `(mat, rstart, rend)` with the 0-based half-open owned row range."""
function _create_dist_mat(n::Int)
    mat = MatCreate(MPI.COMM_WORLD)
    MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, n, n)
    MatSetFromOptions(mat)
    rstart, rend = MatGetOwnershipRange(mat)
    return mat, Int(rstart), Int(rend)
end

"""Preallocate and fill the owned rows of a distributed matrix from COO triplets.
`rows`/`cols` are 1-based Julia indices, `vals` complex. Inserts only entries whose
row lies in this rank's owned band `[rstart, rend)` (0-based). The preallocation
counts (`_owned_coo_nnz`) are computed from the SAME triplet stream, so they match
exactly what is inserted."""
function _fill_dist_mat!(mat, rows, cols, vals, rstart::Int, rend::Int)
    PI = PetscWrap.PetscInt
    d, o = Magrathea._owned_coo_nnz(rows, cols, rstart, rend)
    MatMPIAIJSetPreallocation(mat, PI(0), PI.(d), PI(0), PI.(o))
    @inbounds for k in eachindex(rows)
        r0 = rows[k] - 1
        if rstart <= r0 < rend
            MatSetValue(mat, r0, cols[k] - 1, PetscScalar(vals[k]), INSERT_VALUES)
        end
    end
    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY)
    return mat
end

"""Enumerate the 1-based tau boundary-condition rows that the serial BC functions
(`apply_velocity_/magnetic_/temperature_boundary_conditions!`) overwrite, in the
SAME row layout as `_mhd_index_map` (sections u, v, f, g, h, each a contiguous
`(N+1)` block). Deterministic and identical on every rank.

Per-section rows overwritten (matching the serial code exactly):
- u: row_base+1, row_base+2, row_base+(N+1)-1, row_base+(N+1)
- v: row_base+1, row_base+(N+1)
- f: row_base+1, row_base+(N+1); additionally row_base+(N+1)-1 when bci_magnetic == 2
      (the perfect-conductor ICB second constraint, `row_icb2 = row_icb - 1`)
- g: row_base+1, row_base+(N+1)
- h: row_base+1, row_base+(N+1)
"""
function _mhd_bc_rows(op)
    params = op.params
    N = params.N
    n_per_mode = N + 1
    nb_u = length(op.ll_u); nb_v = length(op.ll_v)
    nb_f = length(op.ll_f); nb_g = length(op.ll_g); nb_h = length(op.ll_h)

    rows = Int[]
    # u section
    for k in 1:nb_u
        rb = (k - 1) * n_per_mode
        push!(rows, rb + 1, rb + 2, rb + n_per_mode - 1, rb + n_per_mode)
    end
    # v section
    for k in 1:nb_v
        rb = (nb_u + k - 1) * n_per_mode
        push!(rows, rb + 1, rb + n_per_mode)
    end
    # f section (poloidal magnetic)
    for k in 1:nb_f
        rb = (nb_u + nb_v + k - 1) * n_per_mode
        push!(rows, rb + 1, rb + n_per_mode)
        if params.bci_magnetic == 2
            push!(rows, rb + n_per_mode - 1)   # row_icb2 second perfect-conductor BC
        end
    end
    # g section (toroidal magnetic)
    for k in 1:nb_g
        rb = (nb_u + nb_v + nb_f + k - 1) * n_per_mode
        push!(rows, rb + 1, rb + n_per_mode)
    end
    # h section (temperature)
    for k in 1:nb_h
        rb = (nb_u + nb_v + nb_f + nb_g + k - 1) * n_per_mode
        push!(rows, rb + 1, rb + n_per_mode)
    end
    return rows
end

"""Apply the MHD tau boundary conditions on the distributed matrices `Amat`, `Bmat`,
mirroring the serial overwrites for this rank's OWNED rows only.

Strategy (value-exact, zero formula duplication): assemble the full *replicated*
serial sparse pencil with BCs already applied via `Magrathea.assemble_mhd_matrices(op)`
— this runs the identical, audited serial BC code. For every BC row the serial code
zeroed `A[row,:]`/`B[row,:]` and wrote a new A row (the B row stays all-zero). We
therefore (1) zero the same rows in both distributed Mats and (2) re-insert exactly
the serial A-row nonzeros, restricted to owned rows.

`MatZeroRows` is COLLECTIVE, so it is called once per matrix on EVERY rank with that
rank's owned BC-row subset (possibly empty) — guaranteeing uniform collective reach.

CAVEAT: builds the full serial A on each rank (replicated). This keeps BC values
provably identical to the serial path but does NOT realize a fully memory-distributed
BC application; the heavy distributed object is still the PETSc Mat/EPS factorization.
"""
function _apply_dist_bcs!(Amat, Bmat, op, rstart::Int, rend::Int)
    # Replicated serial assembly WITH boundary conditions applied (exact serial code).
    Aser, _Bser, _idofs, _info = Magrathea.assemble_mhd_matrices(op)

    bc_rows = _mhd_bc_rows(op)                       # 1-based, identical all ranks

    # Owned subset (0-based global rows) for the collective MatZeroRows.
    owned0 = Int[]
    for r in bc_rows
        r0 = r - 1
        rstart <= r0 < rend && push!(owned0, r0)
    end

    # COLLECTIVE: every rank calls once per matrix (owned subset may be empty).
    _mat_zero_rows(Amat, owned0)
    _mat_zero_rows(Bmat, owned0)

    # The COO interior rows for these BC positions had a different column pattern, and
    # MatZeroRows preserves the (now-zeroed) pattern; the serial BC entries may land in
    # columns outside the original preallocation. Allow those extra allocations rather
    # than error (small one-time cost, BC rows only).
    PetscWrap.MatSetOption(Amat, PetscWrap.MAT_NEW_NONZERO_ALLOCATION_ERR, false)

    # Re-insert the serial A BC-row entries for owned rows. Transpose once so a row
    # of Aser becomes a column of AserT, giving O(nnz_row) access via nzrange.
    AserT = sparse(transpose(Aser))
    rvals = rowvals(AserT); nzv = nonzeros(AserT)
    @inbounds for r in bc_rows
        r0 = r - 1
        (rstart <= r0 < rend) || continue
        for p in nzrange(AserT, r)                   # column r of AserT == row r of Aser
            col0 = rvals[p] - 1                      # 0-based global column
            MatSetValue(Amat, r0, col0, PetscScalar(nzv[p]), INSERT_VALUES)
        end
    end
    MatAssemblyBegin(Amat, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(Amat, MAT_FINAL_ASSEMBLY)
    MatAssemblyBegin(Bmat, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(Bmat, MAT_FINAL_ASSEMBLY)
    return nothing
end

"""Distributed MHD-aware SLEPc solve directly from an `MHDStabilityOperator`. Builds
distributed PETSc A/B from the COO triplets (owned rows only), applies the tau BCs on
the distributed Mats, then runs the shared EPS shift-invert solve + rank-0 gather.
Returns the Magrathea contract `(eigenvalues, eigenvectors, info)`. Requires a complex
PETSc build and a prior `Magrathea.slepc_init!()`."""
function _slepc_mhd_solve(op; nev::Int, sigma, which::Symbol, tol::Float64, maxiter::Int)
    _INITIALIZED[] || error("call Magrathea.slepc_init!() once before a :slepc solve")
    PetscScalar <: Real &&
        error("PETSc/SLEPc must be built with complex scalars (--with-scalar-type=complex)")
    n = op.matrix_size
    Amat, rstart, rend = _create_dist_mat(n)
    Bmat, _, _ = _create_dist_mat(n)
    coo = Magrathea._assemble_mhd_coo(op; owned_julia_rows=(rstart + 1):rend)
    _fill_dist_mat!(Amat, coo.A_rows, coo.A_cols, coo.A_vals, rstart, rend)
    _fill_dist_mat!(Bmat, coo.B_rows, coo.B_cols, coo.B_vals, rstart, rend)
    _apply_dist_bcs!(Amat, Bmat, op, rstart, rend)
    return _eps_solve_and_gather(Amat, Bmat, n; nev=nev, sigma=sigma, which=which,
                                 selection=:maxreal, tol=tol, maxiter=maxiter)
end

# ---------------------------------------------------------------------------
# Distributed constrained-reduction path (LinearStabilityOperator)
# ---------------------------------------------------------------------------

"""Form the distributed reduced matrix `S·A·P` via two `MatMatMult`s, destroying the
intermediate `S·A` product. `Smat`/`Amat`/`Pmat` are caller-owned and untouched."""
function _reduce_dist(Amat, Smat, Pmat)
    SA = _mat_mat_mult(Smat, Amat)
    red = _mat_mat_mult(SA, Pmat)
    MatDestroy(SA)
    return red
end

"""Distributed constrained-reduction SLEPc solve from a `LinearStabilityOperator`.

Builds the constraint reduction `S` (`n_reduced×n_full`) / `P` (`n_full×n_reduced`)
WITHOUT ever forming the full `A` (via `Magrathea._constraint_reduction_from_subblocks`
plus `Magrathea._constraint_projection_matrices`), assembles the tau pencil `(A, B)`
directly into distributed PETSc Mats from owned-row COO triplets
(`Magrathea._assemble_onset_coo`), distributes the sparse `S`/`P` as PETSc Mats, forms the
reduced pencil `Ared = S·A·P`, `Bred = S·B·P` via `MatMatMult`, then runs the shared
EPS shift-invert solve + rank-0 gather on the small reduced pencil. Reduced
eigenvectors are mapped back to full DOF coordinates with `P` on rank 0. Returns the
Magrathea contract `(eigenvalues, eigenvectors, info)`. Requires a complex PETSc build and
a prior `Magrathea.slepc_init!()`."""
function _slepc_constrained_solve(op; nev::Int, sigma, which::Symbol, tol::Float64, maxiter::Int)
    _INITIALIZED[] || error("call Magrathea.slepc_init!() once before a :slepc solve")
    PetscScalar <: Real && error("PETSc/SLEPc must be built with complex scalars")
    red = Magrathea._constraint_reduction_from_subblocks(op)        # no full A
    idofs = Magrathea._onset_interior_dofs(op)
    S, P = Magrathea._constraint_projection_matrices(red, idofs)
    nfull = red.n_full; nred = red.n_reduced
    # distributed owned-row assembly of A, B
    Amat, rs, re = _create_dist_mat(nfull)
    Bmat, _, _ = _create_dist_mat(nfull)
    coo = Magrathea._assemble_onset_coo(op; owned_julia_rows=(rs+1):re)
    _fill_dist_mat!(Amat, coo.A_rows, coo.A_cols, coo.A_vals, rs, re)
    _fill_dist_mat!(Bmat, coo.B_rows, coo.B_cols, coo.B_vals, rs, re)
    # distribute S, P; reduce; solve; reconstruct on rank 0
    Sdist = _to_petsc_dist(S, nred, nfull)
    Pdist = _to_petsc_dist(P, nfull, nred)
    Ared = _reduce_dist(Amat, Sdist, Pdist)
    Bred = _reduce_dist(Bmat, Sdist, Pdist)
    MatDestroy(Amat); MatDestroy(Bmat); MatDestroy(Sdist); MatDestroy(Pdist)
    vals, vecs_red, info = _eps_solve_and_gather(Ared, Bred, nred;
        nev=nev, sigma=sigma, which=which, selection=:maxreal, tol=tol, maxiter=maxiter)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    vecs_full = (rank == 0 && size(vecs_red, 2) > 0) ?
        Matrix{ComplexF64}(P * vecs_red) : Matrix{ComplexF64}(undef, nfull, 0)
    return vals, vecs_full, info
end

# ---------------------------------------------------------------------------
# Distributed triglobal path (CoupledModeProblem)
# ---------------------------------------------------------------------------

"""Distributed triglobal SLEPc solve from a `CoupledModeProblem`. Builds the
single-mode and mode-coupling operators (verbose=false), assembles the block-coupled
pencil `(A, B)` directly into distributed PETSc Mats from owned-row COO triplets
(`Magrathea._assemble_block_coo` with `owned_julia_rows`), then runs the shared EPS
shift-invert solve + rank-0 gather. Uses a small imaginary shift (`σ_target + 1e-6 i`)
to avoid singularity from boundary-condition rows, mirroring the serial triglobal
Krylov path. Returns `(eigenvalues, eigenvectors)` (eigenvectors full `n×nout` on rank
0, empty `n×0` on workers). Requires a complex PETSc build and a prior
`Magrathea.slepc_init!()`."""
function _slepc_triglobal_solve(problem; σ_target, nev::Int, tol::Float64, maxiter::Int)
    _INITIALIZED[] || error("call Magrathea.slepc_init!() once before a :slepc solve")
    PetscScalar <: Real && error("PETSc/SLEPc must be built with complex scalars")
    single = Magrathea.build_single_mode_operators(problem, false)
    coupling = Magrathea.build_mode_coupling_operators(problem, single, false)
    n = problem.total_dofs
    Amat, rs, re = _create_dist_mat(n); Bmat, _, _ = _create_dist_mat(n)
    coo = Magrathea._assemble_block_coo(problem, single, coupling; owned_julia_rows=(rs+1):re)
    _fill_dist_mat!(Amat, coo.A_rows, coo.A_cols, coo.A_vals, rs, re)
    _fill_dist_mat!(Bmat, coo.B_rows, coo.B_cols, coo.B_vals, rs, re)
    shift = ComplexF64(σ_target, 1e-6)
    vals, vecs, _ = _eps_solve_and_gather(Amat, Bmat, n;
        nev=nev, sigma=shift, which=:LR, selection=:maxreal, tol=tol, maxiter=maxiter)
    return vals, vecs
end

function __init__()
    Magrathea._SLEPC_SOLVER[]             = _slepc_solve
    Magrathea._SLEPC_MHD_SOLVER[]         = _slepc_mhd_solve
    Magrathea._SLEPC_CONSTRAINED_SOLVER[] = _slepc_constrained_solve
    Magrathea._SLEPC_TRIGLOBAL_SOLVER[]   = _slepc_triglobal_solve
    Magrathea._SLEPC_INIT[]               = _slepc_init!
    Magrathea._SLEPC_FINALIZE[]           = _slepc_finalize!
    return nothing
end

end # module
