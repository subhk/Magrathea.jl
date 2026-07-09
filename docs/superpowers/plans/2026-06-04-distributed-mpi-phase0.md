# Distributed MPI — Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `backend=:slepc` as a distributed SLEPc eigensolve over `MPI.COMM_WORLD` (any rank count) with replicated Julia assembly — each rank inserts only its owned PETSc rows, MUMPS factorizes in parallel, eigenvectors gather to rank 0.

**Architecture:** Replace the serial `_slepc_solve` body with one `COMM_WORLD` distributed path (N=1 degenerates to rank-0-owns-all). Explicit `slepc_init!`/`slepc_finalize!` lifecycle via core Ref hooks the extension fills. Eigenvalues identical on all ranks; eigenvectors full on rank 0, empty on workers — every divert site is hardened so an `nev`-length ordering never indexes the empty worker eigenvector matrix.

**Tech Stack:** Julia 1.12, package extension, PetscWrap 0.1.5 / SlepcWrap 0.1.3, MPI, MUMPS (in user's PETSc), raw `ccall` into `libpetsc`/`libslepc` for unwrapped primitives (`VecScatterCreateToZero`, `EPSSetDimensions`).

**Spec:** `docs/superpowers/specs/2026-06-04-distributed-mpi-phase0-design.md`. Program roadmap: `project_distributed_mpi_roadmap` memory.

---

## Environment / workflow notes (read first)

- Julia binary (launcher broken): `JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia`. Bash tool needs `dangerouslyDisableSandbox: true`. Julia 1.12.x.
- **No PETSc/MPI here.** Tasks 1 & 3 are fully verifiable locally (pure Julia). Task 2 (distributed solve + raw ccalls) **cannot precompile or run here** — verify by syntax parse + symbol-audit against installed source; runtime correctness is the user's `mpirun` cluster validation (Task 4).
- **Commits:** standing rule = no `git commit` without explicit permission. Each task ends with a commit step — **pause and ask before running it.**

---

## File Structure

- `src/Stability/solver.jl` (modify) — add `_SLEPC_INIT`/`_SLEPC_FINALIZE` Refs, `slepc_init!`/`slepc_finalize!`, pure helper `_petsc_owned_nnz`; extend the `:slepc` error text.
- `src/Magrathea.jl` (modify) — export `slepc_init!`, `slepc_finalize!`.
- `src/Stability/linear.jl`, `src/Stability/triglobal.jl`, `src/solve.jl` (modify) — harden the three `:slepc` divert sites against empty worker eigenvectors.
- `ext/MagratheaSlepcExt/raw_petsc.jl` (create) — raw `ccall`s: `_vec_scatter_to_zero`, `_eps_set_dimensions`.
- `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl` (modify) — distributed `_slepc_solve`, `_slepc_init!`/`_slepc_finalize!`, `_INITIALIZED` guard, register 3 hooks, `include("raw_petsc.jl")`.
- `test/slepc_backend.jl` (modify) — `_petsc_owned_nnz` units, init/finalize stub-error tests, empty-eigenvector tolerance tests, guarded MPI test.

---

## Task 1: Core lifecycle hooks + `_petsc_owned_nnz` (fully verifiable here)

**Files:**
- Modify: `src/Stability/solver.jl`, `src/Magrathea.jl`
- Test: `test/slepc_backend.jl`

- [ ] **Step 1: Write failing tests** (append to `test/slepc_backend.jl`)

```julia
@testset "_petsc_owned_nnz splits diagonal/off-diagonal blocks" begin
    # 4x4: row1→(1,1),(1,3); row2→(2,2); row3→(3,1),(3,4); row4→(4,4)
    M = sparse([1,1,2,3,3,4], [1,3,2,1,4,4], ComplexF64[1,1,1,1,1,1], 4, 4)
    d, o = Magrathea._petsc_owned_nnz(M, 0, 2)      # own rows 1,2; col band [0,2)
    @test d == [1, 1] && o == [1, 0]
    d2, o2 = Magrathea._petsc_owned_nnz(M, 2, 4)    # own rows 3,4; col band [2,4)
    @test d2 == [1, 1] && o2 == [1, 0]
    d3, o3 = Magrathea._petsc_owned_nnz(M, 0, 4)    # own all; everything diagonal-band
    @test o3 == [0, 0, 0, 0] && d3 == [2, 1, 2, 1]
    d4, o4 = Magrathea._petsc_owned_nnz(M, 2, 2)    # empty block
    @test isempty(d4) && isempty(o4)
end

@testset "slepc_init!/finalize! error without extension" begin
    @test_throws ErrorException Magrathea.slepc_init!()
    e = try Magrathea.slepc_init!(); catch err; err end
    @test occursin("SlepcWrap", sprint(showerror, e))
    @test_throws ErrorException Magrathea.slepc_finalize!()
end
```

- [ ] **Step 2: Run, verify FAIL** — `$JL --project=. -e 'using Test; @testset "t" begin include("test/slepc_backend.jl") end'` → FAIL (`_petsc_owned_nnz`/`slepc_init!` undefined).

- [ ] **Step 3: Add to `src/Stability/solver.jl`** (next to the existing `_SLEPC_SOLVER` block)

```julia
const _SLEPC_INIT     = Ref{Union{Nothing,Function}}(nothing)
const _SLEPC_FINALIZE = Ref{Union{Nothing,Function}}(nothing)

"""Initialize SLEPc once per process (collective). Pass PETSc/SLEPc option string.
Requires the MagratheaSlepcExt extension (`using PetscWrap, SlepcWrap`)."""
function slepc_init!(opts::AbstractString="")
    f = _SLEPC_INIT[]
    f === nothing && error(
        "slepc_init! requires the SLEPc extension: `using PetscWrap, SlepcWrap` " *
        "(complex-scalar PETSc build with PETSC_DIR/SLEPC_DIR set).")
    return f(opts)
end

"""Finalize SLEPc once at process end. Requires the MagratheaSlepcExt extension."""
function slepc_finalize!()
    f = _SLEPC_FINALIZE[]
    f === nothing && error(
        "slepc_finalize! requires the SLEPc extension: `using PetscWrap, SlepcWrap`.")
    return f()
end

"""
    _petsc_owned_nnz(M, rstart, rend) -> (d_nnz, o_nnz)

Per-row diagonal/off-diagonal nonzero counts for the owned PETSc row block
`[rstart, rend)` (0-based, half-open) of a replicated `SparseMatrixCSC`, used for
`MatMPIAIJSetPreallocation`. Assumes the column-ownership band equals the row band
`[rstart, rend)` (PETSc default for square MPIAIJ with matching row/col layout).
Returns two `Vector{Int}` of length `rend - rstart`.
"""
function _petsc_owned_nnz(M::SparseMatrixCSC, rstart::Int, rend::Int)
    nloc = rend - rstart
    d = zeros(Int, nloc)
    o = zeros(Int, nloc)
    rows = rowvals(M)
    for col in 1:size(M, 2)
        c0 = col - 1
        for k in nzrange(M, col)
            r0 = rows[k] - 1
            if rstart <= r0 < rend
                i = r0 - rstart + 1
                if rstart <= c0 < rend
                    d[i] += 1
                else
                    o[i] += 1
                end
            end
        end
    end
    return d, o
end
```

- [ ] **Step 4: Extend the `:slepc` error text** — in `_solve_generalized_eigen_slepc` (solver.jl), append to the existing error string: `" Also call Magrathea.slepc_init!() once before solving."`

- [ ] **Step 5: Export in `src/Magrathea.jl`** — add `slepc_init!,` and `slepc_finalize!,` to the `export` list (near other public solve symbols).

- [ ] **Step 6: Run tests, verify PASS** — same command as Step 2 → all PASS.

- [ ] **Step 7: Regression — default suite green** — `$JL --project=. test/runtests.jl 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done`.

- [ ] **Step 8: Commit (ASK USER FIRST)** — `git add src/Stability/solver.jl src/Magrathea.jl test/slepc_backend.jl` / `git commit -m "feat(slepc): add explicit lifecycle hooks + distributed preallocation helper"`

---

## Task 2: Distributed `_slepc_solve` + raw ccalls (NOT runnable here)

**Files:**
- Create: `ext/MagratheaSlepcExt/raw_petsc.jl`
- Modify: `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl`

> Cannot precompile/run here (no PETSc). Verify: (a) `Meta.parseall` of both files; (b) symbol-audit EVERY PETSc/SLEPc symbol + ccall signature against installed PetscWrap 0.1.5 (`~/.julia/packages/PetscWrap/pFwKF/src`) and SlepcWrap 0.1.3 (`~/.julia/packages/SlepcWrap/VVZOZ/src`); (c) `using Magrathea` → `CORE_OK`. Follow PetscWrap's ccall convention: pass `PetscMat`/`PetscVec`/`SlepcEPS` wrappers directly into ccalls (their `cconvert` yields the `Ptr{Cvoid}` handle); reference `PetscWrap.libpetsc`, `PetscWrap.PetscErrorCode`, `PetscWrap.CVec`, `PetscWrap.PetscInt`, `PetscWrap.PetscScalar`, `SlepcWrap.libslepc`.

- [ ] **Step 1: Create `ext/MagratheaSlepcExt/raw_petsc.jl`**

```julia
# Raw ccall bindings for primitives SlepcWrap 0.1.3 / PetscWrap 0.1.5 do not wrap.
# AUDIT each signature against the installed PETSc/SLEPc headers before trusting.

const CVecScatter = Ptr{Cvoid}
const SCATTER_FORWARD = Cint(0)
const _INSERT_VALUES_C = Cint(1)   # PETSc InsertMode INSERT_VALUES

"""Set the number of requested eigenpairs on an EPS (SlepcWrap 0.1.3 lacks a wrapper).
`EPSSetDimensions(eps, nev, ncv=PETSC_DECIDE, mpd=PETSC_DECIDE)`."""
function _eps_set_dimensions(eps, nev::Integer)
    PETSC_DECIDE = PetscWrap.PETSC_DECIDE
    err = ccall((:EPSSetDimensions, SlepcWrap.libslepc), PetscWrap.PetscErrorCode,
                (Ptr{Cvoid}, PetscWrap.PetscInt, PetscWrap.PetscInt, PetscWrap.PetscInt),
                eps.ptr[], PetscWrap.PetscInt(nev), PETSC_DECIDE, PETSC_DECIDE)
    @assert iszero(err)
    return nothing
end

"""Gather a distributed PETSc vector to rank 0 as a `Vector{ComplexF64}`.
Rank 0 gets the full length-`n` array; other ranks get an empty vector.
Encapsulates VecScatterCreateToZero / Begin / End / VecGetArray / destroys."""
function _vec_scatter_to_zero(v::PetscWrap.PetscVec)
    ctx = Ref{CVecScatter}()
    seq = Ref{PetscWrap.CVec}()
    e1 = ccall((:VecScatterCreateToZero, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
               (PetscWrap.CVec, Ptr{CVecScatter}, Ptr{PetscWrap.CVec}), v, ctx, seq)
    @assert iszero(e1)
    e2 = ccall((:VecScatterBegin, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
               (CVecScatter, PetscWrap.CVec, PetscWrap.CVec, Cint, Cint),
               ctx[], v, seq[], _INSERT_VALUES_C, SCATTER_FORWARD)
    @assert iszero(e2)
    e3 = ccall((:VecScatterEnd, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
               (CVecScatter, PetscWrap.CVec, PetscWrap.CVec, Cint, Cint),
               ctx[], v, seq[], _INSERT_VALUES_C, SCATTER_FORWARD)
    @assert iszero(e3)

    # seq[] is sequential on rank 0 (length n) and length 0 elsewhere.
    nref = Ref{PetscWrap.PetscInt}()
    ccall((:VecGetSize, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
          (PetscWrap.CVec, Ref{PetscWrap.PetscInt}), seq[], nref)
    n = Int(nref[])
    out = Vector{ComplexF64}(undef, n)
    if n > 0
        aref = Ref{Ptr{PetscWrap.PetscScalar}}()
        ccall((:VecGetArray, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
              (PetscWrap.CVec, Ref{Ptr{PetscWrap.PetscScalar}}), seq[], aref)
        arr = unsafe_wrap(Array, aref[], n; own=false)
        out .= ComplexF64.(arr)
        ccall((:VecRestoreArray, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
              (PetscWrap.CVec, Ref{Ptr{PetscWrap.PetscScalar}}), seq[], aref)
    end
    ccall((:VecScatterDestroy, PetscWrap.libpetsc), PetscWrap.PetscErrorCode, (Ptr{CVecScatter},), ctx)
    ccall((:VecDestroy, PetscWrap.libpetsc), PetscWrap.PetscErrorCode, (Ptr{PetscWrap.CVec},), seq)
    return out
end
```

- [ ] **Step 2: Rewrite `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl`**

```julia
module MagratheaSlepcExt

using Magrathea
using SparseArrays
using PetscWrap
using SlepcWrap

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
    rstart, rend = MatGetOwnershipRange(mat)          # 0-based, half-open
    d, o = Magrathea._petsc_owned_nnz(M, rstart, rend)
    MatMPIAIJSetPreallocation(mat, 0, d, 0, o)
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

function _slepc_solve(A::SparseMatrixCSC, B::SparseMatrixCSC;
                      nev::Int, sigma, which::Symbol, selection::Symbol,
                      tol::Float64, maxiter::Int, verbosity::Int=0)
    _INITIALIZED[] || error("call Magrathea.slepc_init!() once before a :slepc solve")
    PetscScalar <: Real &&
        error("PETSc/SLEPc must be built with complex scalars (--with-scalar-type=complex)")
    size(A) == size(B) || throw(DimensionMismatch("A and B must match"))
    n = size(A, 1)

    target = sigma === nothing ?
        (which === :LR ? ComplexF64(10, 0) :
         which === :LI ? ComplexF64(0, 10) : ComplexF64(1, 0)) :
        ComplexF64(sigma)

    Amat = _to_petsc_dist(A, n)
    Bmat = _to_petsc_dist(B, n)

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
    vecs = rank == 0 ? Matrix{ComplexF64}(undef, n, nout) :
                       Matrix{ComplexF64}(undef, n, 0)
    vr, vi = MatCreateVecs(Amat)
    for j in 0:(nout - 1)
        vpr, vpi, vecr, veci = EPSGetEigenpair(eps, j, vr, vi)
        vals[j + 1] = ComplexF64(vpr, vpi)          # collective: same on all ranks
        full = _vec_scatter_to_zero(vecr)           # length n on rank 0, else 0
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

function __init__()
    Magrathea._SLEPC_SOLVER[]   = _slepc_solve
    Magrathea._SLEPC_INIT[]     = _slepc_init!
    Magrathea._SLEPC_FINALIZE[] = _slepc_finalize!
    return nothing
end

end # module
```
Note: `MPI` is used directly now (`MPI.COMM_WORLD`, `MPI.Comm_rank/size`). PetscWrap re-exports `MPI`, but to be safe add `using PetscWrap.MPI` or `import MPI` — confirm during the audit which makes `MPI` available; if neither, add `MPI` to `[weakdeps]` and the extension trigger.

- [ ] **Step 3: Symbol + MPI-availability audit.** For every new symbol (`MatMPIAIJSetPreallocation` 5-arg form `(mat, dnz, d_nnz, onz, o_nnz)`, `MatGetOwnershipRange`, `MatCreateVecs`, `EPSGetEigenpair`, `MPI.COMM_WORLD/Comm_rank/Comm_size`, `PetscWrap.PetscInt/PetscScalar/CVec/libpetsc/PETSC_DECIDE`, `SlepcWrap.libslepc`, `eps.ptr`/`SlepcEPS` field, `VecScatterCreateToZero`/`Begin`/`End`/`Destroy` arg types) confirm against the installed source and PETSc man pages. Produce a table: symbol → found (file:line) / adjusted. Confirm how `MPI` is reachable in the extension and wire it (weakdep if needed).

- [ ] **Step 4: Parse check** — `$JL -e 'for f in ("ext/MagratheaSlepcExt/raw_petsc.jl","ext/MagratheaSlepcExt/MagratheaSlepcExt.jl"); Meta.parseall(read(f,String)); end; println("PARSE_OK")'` → `PARSE_OK`.

- [ ] **Step 5: Core still PETSc-free** — `$JL --project=. -e 'using Magrathea; println("CORE_OK")'` (sandbox off) → `CORE_OK`.

- [ ] **Step 6: Commit (ASK USER FIRST)** — `git add ext/MagratheaSlepcExt/` / `git commit -m "feat(slepc): distributed COMM_WORLD solve, MUMPS shift-invert, rank-0 gather"`

---

## Task 3: Harden divert sites for empty worker eigenvectors (verifiable here)

**Files:**
- Modify: `src/Stability/linear.jl` (the `:slepc` divert), `src/Stability/triglobal.jl` (the `:slepc` divert)
- Test: `test/slepc_backend.jl`

Rationale: on workers, `_slepc_solve` returns `nev` eigenvalues but an `n×0` eigenvector matrix. Any ordering/`perm` of length `nev` must NOT index the empty matrix.

- [ ] **Step 1: Write failing test** (append to `test/slepc_backend.jl`) — simulate the worker contract directly against the ordering logic:

```julia
@testset "empty worker eigenvectors survive ordering" begin
    # Worker contract: nev eigenvalues, 0-column eigenvector matrix.
    vals = ComplexF64[0.3+1im, -0.2+0im, 0.5-1im]
    vecs0 = Matrix{ComplexF64}(undef, 10, 0)
    perm = sortperm(real.(vals); rev=true)
    # The guarded expression used at every divert site must not BoundsError:
    out = size(vecs0, 2) == 0 ? vecs0 : vecs0[:, perm]
    @test size(out) == (10, 0)
    # _eigvecs_to_matrix must accept an empty vector-of-vectors with nev eigenvalues
    empty_vv = Vector{Vector{ComplexF64}}()
    M = Magrathea._eigvecs_to_matrix(vals, empty_vv, Float64)
    @test size(M, 2) == 0
end
```

- [ ] **Step 2: Run, verify the test passes for the guarded form** (it documents the required guard; run `$JL --project=. -e 'using Test; @testset "t" begin include("test/slepc_backend.jl") end'`). If `_eigvecs_to_matrix` errors on the empty vector-of-vectors, that's a real gap → its `AbstractVector{<:AbstractVector}` method already returns `Matrix{Complex{T}}(undef,0,length(eigenvalues))` for `isempty`; confirm. Expected: PASS.

- [ ] **Step 3: Guard the triglobal divert** — in `src/Stability/triglobal.jl`, change the `:slepc` divert's return from `return vals_s[perm], vecs_s[:, perm]` to:

```julia
        return vals_s[perm], (size(vecs_s, 2) == 0 ? vecs_s : vecs_s[:, perm])
```

- [ ] **Step 4: Guard the constrained-hydro divert** — in `src/Stability/linear.jl`, the `:slepc` branch builds `vecs_full = [_reconstruct_full_vector(reduction, vecs_s[:, j]) for j in 1:size(vecs_s, 2)]` (empty on workers) then returns `vals_s[ordering], vecs_full[ordering], info_s`. Change the return to not index an empty `vecs_full`:

```julia
        return vals_s[ordering], (isempty(vecs_full) ? vecs_full : vecs_full[ordering]), info_s
```

(The galerkin divert in `src/solve.jl` already feeds `evecs_full` straight to `_eigvecs_to_matrix` without an `ordering` index, so an empty `evecs_full` yields a `0×nev` matrix — no change needed. Confirm by reading it.)

- [ ] **Step 5: Run the full default suite** — `$JL --project=. test/runtests.jl 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done` (these guards are no-ops for the non-empty `:krylovkit` default path; this confirms no regression).

- [ ] **Step 6: Commit (ASK USER FIRST)** — `git add src/Stability/linear.jl src/Stability/triglobal.jl test/slepc_backend.jl` / `git commit -m "fix(slepc): tolerate empty worker eigenvectors at divert sites"`

---

## Task 4: Guarded MPI cluster test + docs

**Files:**
- Modify: `test/slepc_backend.jl`, `README.md`, the existing serial SLEPc README section.

- [ ] **Step 1: Replace the prior `PETSC_DIR`-guarded serial test** with a distributed-aware one (append/replace in `test/slepc_backend.jl`):

```julia
@testset "Distributed SLEPc lifecycle + solve (requires PETSc+MUMPS under MPI)" begin
    if !haskey(ENV, "PETSC_DIR")
        @info "PETSC_DIR unset — skipping distributed SLEPc test (run under mpirun on a PETSc+MUMPS build)"
        @test true
    else
        @eval using PetscWrap, SlepcWrap
        Magrathea.slepc_init!("-eps_gen_non_hermitian -st_type sinvert -st_pc_type lu " *
                          "-st_pc_factor_mat_solver_type mumps -eps_target_magnitude")
        p = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1.0, ricb=0.35,
                      m=1, lmax=3, N=12, B0_type=dipole, B0_amplitude=1.0)
        op = MHDStabilityOperator(p)
        A, B, _, _ = assemble_mhd_matrices(op)
        vK, _, _ = Magrathea.solve_eigenvalue_problem(A, B; nev=4, sigma=0.0, backend=:krylovkit)
        vS, _, _ = Magrathea.solve_eigenvalue_problem(A, B; nev=4, sigma=0.0, backend=:slepc)
        @test isapprox(sort(real.(vK))[1:2], sort(real.(vS))[1:2]; rtol=1e-4)
        Magrathea.slepc_finalize!()
    end
end
```

- [ ] **Step 2: Verify it skips cleanly here** — `$JL --project=. -e 'using Test; @testset "t" begin include("test/slepc_backend.jl") end'` → all PASS, output shows the "skipping distributed SLEPc test" line.

- [ ] **Step 3: Update the README SLEPc section** to document distributed use:
  - Requires PETSc+SLEPc with **complex scalars** AND **MUMPS** (`--download-mumps` or system), plus MPI; set `PETSC_DIR`/`PETSC_ARCH`/`SLEPC_DIR`.
  - Driver pattern:
    ```julia
    using Magrathea, PetscWrap, SlepcWrap
    Magrathea.slepc_init!("-eps_gen_non_hermitian -st_type sinvert -st_pc_type lu -st_pc_factor_mat_solver_type mumps -eps_target_magnitude")
    result = solve(problem; backend=:slepc)
    # eigenvalues valid on all ranks; eigenvectors on rank 0 only:
    # if MPI.Comm_rank(MPI.COMM_WORLD) == 0  ... use result.eigenvectors ... end
    Magrathea.slepc_finalize!()
    ```
  - Launch: `mpirun -n N julia --project=. driver.jl`.
  - Note: replicated assembly (full matrix per rank); eigenvectors gathered to rank 0; `slepc_init!`/`slepc_finalize!` once per process.

- [ ] **Step 4: Full suite green** — `$JL --project=. test/runtests.jl 2>&1 | grep -iE "Error During Test|did not pass|skipping distributed"; echo done` → "skipping distributed" line + `done`, no failures.

- [ ] **Step 5: Commit (ASK USER FIRST)** — `git add test/slepc_backend.jl README.md` / `git commit -m "test+docs: distributed SLEPc guarded test and mpirun usage"`

---

## Self-review notes

- **Spec coverage:** lifecycle hooks + `slepc_init!`/`slepc_finalize!` (T1); `_petsc_owned_nnz` pure helper + units (T1); distributed `_to_petsc_dist` owned-row insert + preallocation (T2); MUMPS shift-invert via option string + per-solve `EPSSetTarget`/`_eps_set_dimensions` (T2); rank-0 `VecScatterCreateToZero` gather (T2 raw_petsc.jl); worker empty-eigenvector contract hardened at all divert sites (T3); error handling — uninit/real-scalar/nconv==0/extension-absent (T1+T2); guarded MPI test + docs (T4). All spec sections mapped.
- **Type/contract consistency:** `_slepc_solve` returns `(Vector{ComplexF64}, Matrix{ComplexF64}, Dict)` on all ranks (eigenvectors `n×nev` rank 0 / `n×0` workers); divert sites guard `size(vecs,2)==0` before applying an `nev`-length `perm`/`ordering`. Hook names `_SLEPC_SOLVER`/`_SLEPC_INIT`/`_SLEPC_FINALIZE`; helpers `_petsc_owned_nnz`, `_eps_set_dimensions`, `_vec_scatter_to_zero`, `_to_petsc_dist`, `_sort_indices_local` used consistently.
- **Verification honesty:** T1 & T3 run here (pure Julia); T2 verified by parse + symbol-audit + CORE_OK only; T4 skips here and is the user's `mpirun` cluster validation. Every raw ccall flagged for audit against installed headers.
