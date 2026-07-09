# Distributed MPI — Phase 0 design (distributed solve + lifecycle + gather)

Date: 2026-06-04
Status: approved (design); implementation pending
Program: see [[project_distributed_mpi_roadmap]] — this is Phase 0 of 6.

## Goal

Make `backend=:slepc` run a **distributed** SLEPc eigensolve over `MPI.COMM_WORLD`
(any rank count), with **replicated** Julia assembly: every rank holds the full
`A`/`B` (as today) but inserts only its owned PETSc rows; MUMPS factorizes the
shifted operator in parallel; eigenvectors are gathered to rank 0. Delivers the
parallel-eigensolve speedup that motivates the whole program; later phases (1–5)
distribute the assembly itself.

## Constraints (carried from feasibility probe + decisions)

- Installed PetscWrap 0.1.5 / SlepcWrap 0.1.3 expose `MatCreate(comm)`,
  `MatGetOwnershipRange`, `MatMPIAIJSetPreallocation`, `MatSetValues`,
  `VecGet{OwnershipRange,Size,LocalSize,Array}`, `EPSCreate(comm)`.
- **No `VecScatter` wrapper** → the rank-0 gather is hand-written `ccall`s into
  `libpetsc` (`VecScatterCreateToZero`/`Begin`/`End`/`Destroy`).
- Parallel direct solver = **MUMPS** (configured via PETSc option string).
- **Lifecycle: explicit** `slepc_init!()` / `slepc_finalize!()` (no per-solve init).
- **Gather target: rank 0 only.**
- **Untestable in this environment** (no PETSc/MPI). Only serial-pure logic, stub
  errors, unchanged `:krylovkit` default, and parse/load are verifiable here.

## Approach

One `COMM_WORLD` distributed code path that subsumes the serial case (rank count 1 ⇒
rank 0 owns all rows). It **replaces** the current serial `_slepc_solve` body. No
separate serial path (rejected: two bodies drift).

## Architecture

### Core (`src/Stability/solver.jl`, PETSc-free)

Three registry hooks the extension fills on load (mirrors the existing
`_SLEPC_SOLVER`):
```julia
const _SLEPC_SOLVER   = Ref{Union{Nothing,Function}}(nothing)  # already exists
const _SLEPC_INIT     = Ref{Union{Nothing,Function}}(nothing)
const _SLEPC_FINALIZE = Ref{Union{Nothing,Function}}(nothing)
```
Exported user-facing functions (error actionably if the hook is `nothing`):
- `slepc_init!(opts::AbstractString="")` → forwards to `_SLEPC_INIT[]`.
- `slepc_finalize!()` → forwards to `_SLEPC_FINALIZE[]`.

`_solve_generalized_eigen_slepc` (exists) keeps forwarding to `_SLEPC_SOLVER[]`;
its error message gains "…and call `Magrathea.slepc_init!()` once before solving."

**Serial-testable pure helper (lives in core, no PETSc):**
```julia
# Diagonal/off-diagonal nnz counts for the owned row block [rstart, rend) (0-based,
# half-open, PETSc convention) of a replicated SparseMatrixCSC, split at the same
# [rstart,rend) column band that PETSc owns on this rank. Returns (d_nnz, o_nnz)
# as Vector{Int} of length (rend-rstart).
function _petsc_owned_nnz(M::SparseMatrixCSC, rstart::Int, rend::Int) -> (Vector{Int}, Vector{Int})
```
This is the fiddly preallocation math, isolated and unit-testable in serial.

### Extension (`ext/MagratheaSlepcExt/`)

- `MagratheaSlepcExt.jl` — `__init__` registers all three hooks (`_SLEPC_SOLVER`,
  `_SLEPC_INIT`, `_SLEPC_FINALIZE`). Holds an `Ref{Bool}` `_INITIALIZED` guard.
  - `_slepc_init!(opts)` → if not initialized, `SlepcInitialize(opts)` (collective),
    set guard. Idempotent.
  - `_slepc_finalize!()` → if initialized, `SlepcFinalize()`, clear guard.
  - `_slepc_solve(A,B; nev,sigma,which,selection,tol,maxiter,verbosity)` — the
    distributed solve (below). Errors if `!_INITIALIZED[]`.
- `raw_petsc.jl` — hand-written `ccall` bindings for what SlepcWrap 0.1.3 lacks:
  `_vec_scatter_to_zero(vr) -> Vector{ComplexF64}` (returns the full vector on rank 0,
  empty on workers, encapsulating `VecScatterCreateToZero`/`Begin`/`End`/
  `VecGetArray`/`VecScatterDestroy`/`VecDestroy`), and `_eps_set_dimensions(eps, nev)`
  (`EPSSetDimensions`). Keeps raw C out of the solver body.

## Data flow (per `_slepc_solve`, collective on all ranks)

1. Guard: `_INITIALIZED[] || error("call Magrathea.slepc_init!() first")`. Complex-build
   guard (`PetscScalar <: Real` → error).
2. `n = size(A,1)`. `target` from `sigma`/`which` (same rule as serial).
3. Build distributed `Amat`, `Bmat`:
   `MatCreate(COMM_WORLD)`; `MatSetSizes(PETSC_DECIDE,PETSC_DECIDE,n,n)`;
   `MatSetType` MPIAIJ / `MatSetFromOptions`; `rstart,rend = MatGetOwnershipRange`;
   `d,o = _petsc_owned_nnz(M, rstart, rend)`; `MatMPIAIJSetPreallocation(0,d,0,o)`;
   insert owned rows only (`for grow in rstart:rend-1`, walk that Julia row, one
   `MatSetValues`/`MatSetValue` per entry, 0-based); `MatAssemblyBegin/End`.
4. `eps = EPSCreate(COMM_WORLD)`; `EPSSetOperators`; `EPSSetTarget(target)`
   (wrapped); set `nev` per-solve via a **raw `ccall` to `EPSSetDimensions`**
   (`EPSSetDimensions(eps, nev, PETSC_DECIDE, PETSC_DECIDE)`) — SlepcWrap 0.1.3 does
   not wrap it and the global option string cannot carry a per-solve `nev`;
   `EPSSetWhichEigenpairs(EPS_TARGET_MAGNITUDE)`; `EPSSetFromOptions`; `EPSSetUp`;
   `EPSSolve`. (Problem-type + spectral-transform + solver — invariant across solves
   — come from the option string set once in `slepc_init!`:
   `-eps_gen_non_hermitian -st_type sinvert -st_pc_type lu
   -st_pc_factor_mat_solver_type mumps -eps_target_magnitude`. `nev` and `target` are
   the only per-solve settings, applied via the ccall + `EPSSetTarget`.)
5. `nconv = EPSGetConverged`; `nout = min(nconv, nev)`; `nout==0` → cleanup + error.
6. Eigenvalues: `EPSGetEigenvalue(eps, j)` → `vals[j+1]` (collective, identical all
   ranks). Eigenvectors: `EPSGetEigenpair` into a work Vec `vr`; `full =
   _vec_scatter_to_zero(vr)`; on rank 0 copy `full` into `vecs[:, j+1]`.
7. Destroy `eps`, `Amat`, `Bmat` (NOT SlepcFinalize — lifecycle is explicit).
8. Sort by `selection` (same `_sort_indices_local`). Return:
   - rank 0: `(vals[perm], vecs[:,perm], info)` with full `vecs`.
   - workers: `(vals[perm], Matrix{ComplexF64}(undef,n,0), info)` (empty eigenvectors).

## Magrathea contract on workers

`_dispatch_eigen` and the downstream reconstruction (linear.jl / galerkin column
loops, `_eigvecs_to_matrix`) must tolerate an `n×0` (or `0`-column) eigenvector
result without error: worker `StabilityResult.eigenvectors` ends up empty;
eigenvalues are valid everywhere. Driver reads eigenvectors on rank 0
(`MPI.Comm_rank(MPI.COMM_WORLD)==0`). Audit each site for `size(...,2)==0` safety.

## Error handling

- `slepc_init!` not called before a solve → actionable error.
- Extension not loaded → existing `_solve_generalized_eigen_slepc` error (names
  PetscWrap/SlepcWrap), extended to mention `slepc_init!`.
- Real-scalar PETSc → error pointing at `--with-scalar-type=complex`.
- MUMPS missing → surface PETSc's `MatSolverType mumps` error (don't swallow).
- `nconv==0` → error after destroying PETSc objects.
- Double `slepc_init!` → no-op (guarded). `slepc_finalize!` when not init → no-op.

## Testing

**Runs here (no PETSc):**
- `_petsc_owned_nnz` unit tests: hand-built small CSC + several `(rstart,rend)`
  bands (full range, split range, empty range) → assert exact `d_nnz`/`o_nnz`.
  (Pure function — the core correctness of preallocation, tested without MPI.)
- `backend=:slepc` without extension → existing error; with extension absent,
  `slepc_init!()`/`slepc_finalize!()` error actionably.
- `:krylovkit` default unchanged (existing suite + smoke asserts).
- `Meta.parseall` of the extension + `vecscatter.jl`; `using Magrathea` → `CORE_OK`.

**Cluster-validate (user, on a complex-scalar PETSc+MUMPS build):**
- `mpirun -n {1,2,4} julia driver.jl`: `slepc_init!(); solve(prob; backend=:slepc); slepc_finalize!()`.
- Leading eigenvalues match `:krylovkit` within tol on all ranks; rank-0
  eigenvectors correct; workers' empty eigenvectors don't crash the driver.
- A guarded test (skipped unless `PETSC_DIR` set and run under MPI) asserting the above.

## Files

- `src/Stability/solver.jl` — add `_SLEPC_INIT`/`_SLEPC_FINALIZE` Refs,
  `slepc_init!`/`slepc_finalize!`, `_petsc_owned_nnz`; extend the slepc error text.
- `src/Magrathea.jl` — export `slepc_init!`, `slepc_finalize!`.
- `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl` — distributed `_slepc_solve`, init/finalize,
  register 3 hooks, `_INITIALIZED` guard.
- `ext/MagratheaSlepcExt/raw_petsc.jl` — hand-written `ccall`s for primitives SlepcWrap
  0.1.3 doesn't wrap: `VecScatterCreateToZero` gather helper and `EPSSetDimensions`.
- `test/slepc_backend.jl` — `_petsc_owned_nnz` unit tests; init/finalize stub-error
  tests; the MPI equivalence test (guarded/skip).

## Scope / out of scope

- IN: distributed solve over COMM_WORLD, explicit lifecycle, rank-0 gather, MUMPS
  shift-invert, replicated assembly.
- OUT (later phases): distributed assembly (each rank builds only its rows),
  distributed reductions, gather-to-all, non-MUMPS solvers, SuperLU_DIST.

## Verification limitation

No PETSc/MPI here. Only `_petsc_owned_nnz`, stub/error behavior, unchanged
`:krylovkit`, and parse/load are verified locally. Distributed correctness
(ownership inserts, MUMPS factorization, VecScatter gather, worker empty-result
handling) must be confirmed by the user under `mpirun` on a PETSc+MUMPS build.
