# SLEPc/PETSc eigensolver backend — design

Date: 2026-06-03
Status: approved (design); implementation pending

## Goal

Add SLEPc (via `SlepcWrap.jl` + `PetscWrap.jl`) as an alternative backend for the
generalized eigenvalue problem `A x = σ B x` solved throughout Magrathea.jl, selectable
per solve. KrylovKit remains the default and is unchanged.

## Constraints discovered (feasibility probe)

- `PetscWrap.jl` and `SlepcWrap.jl` resolve from the General registry.
- `PetscWrap` does **not** bundle PETSc. It `dlopen`s a **system** `libpetsc.so`
  located via `PETSC_DIR`/`PETSC_ARCH`; `SlepcWrap` likewise needs `SLEPC_DIR`.
  Both pull `MPI.jl`.
- This dev/CI machine has **no PETSc build**, so the extension cannot be loaded or
  runtime-tested here. The SLEPc solve will be exercised only on a machine with a
  PETSc/SLEPc build.
- Therefore PETSc/SLEPc/MPI **cannot** be hard dependencies (would break
  `using Magrathea` for every user without PETSc). They must be optional **weak
  dependencies** behind a package extension — the same mechanism already used for
  `Makie` (`MagratheaMakieExt`) and `RecipesBase` (`MagratheaRecipesBaseExt`).
- PETSc must be built with **complex scalars** (`--with-scalar-type=complex`):
  Magrathea.jl's `A`, `B` pencils are `Complex`.

## Chosen approach: additive backend hook + weak-dependency extension

Rejected alternatives:
- *Unified `_generalized_eigen` refactor of every path* — cleaner long-term but
  rewrites the well-tested KrylovKit paths, risking regression on working code.
- *Separate `solve_slepc` entry point* — duplicates each path's matrix/constraint
  assembly logic.

The additive approach keeps every existing KrylovKit code path byte-for-byte
unchanged and adds a `backend` switch that diverts the **same** `(A, B)` pencil to
the extension only when `backend = :slepc`.

## Architecture

### Core (always loaded, no PETSc needed)

- **Hook + dispatcher in `src/Stability/solver.jl`:**
  - `const _SLEPC_SOLVER = Ref{Union{Nothing,Function}}(nothing)` — a registry slot the
    extension populates with its solver callable in its `__init__`. Core never
    references PETSc/SLEPc symbols, so the core load path stays PETSc-free and there is
    no method redefinition or type piracy.
  - `_solve_generalized_eigen_slepc(A, B; nev, sigma, which, selection, tol, maxiter,
    kwargs...)` — core function that checks `_SLEPC_SOLVER[]`: if `nothing`, **throws an
    informative error** ("SLEPc backend requires `PetscWrap` and `SlepcWrap` to be
    loaded, plus a complex-scalar PETSc build with `PETSC_DIR`/`SLEPC_DIR` set");
    otherwise forwards to the registered callable.
  - `_dispatch_eigen(A, B; backend::Symbol, nev, sigma, which, selection, tol,
    maxiter, krylovdim, verbosity)` returns the existing
    `(eigenvalues::Vector{Complex}, eigenvectors::Matrix{Complex}, info)` contract:
    - `backend == :krylovkit` (default) → existing `_krylov_eigensolve`.
    - `backend == :slepc` → `_solve_generalized_eigen_slepc(...)`.
    - else → `ArgumentError` listing valid values.

### Integration sites (each gains a `backend` kwarg, default `:krylovkit`)

1. `solver.jl: solve_eigenvalue_problem(A, B; …)` — used by the MHD tau path.
   Replace the direct `_krylov_eigensolve` call with `_dispatch_eigen`.
2. `linear.jl` constrained hydro solve — after the constrained reduction produces the
   reduced pencil `(A_r, B_r)`, route it through `_dispatch_eigen`. The bespoke
   KrylovKit shift-invert remains the `:krylovkit` branch (unchanged); `:slepc`
   diverts the reduced pencil. Reduced eigenvectors come back in the same basis, so
   the existing full-eigenvector reconstruction is unchanged.
3. `triglobal.jl` — after the coupled-mode pencil `(A, B)` is assembled, route through
   `_dispatch_eigen` before the shift-invert `eigsolve`.
4. Dense onset path (`linear.jl`/`onset.jl`) — when `backend == :slepc`, sparsify the
   dense `(A, B)` (`sparse(A)`, `sparse(B)`) and call `_dispatch_eigen`; otherwise the
   existing `eigen`-based dense solve.

5. `solve(::OnsetProblem/::BiglobalProblem/::TriglobalProblem/::MHDProblem; backend=:krylovkit, …)`
   in `src/solve.jl` — thread `backend` down to the relevant solver call. Default
   keeps current behavior.

### Extension `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl`

Triggered by weakdeps `[PetscWrap, SlepcWrap]` (and `MPI`). On load, registers its
solver with the core hook. Implements `solve_generalized_eigen_slepc(A, B; nev,
sigma, which, selection, tol, maxiter, verbosity)`:

1. Ensure MPI initialized (guarded; serial — single rank, `MPI.COMM_SELF` or world
   size 1). Distributed matrix assembly is **out of scope**.
2. Build PETSc `Mat` (AIJ/sparse) for `A` and for the shifted operator `A - σB`
   from `SparseMatrixCSC` (CSC → CSR/COO column-walk; preallocate nnz per row).
3. Create SLEPc `EPS`: problem type `EPS_GNHEP` (generalized non-Hermitian); set
   operators `(A, B)`; spectral transform `ST` = shift-invert (`STSINVERT`) with
   `target = sigma` (use the same shift-selection rule as the KrylovKit path when
   `sigma === nothing`: `:LR`→10, `:LI`→10i, else 1); `EPS_TARGET_MAGNITUDE`; request
   `nev` (with a modest `ncv`/`mpd` heuristic).
4. Solve; read convergence reason and iteration count.
5. Extract eigenpairs into Julia `Vector{Complex{T}}` / `Matrix{Complex{T}}`; apply
   the same `_sort_indices(..., selection)` ordering used by KrylovKit.
6. Destroy PETSc objects. Return `(eigenvalues, eigenvectors, info)` with
   `info["solver"] = :slepc` plus `target`, `nconv`, `its`, `eps_type`,
   `converged_reason`.

`T` is derived from `eltype(A)` exactly as in `_krylov_eigensolve`.

## Data flow

`solve(problem; backend=:slepc)` → path-specific assembly produces `(A,B)` (and any
constraint reduction) → `_dispatch_eigen(A,B; backend=:slepc, …)` →
`solve_generalized_eigen_slepc` (extension) → `(values, vectors, info)` → existing
per-path eigenvector reconstruction + `StabilityResult`. Identical wrapping to the
KrylovKit path; only the inner solve differs.

## Error handling

- Backend `:slepc` selected, extension not loaded → actionable `ErrorException`
  naming the required packages and env vars. (Testable without PETSc.)
- PETSc built with real scalars → detect at first complex-matrix transfer and raise a
  clear error pointing at `--with-scalar-type=complex`.
- SLEPc non-convergence (`nconv < nev`) → warn and return whatever converged, with
  `converged_reason` in `info` (mirrors KrylovKit's behavior of returning available
  modes).
- Unknown `backend` symbol → `ArgumentError` listing valid values.

## Testing strategy

- **Runs here (no PETSc):**
  - `solve_eigenvalue_problem(A,B; backend=:slepc)` and `solve(problem; backend=:slepc)`
    throw the documented "load PetscWrap/SlepcWrap" error.
  - `backend=:krylovkit` (default) results identical to current — covered by the
    existing suite; add explicit `backend=:krylovkit` smoke asserts.
  - Unknown `backend` → `ArgumentError`.
- **Guarded (runs only where `PETSC_DIR` is set — the user's build, not here):**
  - SLEPc vs KrylovKit eigenvalue agreement on a small MHD/onset problem within
    tolerance (sorted leading eigenvalues match). Wrapped in
    `if haskey(ENV, "PETSC_DIR") … end` and noted as skipped otherwise (no silent
    pass).

## Packaging

- `Project.toml`:
  - `[weakdeps]`: `PetscWrap`, `SlepcWrap`, `MPI`.
  - `[extensions]`: `MagratheaSlepcExt = ["PetscWrap", "SlepcWrap"]` (MPI is a transitive
    dep of both; include explicitly if the extension uses it directly).
  - `[compat]`: pin PetscWrap/SlepcWrap/MPI to the resolved versions.
- No change to `[deps]` — core load path stays PETSc-free.

## Out of scope (explicit)

- Distributed (multi-rank) matrix assembly / parallel eigensolve.
- PETSc_jll/SLEPc_jll artifact wiring (user supplies a system build).
- Replacing KrylovKit as the default.
- Real-scalar PETSc support.

## Verification limitation (must communicate)

The actual SLEPc solve **cannot be run in this environment** (no PETSc build). The
implementation will be verified here only for: correct extension stub/error behavior,
unchanged `:krylovkit` defaults, and successful precompile of core. End-to-end SLEPc
correctness must be confirmed by the user on a PETSc/SLEPc-enabled machine using the
guarded equivalence test.
