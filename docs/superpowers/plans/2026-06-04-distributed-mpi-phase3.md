# Distributed MPI — Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distribute the constrained (tau-elimination) reduction for onset/biglobal by expressing it as `A_red = S·A·P` and computing it with distributed PETSc `MatMatMult`.

**Architecture:** Core gains a PETSc-free `_constraint_projection_matrices(reduction, interior_dofs) -> (S, P)` (serial-verified: `S·A·P == _constrained_reduced_matrices`). Extension adds a raw `MatMatMult` ccall + `_reduce_dist` + a `_SLEPC_CONSTRAINED_SOLVER` flow that assembles full A/B (replicated), builds/distributes S,P, reduces distributed, solves, and reconstructs `P·reduced` on rank 0.

**Tech Stack:** Julia 1.12; PetscWrap/SlepcWrap/MPI/MUMPS (cluster); raw `MatMatMult` ccall.

**Spec:** `docs/superpowers/specs/2026-06-04-distributed-mpi-phase3-design.md`. Roadmap: `project_distributed_mpi_roadmap` memory.

---

## Environment / workflow notes
- Julia binary: `JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia`. Bash needs `dangerouslyDisableSandbox: true`. Julia 1.12.x.
- **T1 verifiable here** (pure Julia). **T2 cluster-only** (no PETSc) → parse + symbol-audit + `CORE_OK`; runtime is the user's `mpirun`. **T3** wiring + docs.
- **Commits:** no `git commit` without explicit permission — each commit step pauses for the user.

---

## File Structure
- `src/Stability/linear.jl` (modify) — add `_constraint_projection_matrices`.
- `src/Stability/solver.jl` (modify) — `_SLEPC_CONSTRAINED_SOLVER` Ref + `_solve_constrained_slepc(op;…)` hook.
- `src/solve.jl` (modify) — route `solve(::OnsetProblem/::BiglobalProblem; backend=:slepc)` to the hook.
- `ext/MagratheaSlepcExt/raw_petsc.jl` (modify) — `_mat_mat_mult`.
- `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl` (modify) — `_reduce_dist`, `_solve_constrained_slepc`, register hook.
- `test/distributed_reduction.jl` (create) — serial tests; wired into `runtests.jl`.

---

## Task 1: `_constraint_projection_matrices` + serial equivalence (verifiable here)

**Files:** Modify `src/Stability/linear.jl`; create `test/distributed_reduction.jl`.

- [ ] **Step 1: Create `test/distributed_reduction.jl` with the failing equivalence test**

```julia
using Test
using LinearAlgebra
using SparseArrays
using Magrathea

@testset "S·A·P reproduces the constrained reduction" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
    op = LinearStabilityOperator(params)
    A_full, B_full, idofs, bdofs = assemble_matrices(op)
    A_red, B_red, reduction = Magrathea._constrained_reduced_matrices(A_full, B_full, op, idofs, bdofs)

    S, P = Magrathea._constraint_projection_matrices(reduction, idofs)

    @test size(P) == (reduction.n_full, reduction.n_reduced)
    @test size(S) == (reduction.n_reduced, reduction.n_full)
    @test nnz(S) == reduction.n_reduced                 # exactly one 1 per row
    @test Matrix(S * A_full * P) ≈ A_red rtol=1e-10
    @test Matrix(S * B_full * P) ≈ B_red rtol=1e-10
end
```

- [ ] **Step 2: Run, verify FAIL** — `$JL --project=. -e 'using Test; @testset "t" begin include("test/distributed_reduction.jl") end'` → FAIL (`_constraint_projection_matrices` undefined).

- [ ] **Step 3: Add `_constraint_projection_matrices` to `src/Stability/linear.jl`** (after `_reconstruct_full_vector`)

```julia
"""
    _constraint_projection_matrices(reduction, interior_dofs) -> (S, P)

Express the block-wise constraint reduction as two sparse matrices: `P`
(`n_full × n_reduced`, block-diagonal nullspace basis) and `S` (`n_reduced × n_full`,
interior-row selector), such that `S * A_full * P == _constrained_reduced_matrices(...)`.
"""
function _constraint_projection_matrices(reduction::ConstraintReduction{T},
                                         interior_dofs::Vector{Int}) where {T<:Real}
    length(interior_dofs) == reduction.n_reduced ||
        error("interior_dofs length $(length(interior_dofs)) != n_reduced $(reduction.n_reduced)")

    # P: block-diagonal basis (full_indices × reduced_indices)
    Pi = Int[]; Pj = Int[]; Pv = Complex{T}[]
    for block in reduction.blocks
        fr = block.full_indices
        rc = block.reduced_indices
        @inbounds for (cj, c) in enumerate(rc), (ri, r) in enumerate(fr)
            v = block.basis[ri, cj]
            push!(Pi, r); push!(Pj, c); push!(Pv, v)
        end
    end
    P = sparse(Pi, Pj, Pv, reduction.n_full, reduction.n_reduced)

    # S: row selector S[i, interior_dofs[i]] = 1
    S = sparse(collect(1:reduction.n_reduced), interior_dofs,
               ones(Complex{T}, reduction.n_reduced),
               reduction.n_reduced, reduction.n_full)

    return S, P
end
```

- [ ] **Step 4: Run, verify PASS** — same command as Step 2 → PASS.

- [ ] **Step 5: Regression — package loads + onset path unaffected** — `$JL --project=. -e 'using Test; include("test/onset_api.jl")' 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done`.

- [ ] **Step 6: Commit (ASK USER FIRST)** — `git add src/Stability/linear.jl test/distributed_reduction.jl` / `git commit -m "feat(mpi): S·A·P matrix form of the constrained reduction"`

---

## Task 2: Extension — distributed `MatMatMult` reduce + onset/biglobal rewire (NOT runnable here)

**Files:** Modify `ext/MagratheaSlepcExt/raw_petsc.jl`, `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl`, `src/Stability/solver.jl`, `src/solve.jl`.

> Cannot run here. Verify: `Meta.parseall`, symbol-audit vs installed PetscWrap 0.1.5, `CORE_OK`, full default suite green.

- [ ] **Step 1: Core hook in `src/Stability/solver.jl`** (next to `_SLEPC_MHD_SOLVER`)
```julia
const _SLEPC_CONSTRAINED_SOLVER = Ref{Union{Nothing,Function}}(nothing)

function _solve_constrained_slepc(op; kwargs...)
    f = _SLEPC_CONSTRAINED_SOLVER[]
    f === nothing && error("backend=:slepc (distributed constrained reduction) requires `using PetscWrap, SlepcWrap` and Magrathea.slepc_init!().")
    return f(op; kwargs...)
end
```

- [ ] **Step 2: `_mat_mat_mult` in `ext/MagratheaSlepcExt/raw_petsc.jl`**
```julia
const MAT_INITIAL_MATRIX = Cint(0)   # PETSc MatReuse

"""Distributed `C = A*B` via PETSc `MatMatMult` (unwrapped in PetscWrap 0.1.5).
`MatMatMult(A, B, MatReuse scall, PetscReal fill, Mat *C)`; `fill=PETSC_DEFAULT (-2.0)`."""
function _mat_mat_mult(A::PetscWrap.PetscMat, B::PetscWrap.PetscMat)
    C = PetscWrap.PetscMat(A.comm)
    PR = PetscWrap.PetscReal
    @assert iszero(ccall((:MatMatMult, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
        (PetscWrap.CMat, PetscWrap.CMat, Cint, PR, Ptr{PetscWrap.CMat}),
        A, B, MAT_INITIAL_MATRIX, PR(-2.0), C.ptr))
    return C
end
```
AUDIT: confirm `PetscWrap.PetscReal` exists and `PetscMat`'s constructor/field is `.ptr::Ref{CMat}` with a `.comm` (it does — mirrors `PetscVec`). Confirm `MatReuse`/`PETSC_DEFAULT` values against the installed PETSc headers.

- [ ] **Step 3: `_reduce_dist` + `_solve_constrained_slepc` in `MagratheaSlepcExt.jl`**
```julia
"""Distributed S·A·P reduction (destroys the SA intermediate)."""
function _reduce_dist(Amat, Smat, Pmat)
    SA = _mat_mat_mult(Smat, Amat)
    red = _mat_mat_mult(SA, Pmat)
    MatDestroy(SA)
    return red
end

function _slepc_constrained_solve(op; nev::Int, sigma, which::Symbol, tol::Float64, maxiter::Int)
    _INITIALIZED[] || error("call Magrathea.slepc_init!() once before a :slepc solve")
    PetscScalar <: Real && error("PETSc/SLEPc must be built with complex scalars")
    # 1. replicated full assembly + reduction (as in the serial path)
    A_full, B_full, idofs, bdofs = Magrathea.assemble_matrices(op)
    A_red_ref, B_red_ref, reduction =
        Magrathea._constrained_reduced_matrices(A_full, B_full, op, idofs, bdofs)  # for `reduction`/P
    S, P = Magrathea._constraint_projection_matrices(reduction, idofs)
    nred = reduction.n_reduced
    nfull = reduction.n_full
    # 2. distribute full A, B, S, P (reuse Phase-0 replicated insert)
    Adist = _to_petsc_dist(sparse(A_full), nfull)
    Bdist = _to_petsc_dist(sparse(B_full), nfull)
    Sdist = _to_petsc_dist(S, nfull)        # S is nred×nfull — see note
    Pdist = _to_petsc_dist(P, nfull)        # P is nfull×nred — see note
    # 3. distributed reduce
    Ared = _reduce_dist(Adist, Sdist, Pdist)
    Bred = _reduce_dist(Bdist, Sdist, Pdist)
    MatDestroy(Adist); MatDestroy(Bdist); MatDestroy(Sdist); MatDestroy(Pdist)
    # 4. solve reduced pencil (nred×nred); gather reduced eigenvectors to rank 0
    vals, vecs_red, info = _eps_solve_and_gather(Ared, Bred, nred;
        nev=nev, sigma=sigma, which=which, selection=:maxreal, tol=tol, maxiter=maxiter)
    # 5. reconstruct full eigenvectors on rank 0 via P (replicated Julia sparse)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    vecs_full = (rank == 0 && size(vecs_red, 2) > 0) ?
        Matrix{ComplexF64}(P * vecs_red) : Matrix{ComplexF64}(undef, nfull, 0)
    return vals, vecs_full, info
end
```
NOTE on `_to_petsc_dist` for non-square `S`/`P`: Phase-0 `_to_petsc_dist(M, n)` assumes `n×n`. `S` is `nred×nfull` and `P` is `nfull×nred` (rectangular). Generalize `_to_petsc_dist` to `_to_petsc_dist(M, nrows, ncols)` (set sizes `(PETSC_DECIDE, PETSC_DECIDE, nrows, ncols)`, preallocate via the rectangular analog of `_petsc_owned_nnz` — the diagonal-band split still uses the row range; for rectangular MPIAIJ PETSc's `MatMPIAIJSetPreallocation` still takes per-owned-row d/o counts where the "diagonal block" columns are this rank's column ownership). For simplicity and since these matrices are built once, you MAY instead preallocate generously (`MatMPIAIJSetPreallocation(mat, PI(maxnnz), C_NULL, PI(maxnnz), C_NULL)`) or set `MatSetOption(MAT_NEW_NONZERO_ALLOCATION_ERR, false)` and skip exact preallocation. Pick the simplest correct option and report it; exact rectangular preallocation is a perf detail, not correctness.

- [ ] **Step 4: register hook** — in `__init__`, add `Magrathea._SLEPC_CONSTRAINED_SOLVER[] = _slepc_constrained_solve`.

- [ ] **Step 5: rewire `solve(::OnsetProblem/::BiglobalProblem; backend=:slepc)`** in `src/solve.jl`. Read how each currently reaches the solver (Onset → `solve_onset_problem` → `solve_eigenvalue_problem(op; backend=:slepc)`; Biglobal → `solve_biglobal_problem`). When `backend === :slepc`, call `eigenvalues, eigenvectors, info = Magrathea._solve_constrained_slepc(op; nev, sigma, which, tol, maxiter)` and use them in the `StabilityResult` wrapping (operator=op, info=info). Keep `:krylovkit` untouched. Determine the cleanest interception point (likely in `solve_onset_problem`/`solve_biglobal_problem`, where `op` is available) and report exactly what you changed.

- [ ] **Step 6: Parse + symbol-audit + CORE_OK + full suite**
  - `$JL -e 'for f in ("ext/MagratheaSlepcExt/raw_petsc.jl","ext/MagratheaSlepcExt/MagratheaSlepcExt.jl"); Meta.parseall(read(f,String)); end; println("PARSE_OK")'` → PARSE_OK.
  - audit `MatMatMult`, `PetscReal`, `MatDestroy`, the rectangular `_to_petsc_dist` against installed source.
  - `$JL --project=. -e 'using Magrathea; println("CORE_OK ", Magrathea._SLEPC_CONSTRAINED_SOLVER[]===nothing)'` (sandbox off) → `CORE_OK true`.
  - `$JL --project=. test/runtests.jl 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done` (the `:krylovkit` onset/biglobal default must be unaffected).

- [ ] **Step 7: Commit (ASK USER FIRST)** — `git add ext/MagratheaSlepcExt/ src/Stability/solver.jl src/solve.jl` / `git commit -m "feat(mpi): distributed S·A·P constrained reduction via MatMatMult"`

---

## Task 3: Wire tests + full suite + docs

**Files:** Modify `test/runtests.jl`, `README.md`, `test/distributed_reduction.jl`.

- [ ] **Step 1: Wire** — add `include("distributed_reduction.jl")` to `test/runtests.jl` (near the other distributed includes).

- [ ] **Step 2: Guarded cluster note-test** appended to `test/distributed_reduction.jl`:
```julia
@testset "Distributed constrained reduction (requires PETSc+MUMPS under MPI)" begin
    if !haskey(ENV, "PETSC_DIR")
        @info "PETSC_DIR unset — skipping distributed constrained reduction test (validate under mpirun; see README)"
        @test true
    else
        @info "Run the mpirun onset/biglobal :slepc-vs-:krylovkit spectrum comparison manually; see README."
        @test true
    end
end
```

- [ ] **Step 3: Full suite green** — `$JL --project=. test/runtests.jl 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done`; confirm `S·A·P reproduces the constrained reduction` appears in the log.

- [ ] **Step 4: README** — note onset/biglobal `:slepc` now distribute the reduction (`S·A·P` via PETSc `MatMatMult`); full-matrix assembly still replicated (Phase 4). Same `mpirun` driver.

- [ ] **Step 5: Commit (ASK USER FIRST)** — `git add test/runtests.jl test/distributed_reduction.jl README.md` / `git commit -m "test+docs: distributed constrained reduction"`

---

## Self-review notes
- **Spec coverage:** `_constraint_projection_matrices` + `S·A·P` equivalence (T1); `_mat_mat_mult`/`_reduce_dist` (T2); onset/biglobal `:slepc` rewire via `_SLEPC_CONSTRAINED_SOLVER` (T2); rank-0 `P·reduced` reconstruction (T2 Step 3); suite wiring + guarded cluster test + docs (T3). All spec sections mapped.
- **Placeholder note:** the rectangular `_to_petsc_dist` preallocation is given two correct options (exact rectangular d/o, or generous prealloc + `MAT_NEW_NONZERO_ALLOCATION_ERR=false`) — the implementer picks the simpler; this is a perf detail, not correctness, and is cluster-tuned.
- **Type consistency:** `_constraint_projection_matrices(reduction, interior_dofs) -> (S::SparseMatrixCSC, P::SparseMatrixCSC)`; `S` is `n_reduced×n_full`, `P` is `n_full×n_reduced`; `_reduce_dist(A,S,P)=S·A·P`; reconstruction `P*reduced`. Consistent with the spec and the serial reduction. The test reuses the `reduction` returned by `_constrained_reduced_matrices` (nullspace non-uniqueness).
- **Verification honesty:** T1 + the `S·A·P` equivalence run here; T2's `MatMatMult`/distributed reduce/onset-biglobal solve are cluster-only (the most cluster-dependent phase), validated by the `mpirun` eigenvalue-match run.
