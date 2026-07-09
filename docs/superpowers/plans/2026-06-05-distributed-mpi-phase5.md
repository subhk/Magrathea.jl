# Distributed MPI — Phase 5 Implementation Plan (final)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distribute the triglobal coupled-pencil assembly (each rank inserts only its owned rows of the `n_total×n_total` pencil), feeding the Phase-0 distributed solve. `single_mode_ops`/`coupling_ops` stay replicated (per-m + coupling distribution deferred with the Vector-SH rewrite). Add a `build_single_mode_operator` seam.

**Architecture:** Owned filter on `_append_block_entries!`; `_assemble_block_coo(problem, single_mode_ops, coupling_ops; owned_julia_rows)` (owned-row COO + per-block skip); `assemble_block_matrices` thin wrapper (byte-identical). Extension `_slepc_triglobal_solve` builds replicated ops, then distributed owned-row pencil assembly + Phase-0 solve.

**Tech Stack:** Julia 1.12; PetscWrap/SlepcWrap/MPI/MUMPS (cluster).

**Spec:** `docs/superpowers/specs/2026-06-05-distributed-mpi-phase5-design.md`. Roadmap: `project_distributed_mpi_roadmap` memory.

---

## Environment / workflow notes
- Julia binary: `JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia`. Bash needs `dangerouslyDisableSandbox: true`. Julia 1.12.x.
- **T1, T2 verifiable here.** **T3 cluster-only.** **T4** wiring/docs.
- **Commits:** no `git commit` without explicit permission — each commit step pauses for the user.
- `assemble_block_matrices(problem, single_mode_ops, coupling_ops, verbose)` and `build_single_mode_operators(problem, verbose)` take a trailing `verbose` arg.

---

## Task 1: owned filter + `_assemble_block_coo` + wrapper

**Files:** Modify `src/Stability/triglobal.jl`; create `test/distributed_triglobal.jl`.

- [ ] **Step 1: failing partition test** — `test/distributed_triglobal.jl`. First READ `test/triglobal.jl` lines ~28–54 for the `_basic_state_3d_with_modes` helper + the coupling fixture (uphi `(1,1)` etc., `m_range=1:2`, lmax 4, Nr 12). Build the same `problem` via `setup_coupled_mode_problem` + `build_single_mode_operators(problem, false)` + `build_mode_coupling_operators(problem, single, false)`:
```julia
using Test
using SparseArrays
using Magrathea

@testset "triglobal coupled-pencil COO partition-reassembles" begin
    # build a small coupled problem (mirror test/triglobal.jl fixture)
    <CONSTRUCT problem, single, coupling as in test/triglobal.jl ~line 333>
    full = Magrathea._assemble_block_coo(problem, single, coupling)
    n = full.n
    A_full = sparse(full.A_rows, full.A_cols, full.A_vals, n, n)
    B_full = sparse(full.B_rows, full.B_cols, full.B_vals, n, n)
    cuts = [0, fld(n,3), fld(2n,3), n]
    Ar=Int[]; Ac=Int[]; Av=ComplexF64[]; Br=Int[]; Bc=Int[]; Bv=ComplexF64[]
    for i in 1:3
        R = (cuts[i]+1):cuts[i+1]
        c = Magrathea._assemble_block_coo(problem, single, coupling; owned_julia_rows=R)
        @test all(r -> r in R, c.A_rows); @test all(r -> r in R, c.B_rows)
        append!(Ar,c.A_rows); append!(Ac,c.A_cols); append!(Av,c.A_vals)
        append!(Br,c.B_rows); append!(Bc,c.B_cols); append!(Bv,c.B_vals)
    end
    @test sparse(Ar,Ac,Av,n,n) == A_full
    @test sparse(Br,Bc,Bv,n,n) == B_full
end
```

- [ ] **Step 2: run, verify FAIL** (`_assemble_block_coo` undefined).

- [ ] **Step 3: add `owned` kwarg to `_append_block_entries!`** (src/Stability/triglobal.jl ~1872):
```julia
function _append_block_entries!(row_idx, col_idx, val_idx, block, rows, cols, tol;
                                owned::Union{Nothing,UnitRange{Int}}=nothing)
    for (local_i, global_i) in enumerate(rows)
        (owned === nothing || global_i in owned) || continue
        for (local_j, global_j) in enumerate(cols)
            val = block[local_i, local_j]
            if abs(val) > tol
                push!(row_idx, global_i); push!(col_idx, global_j); push!(val_idx, val)
            end
        end
    end
end
```

- [ ] **Step 4: write `_assemble_block_coo(problem, single_mode_ops, coupling_ops; owned_julia_rows=nothing)`** — copy `assemble_block_matrices`' COO body (the two loops), passing `owned=owned_julia_rows` into `_append_block_entries!`, and skipping a block whose target rows don't intersect `owned`:
```julia
function _assemble_block_coo(problem::CoupledModeProblem{T}, single_mode_ops::Dict,
                             coupling_ops::Dict; owned_julia_rows=nothing) where T
    n_total = problem.total_dofs
    row_A=Int[]; col_A=Int[]; val_A=Complex{T}[]
    row_B=Int[]; col_B=Int[]; val_B=Complex{T}[]
    tol = T(1e-14)
    owns(rng) = owned_julia_rows === nothing || !isempty(intersect(rng, owned_julia_rows))
    for m in problem.m_range
        br = problem.block_indices[m]
        owns(br) || continue
        _append_block_entries!(row_A,col_A,val_A, single_mode_ops[m].A, br, br, tol; owned=owned_julia_rows)
        _append_block_entries!(row_B,col_B,val_B, single_mode_ops[m].B, br, br, tol; owned=owned_julia_rows)
    end
    for ((m_from, m_to), C) in coupling_ops
        isempty(C) && continue
        rt = problem.block_indices[m_to]; rf = problem.block_indices[m_from]
        owns(rt) || continue
        _append_block_entries!(row_A,col_A,val_A, C, rt, rf, tol; owned=owned_julia_rows)
    end
    return (A_rows=row_A, A_cols=col_A, A_vals=val_A,
            B_rows=row_B, B_cols=col_B, B_vals=val_B, n=n_total)
end
```

- [ ] **Step 5: make `assemble_block_matrices` a thin wrapper**
```julia
function assemble_block_matrices(problem::CoupledModeProblem{T}, single_mode_ops::Dict,
                                  coupling_ops::Dict, verbose::Bool) where T
    c = _assemble_block_coo(problem, single_mode_ops, coupling_ops)
    A = sparse(c.A_rows, c.A_cols, c.A_vals, c.n, c.n)
    B = sparse(c.B_rows, c.B_cols, c.B_vals, c.n, c.n)
    if verbose
        println("  Matrix size: $(c.n) × $(c.n)")
        println("  nnz(A): $(nnz(A))  nnz(B): $(nnz(B))")
    end
    return A, B
end
```

- [ ] **Step 6: run partition test → PASS; triglobal regression** — `$JL --project=. -e 'using Test; @testset "t" begin include("test/distributed_triglobal.jl") end'` PASS; then `$JL --project=. -e 'using Test; include("test/triglobal.jl")' 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done`.

- [ ] **Step 7: Commit (ASK USER FIRST)** — `git add src/Stability/triglobal.jl test/distributed_triglobal.jl` / `git commit -m "feat(mpi): distributed triglobal coupled-pencil COO assembly"`

---

## Task 2: `build_single_mode_operator` seam

**Files:** Modify `src/Stability/triglobal.jl`; extend `test/distributed_triglobal.jl`.

- [ ] **Step 1: failing match test** — append:
```julia
@testset "build_single_mode_operator matches the all-builder" begin
    <CONSTRUCT the same `problem` as Task 1>
    all_ops = Magrathea.build_single_mode_operators(problem, false)
    for m in problem.m_range
        one = Magrathea.build_single_mode_operator(problem, m)
        @test one.A ≈ all_ops[m].A
        @test one.B ≈ all_ops[m].B
    end
end
```

- [ ] **Step 2: run, verify FAIL** (`build_single_mode_operator` undefined).

- [ ] **Step 3: extract the per-m body.** READ `build_single_mode_operators(problem, verbose)` (~line 394) fully. Create `build_single_mode_operator(problem::CoupledModeProblem{T}, m::Int)` containing exactly its loop body for one `m` (recompute `basic_state_axis = axisymmetric_basic_state(problem.params.basic_state_3d)` and `has_axisymmetric = _has_nonzero_basic_state(basic_state_axis)` at the top, then the params_m → op_m → `assemble_matrices` → `_constrained_reduced_matrices` → conj-if-`m<0` → construct/return the `SingleModeOperator`). Then refactor `build_single_mode_operators` to compute `basic_state_axis`/`has_axisymmetric` once and loop `single_mode_ops[m] = build_single_mode_operator(problem, m)` — OR keep its body and just ADD the singular function (less churn; the match test guards equivalence). Pick the lower-risk option and report; the singular builder MUST produce the identical `SingleModeOperator` as the loop (verified by the test).

- [ ] **Step 4: run match test → PASS; regression** — `$JL --project=. -e 'using Test; @testset "t" begin include("test/distributed_triglobal.jl") end'` PASS; `$JL --project=. -e 'using Test; include("test/triglobal.jl")' 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done`.

- [ ] **Step 5: Commit (ASK USER FIRST)** — `git add src/Stability/triglobal.jl test/distributed_triglobal.jl` / `git commit -m "feat(mpi): on-demand single-mode operator builder (per-m seam)"`

---

## Task 3: extension distributed triglobal solve + rewire (NOT runnable here)

**Files:** Modify `src/Stability/solver.jl`, `src/Stability/triglobal.jl`, `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl`.

> Cluster-only. Verify: `Meta.parseall`, `CORE_OK`, full default suite green.

- [ ] **Step 1: core hook** in `src/Stability/solver.jl`:
```julia
const _SLEPC_TRIGLOBAL_SOLVER = Ref{Union{Nothing,Function}}(nothing)

function _solve_triglobal_slepc(problem; kwargs...)
    f = _SLEPC_TRIGLOBAL_SOLVER[]
    f === nothing && error("backend=:slepc (distributed triglobal) requires `using PetscWrap, SlepcWrap` and Magrathea.slepc_init!().")
    return f(problem; kwargs...)
end
```

- [ ] **Step 2: extension `_slepc_triglobal_solve`** in `MagratheaSlepcExt.jl`:
```julia
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
    T = real(eltype(coo.A_vals))
    shift = ComplexF64(σ_target, 1e-6)
    vals, vecs, _ = _eps_solve_and_gather(Amat, Bmat, n;
        nev=nev, sigma=shift, which=:LR, selection=:maxreal, tol=tol, maxiter=maxiter)
    return vals, vecs        # 2-tuple, matching solve_block_eigenvalue_problem
end
```
Register in `__init__`: `Magrathea._SLEPC_TRIGLOBAL_SOLVER[] = _slepc_triglobal_solve`.

- [ ] **Step 3: rewire `solve_triglobal_eigenvalue_problem`** (`src/Stability/triglobal.jl` ~1998). READ it: it builds `single_mode_ops`, `coupling_ops`, calls `assemble_block_matrices` then `solve_block_eigenvalue_problem(A,B,σ_target,nev,verbose; backend=backend)`. When `backend === :slepc`, replace the assemble+solve with `eigenvalues, eigenvectors = Magrathea._solve_triglobal_slepc(problem; σ_target=σ_target, nev=nev, tol=T(1e-8), maxiter=200)` (the distributed path builds its own ops, so skip the replicated `assemble_block_matrices` + `solve_block_eigenvalue_problem`). Keep `:krylovkit` path EXACTLY as is. (Also leave the existing `:slepc` divert inside `solve_block_eigenvalue_problem` — now unreachable for triglobal but harmless; or remove it. Report which.)

- [ ] **Step 4: verify** — `PARSE_OK`; `$JL --project=. -e 'using Magrathea; println("CORE_OK ", Magrathea._SLEPC_TRIGLOBAL_SOLVER[]===nothing)'` → `CORE_OK true`; `$JL --project=. test/runtests.jl 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done` (the `:krylovkit` triglobal default unaffected).

- [ ] **Step 5: Commit (ASK USER FIRST)** — `git add src/Stability/solver.jl src/Stability/triglobal.jl ext/MagratheaSlepcExt/` / `git commit -m "feat(mpi): distributed triglobal :slepc solve"`

---

## Task 4: wire tests + full suite + docs

- [ ] **Step 1:** add `include("distributed_triglobal.jl")` to `test/runtests.jl`.
- [ ] **Step 2:** append a `PETSC_DIR`-guarded cluster note-test to `test/distributed_triglobal.jl` (mirror prior phases' skip pattern).
- [ ] **Step 3:** full suite green — `$JL --project=. test/runtests.jl 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done`; confirm the triglobal partition test appears.
- [ ] **Step 4:** README — triglobal `:slepc` now distributes the coupled-pencil assembly (per-m operators + coupling still replicated, pending the Vector-SH rewrite). With this, all four problem families have a distributed `:slepc` path.
- [ ] **Step 5: Commit (ASK USER FIRST)** — `git add test/runtests.jl test/distributed_triglobal.jl README.md` / `git commit -m "test+docs: distributed triglobal"`

---

## Self-review notes
- **Spec coverage:** owned filter on `_append_block_entries!` (T1), `_assemble_block_coo` + wrapper (T1), `build_single_mode_operator` seam (T2), `_SLEPC_TRIGLOBAL_SOLVER` hook + `_slepc_triglobal_solve` + rewire (T3), suite/docs (T4). All spec sections mapped. Coupling/per-m stay replicated per the decision.
- **Placeholder note:** T1/T2 Step 1 say "CONSTRUCT problem as in test/triglobal.jl" — bounded (copy the existing fixture); the implementer reads that file. Acceptable.
- **Type consistency:** `_assemble_block_coo(problem, single, coupling; owned_julia_rows)` → NamedTuple `(A_rows,…,n)`; `_append_block_entries!(…; owned)`; `build_single_mode_operator(problem,m)::SingleModeOperator`; `_slepc_triglobal_solve` returns the 2-tuple `(vals, vecs)` matching the rewired caller. Consistent.
- **Verification honesty:** T1 (partition-reassemble) + T2 (builder match) run here; T3 (distributed insert + solve) is cluster-only, validated by the `mpirun` triglobal eigenvalue-match run.
