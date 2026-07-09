# Distributed MPI — Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distribute onset/biglobal assembly (each rank builds only owned rows) and build the reduction basis `P` from per-block tau sub-blocks (no full `A`), making the onset/biglobal `:slepc` path fully distributed.

**Architecture:** A `_emit_block!(rows,cols,vals, row_idx, col_idx, block; owned)` COO helper replaces the dense `A[row_idx,col_idx] .+= block` writes; `_assemble_onset_coo(op; owned_julia_rows)` (radial blocks + basic-state blocks, per-ℓ block-skip) returns triplets; `assemble_matrices` densifies + applies BCs (numerically identical). `_constraint_subblock` reproduces the tau BC equations per block; `_constraint_reduction_from_subblocks` builds `P` from them. Extension rewires `_slepc_constrained_solve` to distributed assembly + sub-block reduction.

**Tech Stack:** Julia 1.12; PetscWrap/SlepcWrap/MPI/MUMPS (cluster).

**Spec:** `docs/superpowers/specs/2026-06-05-distributed-mpi-phase4-design.md`. Roadmap: `project_distributed_mpi_roadmap` memory.

---

## Environment / workflow notes
- Julia binary: `JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia`. Bash needs `dangerouslyDisableSandbox: true`. Julia 1.12.x.
- **T1, T2, T3 verifiable here** (pure Julia). **T4 cluster-only** (no PETSc). **T5** wiring/docs.
- **Commits:** no `git commit` without explicit permission — each commit step pauses for the user.
- Convention: owned PETSc `[rstart,rend)` ↔ Julia `(rstart+1):rend`. `assemble_matrices` applies BCs (`impose_boundary_conditions!`) to boundary rows ONLY; interior rows are pre/post-BC identical; the constrained reduction selects interior rows (so the distributed `A` needs no BC application).

---

## Task 1: COO emit helper + distribute the radial assembly (onset)

**Files:** Modify `src/Stability/linear.jl`; create `test/distributed_onset.jl`.

- [ ] **Step 1: failing partition-reassemble test (onset, no basic state)** — `test/distributed_onset.jl`:
```julia
using Test
using SparseArrays
using Magrathea

@testset "onset interior COO partition-reassembles (pre-BC)" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
    op = LinearStabilityOperator(params)
    full = Magrathea._assemble_onset_coo(op)
    n = full.n
    A_pre = sparse(full.A_rows, full.A_cols, full.A_vals, n, n)
    B_pre = sparse(full.B_rows, full.B_cols, full.B_vals, n, n)
    cuts = [0, fld(n,3), fld(2n,3), n]
    Ar=Int[]; Ac=Int[]; Av=ComplexF64[]; Br=Int[]; Bc=Int[]; Bv=ComplexF64[]
    for i in 1:3
        R = (cuts[i]+1):cuts[i+1]
        c = Magrathea._assemble_onset_coo(op; owned_julia_rows=R)
        @test all(r -> r in R, c.A_rows)
        @test all(r -> r in R, c.B_rows)
        append!(Ar,c.A_rows); append!(Ac,c.A_cols); append!(Av,c.A_vals)
        append!(Br,c.B_rows); append!(Bc,c.B_cols); append!(Bv,c.B_vals)
    end
    @test sparse(Ar,Ac,Av,n,n) == A_pre
    @test sparse(Br,Bc,Bv,n,n) == B_pre
end
```

- [ ] **Step 2: run, verify FAIL** (`_assemble_onset_coo` undefined).

- [ ] **Step 3: add the COO emit helper to `src/Stability/linear.jl`**
```julia
"""Push the dense `block` (size length(row_idx)×length(col_idx)) into COO triplets at
global block position (row_idx, col_idx), keeping only rows in `owned` (or all if
`owned===nothing`). 1-based global indices."""
function _emit_block!(rows, cols, vals, row_idx, col_idx, block;
                      owned::Union{Nothing,UnitRange{Int}}=nothing)
    @inbounds for (jc, c) in enumerate(col_idx), (ir, r) in enumerate(row_idx)
        if owned === nothing || r in owned
            v = block[ir, jc]
            push!(rows, r); push!(cols, c); push!(vals, v)
        end
    end
    return nothing
end
```

- [ ] **Step 4: write `_assemble_onset_coo`** by converting the body of `assemble_matrices` (lines ~321–416 for poloidal, ~425–450 for toroidal, the Θ blocks) so that every `A[r_idx, c_idx] .+= block` / `B[r_idx, c_idx] = block` becomes `_emit_block!(A_rows,A_cols,A_vals, r_idx, c_idx, Complex.(block); owned=owned_julia_rows)` (and B-arrays). Add the `owned_julia_rows` kwarg. Guard the two per-ℓ loops: at the top of `for ℓ in poloidal_ls` and `for ℓ in toroidal_ls`, skip if NONE of that ℓ's row-blocks are owned:
```julia
        if owned_julia_rows !== nothing
            rngs = (get(op.index_map,(ℓ,:P),nothing), get(op.index_map,(ℓ,:T),nothing), get(op.index_map,(ℓ,:Θ),nothing))
            any(rng -> rng !== nothing && !isempty(intersect(rng, owned_julia_rows)), rngs) || continue
        end
```
Do NOT call `impose_boundary_conditions!` here (pre-BC). Do NOT call `add_basic_state_operators!` here yet (Task 2). Return:
```julia
    return (A_rows=A_rows, A_cols=A_cols, A_vals=A_vals,
            B_rows=B_rows, B_cols=B_cols, B_vals=B_vals, n=n)
```
(Declare the six COO vectors at the top; keep all the `radial_matrix`/coefficient computations as-is — only the writes change.) For this task, if `op.params.basic_state !== nothing`, `error("basic state assembly distributed in Task 2")` at the top of `_assemble_onset_coo` (lifted in Task 2) so onset is correct now and biglobal isn't silently wrong.

- [ ] **Step 5: make `assemble_matrices` a thin wrapper**
```julia
function assemble_matrices(op::LinearStabilityOperator{T}) where {T<:Real}
    c = _assemble_onset_coo(op)
    A = Matrix(sparse(c.A_rows, c.A_cols, c.A_vals, c.n, c.n))
    B = Matrix(sparse(c.B_rows, c.B_cols, c.B_vals, c.n, c.n))
    if op.params.basic_state !== nothing
        bs_ops = build_basic_state_operators(op.params.basic_state, op, op.params.m)
        add_basic_state_operators!(A, B, bs_ops, op, op.params.m)
    end
    impose_boundary_conditions!(A, B, op)
    # interior/boundary dofs — preserve the EXACT original computation (lines ~454–471)
    <PASTE the original is_boundary / boundary_dofs / interior_dofs block verbatim, using c.n for n>
    return A, B, interior_dofs, boundary_dofs
end
```
READ the original lines 454–471 and paste verbatim. (Note: the basic-state add stays dense in the wrapper for now; Task 2 moves it into `_assemble_onset_coo`.)

- [ ] **Step 6: run partition test → PASS; serial regression** — `$JL --project=. -e 'using Test; @testset "t" begin include("test/distributed_onset.jl") end'` PASS; then `$JL --project=. -e 'using Test; include("test/onset_api.jl")' 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done`.

- [ ] **Step 7: Commit (ASK USER FIRST)** — `git add src/Stability/linear.jl test/distributed_onset.jl` / `git commit -m "feat(mpi): distributed onset COO assembly + densify wrapper"`

---

## Task 2: distribute the basic-state contribution (biglobal)

**Files:** Modify `src/BasicStates/basic_state_operators.jl`, `src/Stability/linear.jl`, `test/distributed_onset.jl`.

- [ ] **Step 1: failing biglobal partition-reassemble test** — append to `test/distributed_onset.jl` a testset like Task 1 Step 1 but building a small biglobal `op` (with a conduction/thermal-wind basic state — mirror `test/mean_flow_stability.jl`'s setup) and asserting the partition reassembles the pre-BC `A`/`B` including basic-state terms.

- [ ] **Step 2: run, verify FAIL** (biglobal hits the `error("basic state … Task 2")`).

- [ ] **Step 3: add a COO-emitting variant of the basic-state add.** In `src/BasicStates/basic_state_operators.jl`, add `add_basic_state_operators_coo!(A_rows,A_cols,A_vals, B_rows,B_cols,B_vals, bs_ops, op, m; owned_julia_rows=nothing)` that mirrors `add_basic_state_operators!` but replaces each `A[out_idx, in_idx] .+= block` with `_emit_block!(A_rows,A_cols,A_vals, out_idx, in_idx, Complex.(block); owned=owned_julia_rows)` (and B). READ the full `add_basic_state_operators!` (line ~619+) and convert every block write; skip a coupling whose `out_idx` rows aren't owned (`isempty(intersect(out_idx, owned)) && continue` per coupling, before computing the block). Keep `add_basic_state_operators!` (dense) for the serial wrapper, OR have it delegate to the COO variant + densify — pick the lower-drift option (delegating is DRY: dense version = densify of COO variant applied to its own arrays; but simplest is to keep both and rely on the partition test + regression to catch drift).

- [ ] **Step 4: wire into `_assemble_onset_coo`** — remove the `error(...)` guard; when `op.params.basic_state !== nothing`, after the radial blocks, call
  `bs_ops = build_basic_state_operators(op.params.basic_state, op, op.params.m); add_basic_state_operators_coo!(A_rows,…,B_vals, bs_ops, op, op.params.m; owned_julia_rows=owned_julia_rows)`. Remove the now-redundant basic-state branch from the `assemble_matrices` wrapper (it comes through the COO now).

- [ ] **Step 5: run biglobal partition test → PASS; regression** — biglobal partition test PASS; `$JL --project=. -e 'using Test; include("test/mean_flow_stability.jl")' 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done`.

- [ ] **Step 6: Commit (ASK USER FIRST)** — `git add src/BasicStates/basic_state_operators.jl src/Stability/linear.jl test/distributed_onset.jl` / `git commit -m "feat(mpi): distributed basic-state (biglobal) COO assembly"`

---

## Task 3: `_constraint_subblock` (tau BC equations) + reduction-from-subblocks

**Files:** Modify `src/Stability/linear.jl`; extend `test/distributed_onset.jl`.

- [ ] **Step 1: failing sub-block-match test** — append:
```julia
@testset "constraint sub-blocks match the BC-overwritten A rows" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
    op = LinearStabilityOperator(params)
    A_bc, _, _, _ = assemble_matrices(op)         # post-BC dense
    for ℓ in op.l_sets[:P]
        idx = op.index_map[(ℓ, :P)]
        ri, it, ot, ro = Magrathea.poloidal_tau_indices(idx)
        sub = Magrathea._constraint_subblock(op, ℓ, :P)
        @test sub ≈ A_bc[[ri, it, ot, ro], idx] rtol=1e-10
    end
    for ℓ in op.l_sets[:T]
        idx = op.index_map[(ℓ, :T)]; r1, r2 = Magrathea.toroidal_boundary_indices(idx)
        @test Magrathea._constraint_subblock(op, ℓ, :T) ≈ A_bc[[r1, r2], idx] rtol=1e-10
    end
    for ℓ in op.l_sets[:Θ]
        idx = op.index_map[(ℓ, :Θ)]; r1, r2 = Magrathea.temperature_boundary_indices(idx)
        @test Magrathea._constraint_subblock(op, ℓ, :Θ) ≈ A_bc[[r1, r2], idx] rtol=1e-10
    end
end
```

- [ ] **Step 2: run, verify FAIL** (`_constraint_subblock` undefined).

- [ ] **Step 3: implement `_constraint_subblock`** in `src/Stability/linear.jl`, reproducing `impose_boundary_conditions!`'s per-block tau rows restricted to the block's own columns (an `(#constraint_rows × Nr)` matrix; `Nr = length(idx)`). READ `impose_boundary_conditions!` (lines ~259+) and mirror exactly per field:
  - `:P` rows (ri, inner_tau, outer_tau, ro): Dirichlet rows are `e_1`/`e_Nr` unit rows (`A[ri,ri]=1` → local row has 1 at local col 1); inner/outer tau are `D1[1,:]`/`D1[end,:]` (no-slip) or `r[1]·D2[1,:]`/`r[end]·D2[end,:]` (stress-free), as block-local Nr-length rows.
  - `:T` rows (riT, roT): unit rows (no-slip) or `-r·D1[1or end,:]` with `+1` on the diagonal (stress-free).
  - `:Θ` rows (riΘ, roΘ): unit rows (fixed_temperature) or `D1[1,:]`/`D1[end,:]` (fixed_flux).
  Return the rows stacked in the SAME order as the constraint-row list used by
  `_constraint_basis_block` (`[ri, inner_tau, outer_tau, ro]` for P; `[riT, roT]`; `[riΘ, roΘ]`).
  Build from `op.cd.D1`, `op.cd.D2`, `op.r`, `op.params.mechanical_bc`, `op.params.thermal_bc`.

- [ ] **Step 4: implement `_constraint_reduction_from_subblocks(op)`** — a copy of `_constraint_reduction` that uses `nullspace(_constraint_subblock(op, ℓ, field))` for each block's basis (no full `A`), returning a `ConstraintReduction{T}` with the same block layout.

- [ ] **Step 5: failing end-to-end spectrum test** — append:
```julia
@testset "reduction from sub-blocks gives the same reduced spectrum" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
    op = LinearStabilityOperator(params)
    A_full, B_full, idofs, bdofs = assemble_matrices(op)
    A_ref, B_ref, _ = Magrathea._constrained_reduced_matrices(A_full, B_full, op, idofs, bdofs)
    red = Magrathea._constraint_reduction_from_subblocks(op)
    S, P = Magrathea._constraint_projection_matrices(red, idofs)
    A_sub = Matrix(S * A_full * P); B_sub = Matrix(S * B_full * P)
    # reduced spectra agree (nullspace basis may differ; spectrum is invariant)
    λref = sort(eigvals(A_ref, B_ref); by=abs)
    λsub = sort(eigvals(A_sub, B_sub); by=abs)
    @test λref ≈ λsub rtol=1e-8
end
```

- [ ] **Step 6: run all three new testsets → PASS** — `$JL --project=. -e 'using Test; @testset "t" begin include("test/distributed_onset.jl") end'`.

- [ ] **Step 7: Commit (ASK USER FIRST)** — `git add src/Stability/linear.jl test/distributed_onset.jl` / `git commit -m "feat(mpi): constraint sub-blocks + reduction-from-subblocks (no full A)"`

---

## Task 4: extension — fully distributed onset/biglobal `:slepc` (NOT runnable here)

**Files:** Modify `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl`.

> Cluster-only. Verify: `Meta.parseall`, symbol-audit, `CORE_OK`, full default suite green.

- [ ] **Step 1: rewrite `_slepc_constrained_solve(op; …)`** (currently the Phase-3 replicated-full-`A` version) to the fully distributed flow:
```julia
function _slepc_constrained_solve(op; nev::Int, sigma, which::Symbol, tol::Float64, maxiter::Int)
    _INITIALIZED[] || error("call Magrathea.slepc_init!() once before a :slepc solve")
    PetscScalar <: Real && error("PETSc/SLEPc must be built with complex scalars")
    # reduction from sub-blocks (cheap, replicated; no full A)
    red = Magrathea._constraint_reduction_from_subblocks(op)
    # interior_dofs: recompute via the boundary mask (same as assemble_matrices tail) WITHOUT full A
    idofs = Magrathea._onset_interior_dofs(op)        # add this tiny helper in core (see note)
    S, P = Magrathea._constraint_projection_matrices(red, idofs)
    nfull = red.n_full; nred = red.n_reduced
    # distributed assembly of A, B (owned rows only)
    Amat, rs, re = _create_dist_mat(nfull); Bmat, _, _ = _create_dist_mat(nfull)
    coo = Magrathea._assemble_onset_coo(op; owned_julia_rows=(rs+1):re)
    _fill_dist_mat!(Amat, coo.A_rows, coo.A_cols, coo.A_vals, rs, re)
    _fill_dist_mat!(Bmat, coo.B_rows, coo.B_cols, coo.B_vals, rs, re)
    # distribute S, P; reduce; solve; reconstruct
    Sdist = _to_petsc_dist(S, nred, nfull); Pdist = _to_petsc_dist(P, nfull, nred)
    Ared = _reduce_dist(Amat, Sdist, Pdist); Bred = _reduce_dist(Bmat, Sdist, Pdist)
    MatDestroy(Amat); MatDestroy(Bmat); MatDestroy(Sdist); MatDestroy(Pdist)
    vals, vecs_red, info = _eps_solve_and_gather(Ared, Bred, nred;
        nev=nev, sigma=sigma, which=which, selection=:maxreal, tol=tol, maxiter=maxiter)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    vecs_full = (rank == 0 && size(vecs_red,2) > 0) ?
        Matrix{ComplexF64}(P * vecs_red) : Matrix{ComplexF64}(undef, nfull, 0)
    return vals, vecs_full, info
end
```
NOTE: add a tiny core helper `_onset_interior_dofs(op)` = the boundary-mask computation from `assemble_matrices`' tail (lines ~454–471) factored out, so the extension gets `interior_dofs` without assembling the full `A`. Add it in Task 1 or here in core; report. (It depends only on `op.index_map`/`l_sets` + the tau-index helpers — pure, serial-testable: assert it equals `assemble_matrices(op)`'s `interior_dofs`.)

- [ ] **Step 2: Parse + CORE_OK + full suite** — `PARSE_OK`; `CORE_OK`; `$JL --project=. test/runtests.jl 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done` (the `:krylovkit` onset/biglobal default unaffected; `_solve_constrained_slepc` only runs under the loaded extension).
- [ ] **Step 3: symbol-audit** any new ccalls (none expected beyond Phase 2/3) + report.
- [ ] **Step 4: Commit (ASK USER FIRST)** — `git add ext/MagratheaSlepcExt/ src/Stability/linear.jl` / `git commit -m "feat(mpi): fully distributed onset/biglobal :slepc (no replicated full A)"`

---

## Task 5: wire tests + full suite + docs

- [ ] **Step 1:** add `include("distributed_onset.jl")` to `test/runtests.jl`.
- [ ] **Step 2:** append a `PETSC_DIR`-guarded cluster note-test to `test/distributed_onset.jl` (mirror the Phase-2/3 skip pattern).
- [ ] **Step 3:** full suite green — `$JL --project=. test/runtests.jl 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done`; confirm the onset partition test appears.
- [ ] **Step 4:** README — onset/biglobal `:slepc` now fully distributed (assembly + reduction; no replicated full matrix). Triglobal/MHD-Galerkin still replicated.
- [ ] **Step 5: Commit (ASK USER FIRST)** — `git add test/runtests.jl test/distributed_onset.jl README.md` / `git commit -m "test+docs: distributed onset/biglobal"`

---

## Self-review notes
- **Spec coverage:** `_onset` COO assembly + densify wrapper (T1); basic-state COO (T2, biglobal); `_constraint_subblock` (tau BC equations) + `_constraint_reduction_from_subblocks` (T3); `_onset_interior_dofs` (T4 note); fully distributed `_slepc_constrained_solve` (T4); suite + docs (T5). Per-spec `_onset_diag_block` was supplanted — `_constraint_subblock` uses the BC formulas directly (the spec correction), so no diag-block helper is needed; the COO assembly inlines the existing radial-block expressions via `_emit_block!`.
- **Placeholder note:** T1 Step 5 and T2/T3 Step 3 say "READ the original … and mirror/paste verbatim" for the boundary-dof tail and the BC formulas — bounded conversions of existing code, with the partition-reassemble + sub-block-match + regression tests as guards.
- **Type consistency:** `_assemble_onset_coo` returns `(A_rows,A_cols,A_vals,B_rows,B_cols,B_vals,n)`; `_emit_block!(rows,cols,vals,row_idx,col_idx,block;owned)`; `_constraint_subblock(op,ℓ,field)::Matrix`; `_constraint_reduction_from_subblocks(op)::ConstraintReduction`; consumed by Phase-3 `_constraint_projection_matrices`. Consistent.
- **Verification honesty:** T1–T3 (partition-reassemble, sub-block match, spectrum equivalence) + `_onset_interior_dofs` run here; T4 (distributed assembly/reduce/solve into PETSc) is cluster-only, validated by the `mpirun` eigenvalue-match run.
