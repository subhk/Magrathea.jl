# Distributed MPI — Phase 4 design (distributed onset/biglobal assembly + distributed P)

Date: 2026-06-05
Status: approved (design); implementation pending
Program: see [[project_distributed_mpi_roadmap]] — Phase 4 of 6.

## Goal

Distribute the onset/biglobal (constrained, hydrodynamic) matrix assembly so no rank
materializes the full `A`/`B`, and build the constraint reduction basis `P` from tiny
per-block constraint sub-blocks (not the full `A`). Combined with Phase 3's distributed
`S·A·P` reduction, this makes the onset/biglobal `:slepc` path fully distributed
(assembly + reduction). Serial dense path stays byte-identical. Triglobal and MHD
Galerkin are out of scope.

## Context

`assemble_matrices(op::LinearStabilityOperator)` (src/Stability/linear.jl) builds DENSE
`A, B = zeros(Complex{T}, n, n)`, filling per-`ℓ` blocks addressed by
`op.index_map[(ℓ, field)]` (UnitRange, `Nr` rows): diagonal blocks
`A[idx, idx] .+= …` (radial-operator combos of `radial_matrix(op, p, d)`), and
off-diagonal P↔T couplings `A[P_idx, T_idx] .+= …`. The serial constrained path
(`_constraint_reduction` → `_constrained_reduced_matrices`) needs DENSE `A`:
`_constraint_basis_block` does `nullspace(A[constraint_rows, idx])` and the reduction
does dense `mul!`.

Key facts enabling distribution:
- Rows of `A` correspond to `(ℓ, field)` equation blocks; the block layout is the
  Phase-1 `op.index_map`. Owned PETSc rows `[rstart, rend)` ↔ Julia `(rstart+1):rend`.
- `_constraint_basis_block` slices `A[constraint_rows, idx]` where `constraint_rows ⊂ idx`
  (the block's own rows) and columns `= idx` (same block). So the constraint sub-block
  is exactly the **diagonal block** of that `(ℓ, field)` restricted to the constraint
  rows — off-diagonal couplings go to *other* columns and never appear in it. Therefore
  `P`'s per-block nullspace can be computed from the diagonal block alone, with no full
  `A`.

## Architecture

### Core (PETSc-free, serial-verifiable)

1. **Per-field diagonal-block helpers** `_onset_diag_block(op, ℓ, field::Symbol) -> Matrix{Complex{T}}`
   (`field` ∈ `:P,:T,:Θ`). Extract the exact `A[idx, idx] .+= …` diagonal-block
   expressions currently inlined in `assemble_matrices` into this reusable function
   (one method per field, or a `field`-switch). Used by both the COO assembly and the
   constraint sub-block, so there is a single source of truth (no drift). Likewise a
   `_onset_offdiag_block(op, ℓ, ℓ′, :P, :T)` for the P↔T couplings, and the B-matrix
   diagonal blocks `_onset_B_diag_block`.

2. **`_assemble_onset_coo(op; owned_julia_rows::Union{Nothing,UnitRange{Int}}=nothing)`**
   — restructure the assembly loop to emit COO triplets for `A` and `B` from the
   diagonal/off-diagonal block helpers, returning a NamedTuple
   `(A_rows, A_cols, A_vals, B_rows, B_cols, B_vals, n)`. With `owned_julia_rows = R`:
   skip building a block whose row range doesn't intersect `R`, and emit only triplets
   whose global row ∈ `R` (a block is dense, so emit its entries with the owned-row
   filter). `owned = nothing` → full triplets.

3. **`assemble_matrices(op)`** becomes a thin wrapper: `c = _assemble_onset_coo(op)`;
   `A = Matrix(sparse(c.A_rows, c.A_cols, c.A_vals, c.n, c.n))`,
   `B = Matrix(...)`; return `(A, B, interior_dofs, boundary_dofs)` with `interior_dofs`/
   `boundary_dofs` computed exactly as today (read the current tail and preserve it).
   **Serial dense output numerically identical** — same block values; the only possible
   difference is FP accumulation order (COO `sparse()` sums duplicate `(row,col)`
   triplets — e.g. the Θ diagonal gets two contributions — which may reorder additions
   vs the sequential `.+=`). Verify with `≈` (machine-precision tolerance), not bitwise.

4. **`_constraint_subblock(op, ℓ, field) -> Matrix{Complex{T}}`** — the post-BC
   `A[constraint_rows, idx]`, i.e. the **tau boundary equations** for that block, NOT
   the diagonal radial block. `assemble_matrices` calls `impose_boundary_conditions!`
   which overwrites the tau rows with BC equations (Dirichlet `=1`; no-slip `D1[1,:]`/
   `D1[end,:]`; stress-free `r·D2`/`-r·D1+I`; fixed-T `=1`; fixed-flux `D1`), and every
   BC entry lands in the block's own columns (`A[row, X_idx]`). So `_constraint_subblock`
   reproduces `impose_boundary_conditions!`'s per-block rows for the constraint local
   rows — a small `(#constraint_rows × Nr)` matrix built directly from `op.cd.D1`/`D2`/
   identity and `p.mechanical_bc`/`p.thermal_bc`. (`_onset_diag_block` is NOT used here.)

5. **`_constraint_reduction_from_subblocks(op) -> ConstraintReduction{T}`** — same
   structure as `_constraint_reduction` but each block's basis is
   `nullspace(_constraint_subblock(op, ℓ, field))` (no full `A`). Returns the same
   `ConstraintReduction` type, consumable by `_constraint_projection_matrices` (Phase 3).

### Extension (PETSc/MPI, cluster-only)

- **Distributed onset assembly:** `_create_dist_mat(n)` → ownership → `_assemble_onset_coo(op; owned_julia_rows=(rstart+1):rend)` → `_fill_dist_mat!` (Phase-2 helpers) for `A` and `B`. **No BC application on the distributed matrix** — `_assemble_onset_coo` is pre-BC, and the boundary rows it contains are projected out by `S` (interior selector) in the reduction. (Unlike MHD tau, where BC rows survive into the solved matrix.)
- **Distributed P / S:** build `reduction = Magrathea._constraint_reduction_from_subblocks(op)` (replicated but cheap — only sub-block nullspaces, no full `A`); `S, P = Magrathea._constraint_projection_matrices(reduction, interior_dofs)`; distribute via the rectangular `_to_petsc_dist` (Phase 3). (P/S construction is replicated-but-cheap; the per-block nullspaces could be split across ranks in a later refinement, but the sub-blocks are tiny so replicated construction is acceptable and is NOT a full-`A` materialization.)
- **Reduce + solve:** Phase-3 `_reduce_dist(Adist, Sdist, Pdist)` / `_eps_solve_and_gather`; rank-0 `P·reduced` reconstruction. Update `_slepc_constrained_solve` (from Phase 3) to use the distributed assembly + sub-block reduction instead of the replicated full-`A` path.

## Data flow

`solve(::OnsetProblem; backend=:slepc)` → `_solve_constrained_slepc(op)` (now fully
distributed): per-rank owned-row assembly of `A`,`B` into distributed Mats; cheap
replicated `reduction` from sub-blocks → `S`,`P` → distribute; distributed `S·A·P`
reduce; EPS solve; rank-0 reconstruction. No rank holds the full `A`/`B`.

## Error handling

- `_onset_diag_block`/`_constraint_subblock` error on unknown `field`.
- `_assemble_onset_coo` with `owned_julia_rows` outside `1:n` → empty contributions
  (no error).
- Inherits Phase-0/3 SLEPc/PETSc errors.

## Testing

**Serial, here (the proofs):**
- **Partition-reassemble:** small onset op; full pre-densify `A_full = sparse(_assemble_onset_coo(op).A...)`; partition `1:n` into chunks; concatenate each chunk's owned COO; `sparse(...)` → assert `== A_full` (and B). Plus assert each chunk emits only owned rows.
- **Densify-wrapper regression:** `assemble_matrices(op)` dense `A`,`B` (and `interior_dofs`/`boundary_dofs`) equal the pre-refactor output — covered by `onset_api.jl`, `mean_flow_stability.jl` staying green, plus an explicit `A ≈ A_ref` check against a saved small case if convenient.
- **Sub-block match:** with `A_bc = assemble_matrices(op)[1]` (the **post-BC** dense
  matrix), for each `(ℓ, field)` assert `_constraint_subblock(op, ℓ, field) ≈
  A_bc[constraint_rows, idx]` (rtol 1e-10) — proves the directly-built tau sub-block
  equals the BC-overwritten slice of the assembled matrix, so
  `_constraint_reduction_from_subblocks` builds the same nullspaces (hence the same
  reduction) as the full-`A` path.
- **End-to-end serial reduction equivalence:** `reduction = _constraint_reduction_from_subblocks(op)`; `S, P = _constraint_projection_matrices(reduction, idofs)`; assert `S·A_full·P` is a valid reduced pencil whose leading eigenvalues match the `:krylovkit` reduced solve within tol (nullspace basis differs from the full-`A` reduction, but the reduced *spectrum* is invariant).

**Cluster (user, complex-scalar PETSc + MUMPS under mpirun):**
- Fully distributed onset/biglobal `:slepc`; `mpirun -n {1,2,4}` leading eigenvalues
  match `:krylovkit`; no rank materializes the full matrix.

## Files

- `src/Stability/linear.jl` — `_onset_diag_block`/`_onset_offdiag_block`/`_onset_B_diag_block`;
  `_assemble_onset_coo`; `assemble_matrices` thin wrapper; `_constraint_subblock`;
  `_constraint_reduction_from_subblocks`.
- `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl` — update `_slepc_constrained_solve` to the
  distributed-assembly + sub-block-reduction flow (reuse Phase-2/3 helpers).
- `test/distributed_onset.jl` (new) — the serial tests above; wired into `runtests.jl`.

## Scope / out of scope

- IN: distributed onset/biglobal dense assembly (COO + densify wrapper), `P` from
  per-block constraint sub-blocks, fully distributed `:slepc` path. Eliminates the
  replicated full `A`/`B` for onset/biglobal.
- OUT: triglobal (Phase 5), MHD Galerkin reduction, gather-to-all, non-MUMPS, splitting
  a single block's nullspace across ranks (sub-block construction stays replicated —
  cheap, not a full-`A`).

## Verification limitation

The COO assembly (partition-reassemble), the densify wrapper (serial regression), and
the sub-block/reduction equivalence are **fully verified here** (pure Julia). The
distributed assembly into PETSc, distributed `P`/`S`, distributed reduce, and the
distributed solve are cluster-only, validated by the `mpirun` eigenvalue-match run.
