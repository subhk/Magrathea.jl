# Distributed MPI — Phase 2 design (distributed MHD tau assembly)

Date: 2026-06-04
Status: approved (design); implementation pending
Program: see [[project_distributed_mpi_roadmap]] — Phase 2 of 6.

## Goal

Distribute the MHD tau-method matrix assembly: each MPI rank computes and inserts
only the matrix rows it owns, with no full Julia matrix materialized on any rank.
This is the first phase that distributes assembly *work* (not just the solve), and
it establishes the pattern (index_map → owned rows → skip non-owned block compute →
direct PETSc insert) that Phases 3–5 will follow. MHD tau only; reductions and the
other problem types come later.

## Context

`assemble_mhd_matrices(op)` builds COO triplets and `sparse()`s them. Row layout is
five contiguous sections `[u][v][f][g][h]` (poloidal velocity, toroidal velocity,
poloidal magnetic, toroidal magnetic, temperature); section `X` has `length(op.ll_X)`
modes, each `n_per_mode = N+1` rows. Within each section's `for (k, l) in
enumerate(op.ll_X)` loop, every `add_block!` writes to `row_base = (section_offset +
k - 1) * n_per_mode` — i.e. **all rows a given per-mode loop iteration writes belong
to that one mode's equations**. So skipping a non-owned mode's loop iteration cleanly
omits exactly that mode's rows.

Phase 1 provides `owned_block_ranges` / `row_to_dof` over an `index_map`. Owned PETSc
rows `[rstart, rend)` (0-based, half-open) correspond to the contiguous Julia row
range `(rstart + 1):rend` (1-based inclusive).

## Architecture

### Core (PETSc-free)

1. **`_mhd_index_map(op) -> Dict{Tuple{Int,Symbol}, UnitRange{Int}}`** (MHD module).
   Reproduces the section layout as the Phase-1 `index_map` shape: iterate sections in
   order `(:u, op.ll_u), (:v, op.ll_v), (:f, op.ll_f), (:g, op.ll_g), (:h, op.ll_h)`
   with a running row offset; for each mode `l` add `(l, section) => (off+1):(off+n_per_mode)`,
   `off += n_per_mode`. Used both for ownership decisions and as the canonical MHD DOF
   map for later phases.

2. **`_assemble_mhd_coo(op; owned_julia_rows::Union{Nothing,UnitRange{Int}}=nothing)`**.
   Extract the existing COO-building body of `assemble_mhd_matrices` into this function,
   returning a NamedTuple of raw triplets and metadata with fields
   `A_rows, A_cols, A_vals, B_rows, B_cols, B_vals, n, interior_dofs, info`.
   - `owned_julia_rows === nothing` (serial): no skipping, no filtering — produces the
     full triplet set, identical to today.
   - `owned_julia_rows = R` (distributed): at the **top of each of the five per-mode
     loops**, compute that mode's row range `mr = (row_base+1):(row_base+n_per_mode)`
     and `isempty(intersect(mr, R)) && continue` — skipping the block computation for
     non-owned modes. `add_block!` gains an `owned` parameter and pushes a triplet only
     when its global row `∈ R` (covers boundary modes split across ranks).

   `_assemble_mhd_coo` covers ONLY the interior block assembly — the part that today
   runs *before* `sparse()`. The tau boundary-condition row overwrites
   (`apply_velocity/magnetic/temperature_boundary_conditions!`, applied to the
   assembled sparse matrix at lines 559–570) are NOT part of it (COO addition cannot
   zero-then-overwrite a row).

3. **`assemble_mhd_matrices(op)`** becomes a thin wrapper: call `_assemble_mhd_coo(op)`
   (owned = nothing), `sparse(...)` the triplets, then apply the existing
   `apply_*_boundary_conditions!` exactly as today → `(A, B, interior_dofs, info)`.
   **Serial behavior byte-identical.**

### Extension (PETSc/MPI, cluster-only)

The Mat must exist before its ownership range is known (PETSc decides the row split at
`MatSetSizes`/`MatSetFromOptions`), but the owned triplets can only be assembled once
that range is known. So the Mat creation and the value insertion are two separate
steps, with the core assembly call sandwiched between them:

4a. **`_create_dist_mat(n) -> (mat, rstart, rend)`** — `MatCreate(COMM_WORLD)`,
   `MatSetSizes(PETSC_DECIDE,PETSC_DECIDE,n,n)`, `MatSetFromOptions`,
   `MatGetOwnershipRange` → return the (empty) Mat and its 0-based half-open range.

4b. **`_fill_dist_mat!(mat, rows, cols, vals, rstart, rend)`** — preallocate via a new
   pure core helper `_owned_coo_nnz(rows, cols, rstart, rend) -> (d_nnz, o_nnz)` (per
   owned row, diagonal band `[rstart,rend)`), then `MatSetValue` each triplet (1-based
   Julia → 0-based PETSc), `MatAssemblyBegin/End`. No full Julia matrix.

5. **MHD `:slepc` solve flow (inverted)** in the extension. `A` and `B` are `n×n` on
   the same comm, so they get the **same** ownership split — create both, take one
   range, assemble owned COO for both in a single core call:
   ```
   (Amat, rstart, rend) = _create_dist_mat(n)
   (Bmat, _, _)         = _create_dist_mat(n)        # identical split
   coo = Magrathea._assemble_mhd_coo(op; owned_julia_rows = (rstart+1):rend)
   _fill_dist_mat!(Amat, coo.A_rows, coo.A_cols, coo.A_vals, rstart, rend)
   _fill_dist_mat!(Bmat, coo.B_rows, coo.B_cols, coo.B_vals, rstart, rend)
   # then distributed BC overwrites on owned rows (6):
   _apply_dist_bcs!(Amat, Bmat, op, rstart, rend)
   ```

6. **Distributed boundary conditions** `_apply_dist_bcs!(Amat, Bmat, op, rstart, rend)`
   (extension). The interior fill (5) wrote junk into the future-BC tau rows; the BC
   step overwrites them. Distributed analog of the serial `A[row,:].=0;
   A[row,block]=vals`: for each tau/BC row the serial `apply_*_boundary_conditions!`
   would set, if it is owned (`∈ [rstart,rend)`), `MatZeroRows` that row then
   `MatSetValues` the BC-equation entries. This ports the velocity (no-slip/stress-free
   poloidal+toroidal), magnetic (f/g), and temperature (fixed-T/flux) BC row formulas
   to operate on the distributed Mat for owned rows only. `MatZeroRows` is collective in
   PETSc — all ranks call it with their owned BC-row lists. **This is the largest
   cluster-only piece: the BC row formulas are intricate and unverifiable here.**

   This replaces Phase 0's replicated full-matrix insert **for the MHD path only**.
   onset/biglobal/triglobal/galerkin keep the Phase-0 replicated `_to_petsc_dist` until
   their phases. With `N=1` (one rank), the range is `[0,n)` → `R = 1:n` → owns
   everything → identical to serial. `_owned_coo_nnz` is pure and lives in core
   (serial-testable). `_assemble_mhd_coo` returns a NamedTuple (fields `A_rows`,
   `A_cols`, `A_vals`, `B_rows`, `B_cols`, `B_vals`, `n`, `interior_dofs`, `info`) so
   the extension reads triplets by name.

## Data flow

`solve(MHDProblem; backend=:slepc)` → extension MHD branch: build distributed `Amat`,
`Bmat` via the inverted flow (each calls `_assemble_mhd_coo` with that Mat's owned
range) → Phase-0 EPS solve → rank-0 eigenvector gather → result. Each rank computes
~1/N of the blocks and stores only its owned rows.

## Error handling

- `_assemble_mhd_coo` with an `owned_julia_rows` outside `1:n` → the intersection logic
  naturally yields empty contributions (a rank owning no MHD rows produces empty
  triplets); no error needed.
- `_to_petsc_dist_coo`: inherits Phase-0 errors (uninitialized SLEPc, real-scalar
  PETSc). Empty owned triplets (rank with no rows) → still creates/assembles its empty
  row block (valid in PETSc).

## Testing

**Serial, runs here (the core correctness proof — no MPI needed):**
- `_mhd_index_map`: for a small MHD operator, assert the row ranges tile `1:n`
  contiguously in `[u][v][f][g][h]` order with `n_per_mode` each, and that
  `row_to_dof`/`dof_to_row` round-trip against it.
- **Partition-and-reassemble (key test):** pick a small MHD problem; build the full
  **pre-BC** interior matrices from `coo = _assemble_mhd_coo(op)` (owned=nothing) →
  `A_pre = sparse(coo.A_rows, coo.A_cols, coo.A_vals, n, n)` (and `B_pre`). Partition
  `1:n` into several contiguous chunks `R_1, …, R_p`; for each call
  `_assemble_mhd_coo(op; owned_julia_rows=R_i)`; concatenate all chunks' triplets and
  `sparse(...)` → assert **exactly equals** `A_pre`/`B_pre`. This proves the block-skip
  + owned-row-filter assembles every interior row exactly once with the same values,
  across an arbitrary partition — distributed *interior* assembly is correct without
  PETSc/MPI. (BCs are not in `_assemble_mhd_coo`; they are verified on the cluster — see
  below.)
- **Serial wrapper regression:** `assemble_mhd_matrices(op)` (now = `sparse(_assemble_mhd_coo(op))`
  then `apply_*_boundary_conditions!`) must equal the pre-refactor output — covered by
  the existing MHD suite (mhd_boundary_conditions.jl, mhd_galerkin.jl) staying green.
- `_owned_coo_nnz`: hand-built triplets + ownership band → assert exact `d_nnz`/`o_nnz`
  (mirrors the Phase-0 `_petsc_owned_nnz` test but from COO).
- Serial regression: `assemble_mhd_matrices` output unchanged (existing MHD suite green).

**Cluster (user, complex-scalar PETSc + MUMPS under mpirun):**
- `_create_dist_mat`/`_fill_dist_mat!` + `_apply_dist_bcs!` + the inverted MHD solve
  flow; `mpirun -n {1,2,4}` eigenvalues match `:krylovkit` within tol (this only holds
  once the distributed BCs are correct — so the cluster run is the BC verification);
  per-rank assembly touches only owned rows.

## Files

- `src/MHD/assembly.jl` — add `_mhd_index_map`; extract `_assemble_mhd_coo` (with the
  `owned_julia_rows` kwarg, per-mode `continue` guards, `add_block!` filter); make
  `assemble_mhd_matrices` a thin wrapper. Add `_owned_coo_nnz`.
- `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl` — `_create_dist_mat`, `_fill_dist_mat!`,
  `_apply_dist_bcs!`; rewire the MHD branch of `_slepc_solve` to the inverted
  distributed-assembly + distributed-BC flow.
- `test/distributed_assembly.jl` (new) — the serial tests above; wired into `runtests.jl`.

## Scope / out of scope

- IN: MHD tau distributed assembly (index_map, owned-COO with block-skipping, direct
  PETSc insert), serial partition-reassemble verification.
- OUT: distributed reductions / Galerkin (Phase 3); onset/biglobal (Phase 4); triglobal
  (Phase 5); gather-to-all; non-MUMPS. Serial assembler semantics unchanged.

## Verification limitation

Distributed **interior** assembly correctness is **fully verified here** via
partition-and-reassemble (pure Julia), and the serial wrapper via the existing suite.
The PETSc pieces — `_create_dist_mat`/`_fill_dist_mat!` and especially the
**distributed BC application** (`_apply_dist_bcs!`, the largest and most intricate
cluster-only component) — plus the inverted solve flow, need cluster validation under
`mpirun` on a complex-scalar PETSc+MUMPS build. The `mpirun` eigenvalue-match test is
what actually verifies the distributed BCs.
