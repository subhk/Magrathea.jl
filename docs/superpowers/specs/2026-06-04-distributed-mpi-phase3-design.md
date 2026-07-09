# Distributed MPI — Phase 3 design (distributed constrained reduction)

Date: 2026-06-04
Status: approved (design); implementation pending
Program: see [[project_distributed_mpi_roadmap]] — Phase 3 of 6 (highest-risk).

## Goal

Distribute the constrained (tau-elimination) reduction used by onset/biglobal:
express the existing block-wise reduce as a sparse triple product `A_red = S·A·P`,
then perform it with distributed PETSc `MatMatMult`. The matrix-form equivalence is
verified serially here; the distributed products are cluster-only. Establishes the
distributed-reduction pattern; Phase 4 wires it to distributed onset/biglobal
assembly. MHD Galerkin reduction is out of scope.

## Context

`_constrained_reduced_matrices` (src/Stability/linear.jl) reduces the full tau-form
`A_full` (n_full×n_full, dense, currently replicated-assembled) to
`A_red` (n_reduced×n_reduced) via, per field block:
`A_red[:, block.reduced_indices] = A_full[interior_dofs, block.full_indices] · block.basis`.
`block.basis` (`ConstraintBasisBlock`) is the dense nullspace of that block's tau
constraint rows; `interior_dofs` (length `n_reduced`) are the non-boundary rows;
`ConstraintReduction.blocks` is block-diagonal over `(ℓ, field)`. Eigenvectors are
reconstructed with `_reconstruct_full_vector(reduction, v) = P·v` (block-wise).

Rewriting block-wise as matrices: with `S` an `n_reduced×n_full` row selector
(`S[i, interior_dofs[i]] = 1`) and `P` the `n_full×n_reduced` block-diagonal basis,
`A_red = S · A_full · P` exactly (since `S·A_full = A_full[interior_dofs, :]` and the
block-diagonal `P` performs the per-block column projection). PETSc distributes this
with `MatMatMult`. Eigenvector reconstruction becomes `full = P · reduced`.

The current onset/biglobal `:slepc` path (SLEPc serial backend) computes the reduced
pencil **block-wise and replicated**, then hands `sparse(A_red), sparse(B_red)` to the
Phase-0 distributed solve. Phase 3 replaces the reduction step with distributed
`MatMatMult`. (Full-`A` assembly remains replicated until Phase 4; because `P`'s
nullspaces are built from `A`'s tau rows, `P`-construction also stays replicated until
then — flagged.)

## Architecture

### Core (PETSc-free, serial-verifiable)

**`_constraint_projection_matrices(reduction::ConstraintReduction{T}, interior_dofs::Vector{Int}) -> (S, P)`**
(src/Stability/linear.jl):
- `P::SparseMatrixCSC{Complex{T},Int}` of size `reduction.n_full × reduction.n_reduced`:
  for each `block in reduction.blocks`, write `block.basis` into rows
  `block.full_indices`, columns `block.reduced_indices`.
- `S::SparseMatrixCSC{Complex{T},Int}` of size `reduction.n_reduced × reduction.n_full`:
  `S[i, interior_dofs[i]] = 1` for `i in 1:reduction.n_reduced`
  (`length(interior_dofs) == reduction.n_reduced`).
- Built from COO triplets (no dense temporaries). Pure; no PETSc.

This is the only new core code. It does NOT change `_constrained_reduced_matrices` or
the serial/KrylovKit path.

### Extension (PETSc/MPI, cluster-only)

- **`_mat_mat_mult(A, B) -> C`** in `raw_petsc.jl` — raw ccall to PETSc
  `MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C)` (PetscWrap 0.1.5 has no
  wrapper). Returns a new distributed `Mat` handle wrapped as a `PetscMat`.
- **`_reduce_dist(Amat, Smat, Pmat) -> A_red`** — `_mat_mat_mult(_mat_mat_mult(Smat, Amat), Pmat)`,
  destroying the intermediate.
- **Onset/biglobal `:slepc` flow** (new MHD-style hook, e.g. `_SLEPC_CONSTRAINED_SOLVER`,
  registered by the extension, used by `solve(::OnsetProblem/::BiglobalProblem; backend=:slepc)`):
  1. assemble full `A_full, B_full` (replicated, as today) and the `ConstraintReduction`
     via the existing `_constraint_reduction`;
  2. `S, P = Magrathea._constraint_projection_matrices(reduction, interior_dofs)`;
  3. distribute `A_full`/`B_full` (Phase-0 `_to_petsc_dist(sparse(·))`), `S`, `P`
     (`_to_petsc_dist`);
  4. `A_red = _reduce_dist(Adist, Sdist, Pdist)`, `B_red = _reduce_dist(Bdist, Sdist, Pdist)`;
  5. `_eps_solve_and_gather(A_red, B_red, n_reduced; …)` → reduced eigenvalues (all
     ranks) + reduced eigenvectors gathered to rank 0;
  6. rank 0: reconstruct `full = P · reduced` (P is the replicated Julia sparse) per
     eigenvector; workers keep empty.
- Keep the existing generic `_slepc_solve(A,B)` and the current onset/biglobal block-wise
  replicated path as the non-distributed fallback / for `N=1` if simpler.

## Data flow

`solve(::OnsetProblem; backend=:slepc)` → `_solve_constrained_slepc(op)` (extension):
replicated full assembly + `S,P` → distribute → distributed `S·A·P` reduce →
distributed EPS solve → rank-0 reduced eigenvectors → `P·reduced` reconstruction →
`StabilityResult` (eigenvalues all ranks; full eigenvectors rank 0).

## Error handling

- Inherits Phase-0 errors (uninitialized SLEPc, real-scalar PETSc). `MatMatMult`
  failure → assertion error from the ccall. `_constraint_projection_matrices` requires
  `length(interior_dofs) == reduction.n_reduced` → error otherwise.

## Testing

**Serial, here (correctness proof):**
- `_constraint_projection_matrices` on a small onset operator:
  - `A_full,B_full,idofs,bdofs = assemble_matrices(op)`;
  - `A_red_ref, B_red_ref, reduction = Magrathea._constrained_reduced_matrices(A_full, B_full, op, idofs, bdofs)`
    — **reuse the returned `reduction`** (its nullspace bases) for `S,P`; do NOT call
    `_constraint_reduction` separately (`nullspace` is not unique, so a fresh call would
    yield different bases and break the equality);
  - `S, P = Magrathea._constraint_projection_matrices(reduction, idofs)`;
  - assert `Matrix(S * A_full * P) ≈ A_red_ref` and `≈ B_red_ref` (tolerance `rtol=1e-10`);
  - shape checks: `size(P) == (reduction.n_full, reduction.n_reduced)`,
    `size(S) == (reduction.n_reduced, reduction.n_full)`, `S` has exactly
    `n_reduced` nonzeros (one per row), `nnz(P)` equals the sum of block basis sizes.
- This proves `S·A·P` reproduces the existing reduction exactly — the distributed
  `MatMatMult` then computes the same product, distributed.

**Cluster (user, complex-scalar PETSc + MUMPS under mpirun):**
- `_mat_mat_mult`/`_reduce_dist` + the distributed onset/biglobal `:slepc` flow;
  `mpirun -n {1,2,4}` onset (and biglobal) leading eigenvalues match `:krylovkit`.

## Files

- `src/Stability/linear.jl` — add `_constraint_projection_matrices`.
- `src/Stability/solver.jl` — add `_SLEPC_CONSTRAINED_SOLVER` Ref + `_solve_constrained_slepc(op; …)` core hook (errors if extension absent).
- `src/solve.jl` — route `solve(::OnsetProblem/::BiglobalProblem; backend=:slepc)` to the
  distributed constrained path (keep `:krylovkit` untouched).
- `ext/MagratheaSlepcExt/raw_petsc.jl` — `_mat_mat_mult` ccall.
- `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl` — `_reduce_dist`, `_solve_constrained_slepc` impl,
  register the hook; reuse `_to_petsc_dist`, `_eps_solve_and_gather`.
- `test/distributed_reduction.jl` (new) — the serial S/P + equivalence tests; wired into
  `runtests.jl`.

## Scope / out of scope

- IN: `S·A·P` matrix-form reduction (core) + distributed `MatMatMult` reduce +
  onset/biglobal `:slepc` rewire. Replicated full-`A` assembly and replicated
  `P`-construction stay (Phase 4). 
- OUT: distributed onset/biglobal assembly (Phase 4), distributed `P`-construction,
  MHD Galerkin reduction, gather-to-all, non-MUMPS.

## Verification limitation

Only `_constraint_projection_matrices` + the `S·A·P == _constrained_reduced_matrices`
equivalence are verifiable here (pure Julia). The distributed `MatMatMult`
(`_mat_mat_mult`/`_reduce_dist`) and the onset/biglobal distributed `:slepc` flow are
cluster-only, validated by the `mpirun` eigenvalue-match run. This is the most
cluster-dependent phase so far.
