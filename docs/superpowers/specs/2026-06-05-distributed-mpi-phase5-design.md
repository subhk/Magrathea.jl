# Distributed MPI — Phase 5 design (distributed triglobal)

Date: 2026-06-05
Status: approved (design); implementation pending
Program: see [[project_distributed_mpi_roadmap]] — Phase 5 of 6 (final phase).

## Goal

Distribute the triglobal coupled-pencil assembly: each rank inserts only its owned rows
of the large `n_total × n_total` coupled pencil (the memory win), feeding the Phase-0
distributed solve. The per-m single-mode operators and the off-diagonal
`build_mode_coupling_operators` stay **replicated** inputs, because the coupling
construction needs the single-mode operators for *both* endpoints of every coupling
pair — so per-m operator distribution is coupled to coupling distribution, and both are
**deferred** until after the pending Vector-SH rewrite (see
[[project_vector_sh_rewrite_plan]]) replaces the coupling code (distributing
soon-to-be-rewritten code would be wasted). A single-m on-demand builder
(`build_single_mode_operator`) is added and serial-tested now as the seam for that
future per-m distribution.

## Context

`assemble_block_matrices(problem, single_mode_ops, coupling_ops, verbose)`
(src/Stability/triglobal.jl ~1829) already builds the coupled pencil via COO:
- diagonal blocks: `single_mode_ops[m].A` / `.B` at global range `problem.block_indices[m]`
  (contiguous per-m ranges; `problem.total_dofs` = `n_total`);
- off-diagonal: `coupling_ops[(m_from, m_to)]` at `(block_indices[m_to], block_indices[m_from])`;
- via `_append_block_entries!(row,col,val, block, rows::UnitRange, cols::UnitRange, tol)`
  which pushes `block[i,j]` (>tol) at global `(rows[i], cols[j])`.

`single_mode_ops` are built by `build_single_mode_operators(problem)` (per-m reduced
`SingleModeOperator`s). The triglobal `:slepc` path currently assembles the full
replicated pencil → `solve_block_eigenvalue_problem(...; backend=:slepc)` → Phase-0
replicated `_to_petsc_dist` solve. The coupled-pencil eigenvectors ARE the triglobal
result (per-m reduced coefficients); no per-m full reconstruction is needed for the
eigensolve.

## Architecture

### Core (PETSc-free, serial-verifiable)

1. **Owned filter on `_append_block_entries!`** — add an `owned::Union{Nothing,UnitRange{Int}}=nothing`
   kwarg; push a triplet only when `global_i ∈ owned` (or `owned===nothing`).
2. **`_assemble_block_coo(problem, single_mode_ops, coupling_ops; owned_julia_rows=nothing)`**
   — the COO body of `assemble_block_matrices`, returning a NamedTuple
   `(A_rows, A_cols, A_vals, B_rows, B_cols, B_vals, n)`. With `owned_julia_rows = R`:
   skip a diagonal block for `m` if `block_indices[m] ∩ R` is empty; skip a coupling
   block `(m_from,m_to)` if `block_indices[m_to] ∩ R` is empty; pass `owned=R` to
   `_append_block_entries!` for boundary blocks split across ranks.
3. **`assemble_block_matrices`** becomes a thin wrapper: `c = _assemble_block_coo(problem, single_mode_ops, coupling_ops)`; `sparse(c.A_rows,…,n_total,n_total)` for A and B; return `(A_coupled, B_coupled)`. Serial byte-identical (COO unchanged for `owned=nothing`).
4. **`build_single_mode_operator(problem, m)`** — extract the per-m body of
   `build_single_mode_operators` into a single-m builder, so the distributed path can
   build only the m-modes a rank touches. `build_single_mode_operators(problem)` becomes
   a loop calling it (or keep both; the singular one is the new public-internal entry).

### Extension (PETSc/MPI, cluster-only)

- **`_SLEPC_TRIGLOBAL_SOLVER` hook** (core Ref + `_solve_triglobal_slepc(problem; …)` that errors if extension absent), registered by the extension.
- **`_slepc_triglobal_solve(problem; nev, sigma, which, tol, maxiter)`**:
  1. Build `single_mode_ops = Magrathea.build_single_mode_operators(problem)` and
     `coupling_ops = Magrathea.build_mode_coupling_operators(problem, single_mode_ops, false)`
     — **replicated** (built once per rank; per-m ops are small relative to the coupled
     pencil, and coupling needs both endpoints).
  2. `n_total = problem.total_dofs`; `Amat, rs, re = _create_dist_mat(n_total)`,
     `Bmat,_,_ = _create_dist_mat(n_total)`; `R = (rs+1):re`.
  3. `coo = Magrathea._assemble_block_coo(problem, single_mode_ops, coupling_ops; owned_julia_rows=R)`;
     `_fill_dist_mat!(Amat, coo.A_rows, coo.A_cols, coo.A_vals, rs, re)`, same for `Bmat`
     — this is the distributed part: each rank inserts only its owned pencil rows; the
     full `n_total × n_total` pencil is never materialized.
  4. `_eps_solve_and_gather(Amat, Bmat, n_total; nev, sigma=Complex{T}(σ_target, 1e-6),
     which, selection=:maxreal, tol, maxiter)` (triglobal shift convention); rank-0
     eigenvectors are the result. Return `(vals, vecs_rank0, info)`.
- Rewire the triglobal `:slepc` path (currently in `solve_block_eigenvalue_problem`'s
  `:slepc` divert) to route through `_solve_triglobal_slepc` when `backend===:slepc`,
  replacing the "replicated pencil → `_to_petsc_dist`" path. Adapt the 2-tuple/3-tuple
  return to what `solve_triglobal_eigenvalue_problem` expects.

> Scope honesty: Phase 5's distributed path keeps `single_mode_ops` and `coupling_ops`
> **replicated** (built once per rank) and distributes the **coupled-pencil assembly**
> (owned-row insertion — the memory win on the large `n_total × n_total` pencil).
> `build_single_mode_operator` is added + serial-tested as the seam for a later per-m
> operator distribution, but is NOT yet wired into the distributed path. Full per-m and
> coupling distribution is future work tied to the Vector-SH rewrite.

## Data flow

`solve(::TriglobalProblem; backend=:slepc)` → `_solve_triglobal_slepc(problem)`:
replicated `single_mode_ops`/`coupling_ops`; distributed owned-row pencil assembly;
distributed EPS solve; rank-0 eigenvectors → `StabilityResult`.

## Error handling

- Inherits Phase-0 SLEPc/PETSc errors. `_assemble_block_coo` with `owned_julia_rows`
  outside `1:n_total` → empty contributions.

## Testing

**Serial, here:**
- **Partition-reassemble:** small coupled triglobal problem (reuse the
  `_basic_state_3d_with_modes` fixture from `test/triglobal.jl`, `m_range=1:2`); build
  `single_mode_ops`, `coupling_ops`; `full = _assemble_block_coo(problem, single, coupling)`;
  `A_full = sparse(full.A_rows,…,n,n)`; partition `1:n_total` into chunks; concatenate
  each chunk's owned COO; `sparse(...)` → assert `== A_full` (and B). Plus owned-row
  containment.
- **`assemble_block_matrices` wrapper regression:** equals the pre-refactor sparse
  output — covered by `test/triglobal.jl` staying green (it calls
  `build_mode_coupling_operators`/the block assembly).
- **On-demand builder match:** `build_single_mode_operator(problem, m).A ≈ build_single_mode_operators(problem)[m].A` (and `.B`) for each m in `m_range`.

**Cluster (user, complex-scalar PETSc + MUMPS under mpirun):**
- distributed triglobal `:slepc`; `mpirun -n {1,2,4}` leading eigenvalues match the
  serial `:krylovkit` triglobal solve.

## Files

- `src/Stability/triglobal.jl` — owned filter on `_append_block_entries!`;
  `_assemble_block_coo`; `assemble_block_matrices` wrapper; `build_single_mode_operator`;
  `_SLEPC_TRIGLOBAL_SOLVER` is declared in `src/Stability/solver.jl` with `_solve_triglobal_slepc`.
- `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl` — `_slepc_triglobal_solve` + register hook.
- `src/Stability/triglobal.jl` (solve path) — rewire `:slepc` to the hook.
- `test/distributed_triglobal.jl` (new) — serial tests; wired into `runtests.jl`.

## Scope / out of scope

- IN: distributed coupled-pencil assembly (owned-row COO + insert), `build_single_mode_operator` seam, triglobal `:slepc` rewire to distributed pencil insertion.
- OUT: distributing `build_mode_coupling_operators` (deferred — Vector-SH rewrite),
  fully distributing per-m operator construction (seam added, not wired as sole path),
  gather-to-all, non-MUMPS.

## Verification limitation

The COO pencil assembly (partition-reassemble), the wrapper regression, and the
on-demand builder match are **verified here** (pure Julia). The distributed pencil
insertion into PETSc and the distributed solve are cluster-only, validated by the
`mpirun` eigenvalue-match run.
