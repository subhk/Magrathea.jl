# Distributed MPI — Phase 1 design (DOF ↔ row-ownership mapping)

Date: 2026-06-04
Status: approved (design); implementation pending
Program: see [[project_distributed_mpi_roadmap]] — Phase 1 of 6.

## Goal

A PETSc-free, serial-testable abstraction that maps between Magrathea.jl's
`(field, ℓ, radial-index)` degrees of freedom and global matrix row indices, and
answers ownership queries for a PETSc row range `[rstart, rend)`. This is the
bookkeeping layer that Phase 2+ distributed assembly uses to insert only the rows
a rank owns. No assembly changes and no PETSc/MPI in this phase.

## Context

Every Magrathea.jl problem lays matrix rows out as **contiguous `Nr`-row blocks keyed by
`(ℓ, field)`**:
- `LinearStabilityOperator` and triglobal `SingleModeOperator` expose this directly
  as `index_map::Dict{Tuple{Int,Symbol}, UnitRange{Int}}` (1-based Julia row ranges,
  `idx:(idx+Nr-1)` per `(ℓ, field)`, fields ordered P, T, Θ).
- MHD assembly uses section offsets (`u/v/f/g/h` × `n_per_mode`) that reduce to the
  same "contiguous range keyed by (mode, field)" shape; an `index_map` of that shape
  can be constructed for it in Phase 2 if needed.

So a global row maps to exactly one `(key, local_radial_index)`. This phase
formalizes that mapping plus ownership filtering over the existing `index_map`.

## Architecture

New file `src/Stability/dof_ownership.jl`, `include`d from `src/Stability/Stability.jl`.
Pure integer logic over a generic `index_map::AbstractDict{K, UnitRange{Int}}`
(K = `Tuple{Int,Symbol}` in practice, but the functions stay generic in K). No
dependency on PETSc, MPI, or any operator struct — it takes the `index_map` itself.

### Components

```julia
"""Map a 1-based global row to its block key and 1-based local radial index."""
function row_to_dof(index_map::AbstractDict{K,UnitRange{Int}}, grow::Int) where {K}
    # returns (key::K, local::Int); errors if grow ∉ any block
end

"""Inverse: block key + 1-based local radial index → 1-based global row."""
function dof_to_row(index_map::AbstractDict{K,UnitRange{Int}}, key::K, local::Int) where {K}
    # returns Int; errors on unknown key or local ∉ 1:length(range)
end

"""Owned blocks for a PETSc ownership range `[rstart, rend)` (0-based, half-open).
Returns, per block whose rows intersect the range, `(key, owned_local_range)` where
`owned_local_range` is the 1-based local radial indices this rank owns. Partial
blocks (a block's Nr rows split across ranks) return only the owned slice. Output is
sorted by block start row (deterministic; `index_map` is an unordered Dict)."""
function owned_block_ranges(index_map::AbstractDict{K,UnitRange{Int}},
                            rstart::Int, rend::Int) where {K}
    # returns Vector{Tuple{K, UnitRange{Int}}}
end
```

### Convention handling

`index_map` ranges are 1-based Julia inclusive (`a:b`). PETSc `rstart/rend` are
0-based, half-open. Internally a block `a:b` corresponds to PETSc rows `[a-1, b)`.
For `owned_block_ranges`: owned PETSc rows = `[max(a-1, rstart), min(b, rend))`; if
that is nonempty (`lo < hi`), the owned 1-based local range is
`(lo - (a-1) + 1) : (hi - 1 - (a-1) + 1)` = `(lo - a + 2) : (hi - a + 1)`.

## Data flow

Phase 2 assembly (a later phase) will: get its PETSc row ownership
(`MatGetOwnershipRange` → `rstart, rend`), call
`owned_block_ranges(op.index_map, rstart, rend)`, and for each `(key, local_range)`
compute/insert only those rows. `row_to_dof`/`dof_to_row` support diagnostics and
any row-level logic later phases need. This phase ships only the mapping.

## Error handling

- `row_to_dof`: `grow < 1` or `grow` beyond the largest block end → `error` naming
  the out-of-range row and the valid maximum.
- `dof_to_row`: `key` not in `index_map` → `error`; `local ∉ 1:length(range)` →
  `error` naming the valid local range.
- `owned_block_ranges`: `rstart`/`rend` are taken as given (PETSc-provided); a block
  fully outside the range is simply omitted; `rstart == rend` (empty owned range, a
  rank with no rows) → returns an empty vector.

## Testing (all serial, runs in this environment)

Unit tests in `test/dof_ownership.jl` (new), wired into `test/runtests.jl`:
- Build a hand layout, e.g.
  `index_map = Dict((1,:P)=>1:4, (2,:P)=>5:8, (1,:Θ)=>9:12)` (Nr=4, 3 blocks, 12 rows).
- **Round-trip:** for every `grow in 1:12`, `dof_to_row(im, row_to_dof(im, grow)...) == grow`.
- **Spot checks:** `row_to_dof(im, 6) == ((2,:P), 2)`; `dof_to_row(im, (1,:Θ), 1) == 9`.
- **Full range:** `owned_block_ranges(im, 0, 12)` → all 3 blocks, each `1:4`, sorted by start row.
- **Partial/split block:** `owned_block_ranges(im, 0, 6)` → `[((1,:P),1:4), ((2,:P),1:2)]`
  (second block split mid-way); `owned_block_ranges(im, 6, 12)` →
  `[((2,:P),3:4), ((1,:Θ),1:4)]`.
- **Subset of blocks:** `owned_block_ranges(im, 4, 8)` → `[((2,:P),1:4)]`.
- **Empty owned range:** `owned_block_ranges(im, 4, 4)` → `[]`.
- **Determinism:** output sorted by block start row regardless of Dict iteration order.
- **Errors:** `row_to_dof(im, 0)` / `row_to_dof(im, 13)` throw; `dof_to_row(im, (9,:P), 1)`
  (unknown key) and `dof_to_row(im, (1,:P), 5)` (bad local) throw.

## Scope / out of scope

- IN: the three mapping/query functions over a generic `index_map`, error handling,
  full serial unit tests, wiring into the suite.
- OUT: any distributed assembly (Phase 2), constructing an MHD `index_map` (deferred
  to when Phase 2 needs it), PETSc/MPI, performance tuning (these maps are tiny).

## Verification

Fully verifiable in this environment — pure integer logic, no PETSc. The unit tests
above are the complete verification; no cluster validation needed for this phase.
