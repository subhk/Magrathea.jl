# Distributed MPI — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a PETSc-free `(field, ℓ, radial) ↔ global-row` mapping + PETSc row-ownership query over the existing `index_map`, so Phase 2 distributed assembly can insert only owned rows.

**Architecture:** One new file `src/Stability/dof_ownership.jl` with three pure-integer functions (`row_to_dof`, `dof_to_row`, `owned_block_ranges`) generic over `index_map::AbstractDict{K,UnitRange{Int}}`, included from `Stability.jl`. Fully serial; no PETSc/MPI.

**Tech Stack:** Julia 1.12, base only (no new deps).

**Spec:** `docs/superpowers/specs/2026-06-04-distributed-mpi-phase1-design.md`. Roadmap: `project_distributed_mpi_roadmap` memory.

---

## Environment / workflow notes

- Julia binary (launcher broken): `JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia`. Bash tool needs `dangerouslyDisableSandbox: true`. Julia 1.12.x.
- **Everything here is verifiable locally** (pure integer logic, no PETSc).
- **Commits:** standing rule = no `git commit` without explicit permission. Each task ends with a commit step — **pause and ask before running it.**
- Conventions: `index_map` ranges are 1-based Julia inclusive (`a:b`); PETSc `rstart/rend` are 0-based half-open `[rstart, rend)`.

---

## File Structure

- `src/Stability/dof_ownership.jl` (create) — the three functions. One responsibility: DOF↔row mapping + ownership filtering.
- `src/Stability/Stability.jl` (modify) — add `include("dof_ownership.jl")`.
- `test/dof_ownership.jl` (create) — serial unit tests.
- `test/runtests.jl` (modify) — `include("dof_ownership.jl")`.

---

## Task 1: `row_to_dof` + `dof_to_row`

**Files:**
- Create: `src/Stability/dof_ownership.jl`
- Modify: `src/Stability/Stability.jl`
- Create: `test/dof_ownership.jl`

- [ ] **Step 1: Create `test/dof_ownership.jl` with failing tests**

```julia
using Test
using Magrathea

@testset "row_to_dof / dof_to_row round-trip" begin
    im = Dict((1,:P)=>1:4, (2,:P)=>5:8, (1,:Θ)=>9:12)   # Nr=4, 3 blocks, 12 rows
    for grow in 1:12
        key, loc = Magrathea.row_to_dof(im, grow)
        @test Magrathea.dof_to_row(im, key, loc) == grow
    end
    @test Magrathea.row_to_dof(im, 6) == ((2,:P), 2)
    @test Magrathea.dof_to_row(im, (1,:Θ), 1) == 9
    @test_throws ErrorException Magrathea.row_to_dof(im, 0)
    @test_throws ErrorException Magrathea.row_to_dof(im, 13)
    @test_throws ErrorException Magrathea.dof_to_row(im, (9,:P), 1)   # unknown key
    @test_throws ErrorException Magrathea.dof_to_row(im, (1,:P), 5)   # bad local
end
```

- [ ] **Step 2: Create `src/Stability/dof_ownership.jl`**

```julia
# =============================================================================
#  DOF <-> global-row mapping and PETSc row-ownership queries.
#
#  Pure integer bookkeeping over an `index_map` (1-based Julia row ranges keyed by
#  (ℓ, field)). No PETSc/MPI — the foundation Phase 2+ distributed assembly uses to
#  insert only the rows a rank owns. Conventions: `index_map` ranges are 1-based
#  inclusive `a:b`; PETSc ownership ranges `[rstart, rend)` are 0-based half-open.
# =============================================================================

"""
    row_to_dof(index_map, grow) -> (key, local)

Map a 1-based global row `grow` to its block `key` and 1-based local radial index.
"""
function row_to_dof(index_map::AbstractDict{K,UnitRange{Int}}, grow::Int) where {K}
    for (key, rng) in index_map
        if grow in rng
            return key, grow - first(rng) + 1
        end
    end
    maxrow = isempty(index_map) ? 0 : maximum(last, values(index_map))
    error("row $grow is outside the DOF layout (valid rows 1:$maxrow)")
end

"""
    dof_to_row(index_map, key, local) -> Int

Inverse of `row_to_dof`: 1-based global row for block `key`, 1-based local index.
"""
function dof_to_row(index_map::AbstractDict{K,UnitRange{Int}}, key::K, local::Int) where {K}
    haskey(index_map, key) || error("unknown DOF block key $key")
    rng = index_map[key]
    (1 <= local <= length(rng)) ||
        error("local index $local outside 1:$(length(rng)) for block $key")
    return first(rng) + local - 1
end
```

- [ ] **Step 3: Wire the include** — in `src/Stability/Stability.jl`, add after `include("solver.jl")`:

```julia
include("dof_ownership.jl")
```

- [ ] **Step 4: Run, verify the round-trip testset passes** — `$JL --project=. -e 'using Test; @testset "t" begin include("test/dof_ownership.jl") end'` → PASS (one testset). (If the file was created test-first and run before Step 2/3, it FAILS with undefined `row_to_dof`; after Step 2/3 it PASSES.)

- [ ] **Step 5: Commit (ASK USER FIRST)** — `git add src/Stability/dof_ownership.jl src/Stability/Stability.jl test/dof_ownership.jl` / `git commit -m "feat(mpi): add DOF<->row mapping (row_to_dof, dof_to_row)"`

---

## Task 2: `owned_block_ranges`

**Files:**
- Modify: `src/Stability/dof_ownership.jl`
- Modify: `test/dof_ownership.jl`

- [ ] **Step 1: Append failing tests to `test/dof_ownership.jl`**

```julia
@testset "owned_block_ranges" begin
    im = Dict((1,:P)=>1:4, (2,:P)=>5:8, (1,:Θ)=>9:12)
    @test Magrathea.owned_block_ranges(im, 0, 12) ==
          [((1,:P),1:4), ((2,:P),1:4), ((1,:Θ),1:4)]          # full range, sorted by start
    @test Magrathea.owned_block_ranges(im, 0, 6) ==
          [((1,:P),1:4), ((2,:P),1:2)]                        # split block 2 mid-way
    @test Magrathea.owned_block_ranges(im, 6, 12) ==
          [((2,:P),3:4), ((1,:Θ),1:4)]
    @test Magrathea.owned_block_ranges(im, 4, 8) == [((2,:P),1:4)]  # subset of blocks
    @test Magrathea.owned_block_ranges(im, 4, 4) == []              # empty owned range
end
```

- [ ] **Step 2: Run, verify FAIL** — `$JL --project=. -e 'using Test; @testset "t" begin include("test/dof_ownership.jl") end'` → FAIL (`owned_block_ranges` undefined).

- [ ] **Step 3: Append `owned_block_ranges` to `src/Stability/dof_ownership.jl`**

```julia
"""
    owned_block_ranges(index_map, rstart, rend) -> Vector{Tuple{K, UnitRange{Int}}}

Blocks whose rows intersect the PETSc ownership range `[rstart, rend)` (0-based,
half-open). For each, returns `(key, owned_local_range)` — the 1-based local radial
indices this rank owns (partial blocks return only the owned slice). Sorted by block
start row for determinism (`index_map` is an unordered Dict).
"""
function owned_block_ranges(index_map::AbstractDict{K,UnitRange{Int}},
                            rstart::Int, rend::Int) where {K}
    out = Tuple{K,UnitRange{Int}}[]
    for (key, rng) in index_map
        a = first(rng)                      # 1-based inclusive
        b = last(rng)
        lo = max(a - 1, rstart)             # owned PETSc rows [lo, hi)
        hi = min(b, rend)
        if lo < hi
            push!(out, (key, (lo - a + 2):(hi - a + 1)))   # back to 1-based local
        end
    end
    sort!(out; by = t -> first(index_map[t[1]]))
    return out
end
```

- [ ] **Step 4: Run, verify PASS** — same command as Step 2 → both testsets PASS.

- [ ] **Step 5: Commit (ASK USER FIRST)** — `git add src/Stability/dof_ownership.jl test/dof_ownership.jl` / `git commit -m "feat(mpi): add owned_block_ranges PETSc-ownership query"`

---

## Task 3: Wire into the suite + full regression

**Files:**
- Modify: `test/runtests.jl`

- [ ] **Step 1: Add to `test/runtests.jl`** — after the existing test includes (near `include("slepc_backend.jl")`):

```julia
include("dof_ownership.jl")
```

- [ ] **Step 2: Full suite green** — `$JL --project=. test/runtests.jl 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done`. Confirm the `dof_ownership` testsets appear in the run (grep the log for `owned_block_ranges`).

- [ ] **Step 3: Commit (ASK USER FIRST)** — `git add test/runtests.jl` / `git commit -m "test(mpi): wire dof_ownership tests into the suite"`

---

## Self-review notes

- **Spec coverage:** `row_to_dof` (T1), `dof_to_row` (T1), `owned_block_ranges` with partial-block + determinism (T2), error handling (T1 error tests + T2 empty-range), suite wiring (T3). All spec sections mapped. The owned-local formula `(lo-a+2):(hi-a+1)` matches the spec and every test expectation (block `5:8` ∩ `[0,6)` → `1:2`; ∩ `[6,12)` → `3:4`; `[4,4)` → omitted).
- **Placeholder scan:** none — every function body and test is complete code.
- **Type consistency:** all three functions generic over `K` with `AbstractDict{K,UnitRange{Int}}`; names `row_to_dof`/`dof_to_row`/`owned_block_ranges` consistent across tasks and the test calls.
- **Verification:** fully local (pure integer logic); no PETSc/MPI/cluster step.
