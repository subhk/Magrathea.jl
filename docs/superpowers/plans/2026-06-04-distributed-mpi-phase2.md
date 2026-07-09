# Distributed MPI — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distribute the MHD tau matrix assembly — each MPI rank computes/inserts only its owned rows (interior via owned-COO with block-skipping; tau BCs overwritten on the distributed PETSc Mat), no full Julia matrix per rank.

**Architecture:** Core gains `_mhd_index_map`, `_owned_coo_nnz` (pure), and `_assemble_mhd_coo(op; owned_julia_rows)` (the extracted pre-BC interior COO body, with per-mode block-skip + owned-row filter); `assemble_mhd_matrices` becomes a thin wrapper (serial byte-identical). The extension creates the distributed Mat, gets ownership, calls back for owned COO, inserts, then overwrites owned BC rows.

**Tech Stack:** Julia 1.12, PetscWrap/SlepcWrap/MPI (cluster), MUMPS. Raw `MatZeroRows` ccall if unwrapped.

**Spec:** `docs/superpowers/specs/2026-06-04-distributed-mpi-phase2-design.md`. Roadmap: `project_distributed_mpi_roadmap` memory.

---

## Environment / workflow notes

- Julia binary: `JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia`. Bash needs `dangerouslyDisableSandbox: true`. Julia 1.12.x.
- **T1 + T2 fully verifiable here** (pure Julia). **T3 + T4 cluster-only** (no PETSc/MPI) → parse + symbol-audit + `CORE_OK` locally; runtime is the user's `mpirun` job. The distributed BC port (`_apply_dist_bcs!`) is the largest blind piece.
- **Commits:** no `git commit` without explicit permission — each commit step pauses for the user.
- Convention: owned PETSc `[rstart,rend)` (0-based half-open) ↔ Julia rows `(rstart+1):rend`.

---

## File Structure

- `src/MHD/assembly.jl` (modify) — add `_mhd_index_map`, `_owned_coo_nnz`; extract `_assemble_mhd_coo`; make `assemble_mhd_matrices` a wrapper.
- `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl` (modify) — `_create_dist_mat`, `_fill_dist_mat!`, `_apply_dist_bcs!`; rewire MHD `_slepc_solve`.
- `test/distributed_assembly.jl` (create) — serial tests; wired into `runtests.jl`.

---

## Task 1: Core pure helpers — `_mhd_index_map` + `_owned_coo_nnz`

**Files:** Modify `src/MHD/assembly.jl`; create/extend `test/distributed_assembly.jl`.

- [ ] **Step 1: Create `test/distributed_assembly.jl` with failing tests**

```julia
using Test
using SparseArrays
using Magrathea

@testset "_mhd_index_map tiles rows by section" begin
    params = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1.0, ricb=0.35,
                       m=1, lmax=3, N=8, B0_type=dipole, B0_amplitude=1.0)
    op = MHDStabilityOperator(params)
    im = Magrathea._mhd_index_map(op)
    n_per_mode = params.N + 1
    # every block is n_per_mode rows, ranges tile 1:matrix_size contiguously
    @test all(length(r) == n_per_mode for r in values(im))
    sorted = sort(collect(values(im)); by=first)
    @test first(sorted[1]) == 1
    for i in 2:length(sorted)
        @test first(sorted[i]) == last(sorted[i-1]) + 1   # contiguous, no gaps/overlap
    end
    @test last(sorted[end]) == op.matrix_size
    # round-trips against the Phase-1 mapping
    key, loc = Magrathea.row_to_dof(im, n_per_mode + 1)
    @test Magrathea.dof_to_row(im, key, loc) == n_per_mode + 1
end

@testset "_owned_coo_nnz counts owned rows by band" begin
    # rows/cols are 1-based Julia COO; ownership band is 0-based [rstart,rend)
    rows = [1, 1, 2, 3, 3, 4]
    cols = [1, 3, 2, 1, 4, 4]
    d, o = Magrathea._owned_coo_nnz(rows, cols, 0, 2)   # own rows 1,2; band cols [0,2)→cols 1,2
    @test d == [1, 1] && o == [1, 0]
    d2, o2 = Magrathea._owned_coo_nnz(rows, cols, 2, 4) # own rows 3,4; band cols 3,4
    @test d2 == [1, 1] && o2 == [1, 0]
    d3, o3 = Magrathea._owned_coo_nnz(rows, cols, 0, 4)
    @test d3 == [2,1,2,1] && o3 == [0,0,0,0]
end
```

- [ ] **Step 2: Run, verify FAIL** — `$JL --project=. -e 'using Test; @testset "t" begin include("test/distributed_assembly.jl") end'` → FAIL (`_mhd_index_map`/`_owned_coo_nnz` undefined).

- [ ] **Step 3: Add both helpers to `src/MHD/assembly.jl`** (top level, before `assemble_mhd_matrices`)

```julia
"""Row layout of the MHD tau matrix as a Phase-1 index_map: keys `(ℓ, section)` for
sections `:u,:v,:f,:g,:h` in order, each a contiguous `(N+1)`-row range."""
function _mhd_index_map(op::MHDStabilityOperator)
    n_per_mode = op.params.N + 1
    im = Dict{Tuple{Int,Symbol}, UnitRange{Int}}()
    off = 0
    for (sec, ls) in ((:u, op.ll_u), (:v, op.ll_v), (:f, op.ll_f), (:g, op.ll_g), (:h, op.ll_h))
        for l in ls
            im[(l, sec)] = (off + 1):(off + n_per_mode)
            off += n_per_mode
        end
    end
    return im
end

"""Per-owned-row diagonal/off-diagonal nnz from COO triplets, for the owned PETSc row
block `[rstart, rend)` (0-based, half-open), diagonal band = same column range. `rows`
and `cols` are 1-based Julia indices. Returns `(d_nnz, o_nnz)`, length `rend-rstart`."""
function _owned_coo_nnz(rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer},
                        rstart::Int, rend::Int)
    nloc = rend - rstart
    d = zeros(Int, nloc); o = zeros(Int, nloc)
    @inbounds for k in eachindex(rows)
        r0 = rows[k] - 1
        if rstart <= r0 < rend
            c0 = cols[k] - 1
            i = r0 - rstart + 1
            (rstart <= c0 < rend) ? (d[i] += 1) : (o[i] += 1)
        end
    end
    return d, o
end
```

- [ ] **Step 4: Run, verify PASS** — same command as Step 2 → both testsets PASS.

- [ ] **Step 5: Commit (ASK USER FIRST)** — `git add src/MHD/assembly.jl test/distributed_assembly.jl` / `git commit -m "feat(mpi): MHD index_map + owned-COO preallocation helper"`

---

## Task 2: Extract `_assemble_mhd_coo` (owned-filtered interior) + thin wrapper

**Files:** Modify `src/MHD/assembly.jl`; extend `test/distributed_assembly.jl`.

This refactors the existing `assemble_mhd_matrices` body. Today (line numbers approximate):
- lines ~246–253: declares `A_rows/A_cols/A_vals/B_rows/B_cols/B_vals`.
- lines ~255–261: `add_block!` closure.
- lines ~268–547: SECTION U/V/F/G/H, each a `for (k,l) in enumerate(op.ll_X)` loop with `row_base = …` and many `add_block!` calls.
- lines ~548–549: `A = sparse(...); B = sparse(...)`.
- lines ~559–596: `apply_*_boundary_conditions!`, `interior_dofs`, `info`, `return`.

- [ ] **Step 1: Append the failing partition-reassemble test to `test/distributed_assembly.jl`**

```julia
@testset "distributed interior COO partition-reassembles to full pre-BC matrix" begin
    params = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1.0, ricb=0.35,
                       m=1, lmax=3, N=8, B0_type=dipole, B0_amplitude=1.0)
    op = MHDStabilityOperator(params)
    full = Magrathea._assemble_mhd_coo(op)                       # owned=nothing → full interior
    n = full.n
    A_pre = sparse(full.A_rows, full.A_cols, full.A_vals, n, n)
    B_pre = sparse(full.B_rows, full.B_cols, full.B_vals, n, n)

    # arbitrary contiguous partition of 1:n into 3 chunks
    cuts = [0, fld(n,3), fld(2n,3), n]
    Ar=Int[]; Ac=Int[]; Av=ComplexF64[]; Br=Int[]; Bc=Int[]; Bv=ComplexF64[]
    for i in 1:3
        R = (cuts[i]+1):cuts[i+1]
        c = Magrathea._assemble_mhd_coo(op; owned_julia_rows=R)
        @test all(r -> r in R, c.A_rows)                     # only owned rows emitted
        @test all(r -> r in R, c.B_rows)
        append!(Ar,c.A_rows); append!(Ac,c.A_cols); append!(Av,c.A_vals)
        append!(Br,c.B_rows); append!(Bc,c.B_cols); append!(Bv,c.B_vals)
    end
    @test sparse(Ar,Ac,Av,n,n) == A_pre                      # reassembles exactly
    @test sparse(Br,Bc,Bv,n,n) == B_pre
end
```

- [ ] **Step 2: Run, verify FAIL** — `_assemble_mhd_coo` undefined.

- [ ] **Step 3: Refactor `assemble_mhd_matrices` into `_assemble_mhd_coo` + wrapper.**
  1. Rename the current `function assemble_mhd_matrices(op::MHDStabilityOperator{T}) where {T}` to
     `function _assemble_mhd_coo(op::MHDStabilityOperator{T}; owned_julia_rows::Union{Nothing,UnitRange{Int}}=nothing) where {T}`.
  2. Change its `add_block!` closure to filter on owned rows:
     ```julia
     function add_block!(rows, cols, vals, block, row_offset, col_offset)
         Is, Js, Vs = findnz(block)
         @inbounds for k in eachindex(Vs)
             grow = Is[k] + row_offset
             if owned_julia_rows === nothing || grow in owned_julia_rows
                 push!(rows, grow); push!(cols, Js[k] + col_offset); push!(vals, Vs[k])
             end
         end
         return nothing
     end
     ```
  3. At the **top of each of the five per-mode loops**, right after the `row_base = …`
     line (lines ~273, ~353, ~417, ~471, ~521), insert the block-skip guard:
     ```julia
     if owned_julia_rows !== nothing &&
        isempty(intersect((row_base+1):(row_base+n_per_mode), owned_julia_rows))
         continue
     end
     ```
  4. Replace everything from `A = sparse(...)` (line ~548) to the end of the function
     with a NamedTuple return of the raw triplets + metadata (NO sparse, NO BCs):
     ```julia
         return (A_rows=A_rows, A_cols=A_cols, A_vals=A_vals,
                 B_rows=B_rows, B_cols=B_cols, B_vals=B_vals,
                 n=n, interior_dofs=Int[], info=Dict{String,Any}())
     end
     ```
     (Drop the old `interior_dofs`/`info`/BC lines from this function — they move to the
     wrapper. `interior_dofs` is recomputed in the wrapper from the assembled `B`.)
  5. Add the thin wrapper that reproduces today's full behavior:
     ```julia
     """Assemble the MHD tau (A, B) sparse matrices with boundary conditions applied."""
     function assemble_mhd_matrices(op::MHDStabilityOperator{T}) where {T}
         c = _assemble_mhd_coo(op)
         A = sparse(c.A_rows, c.A_cols, c.A_vals, c.n, c.n)
         B = sparse(c.B_rows, c.B_cols, c.B_vals, c.n, c.n)
         apply_velocity_boundary_conditions!(A, B, op)
         params = op.params
         if params.Le != 0
             apply_magnetic_boundary_conditions!(A, B, op, :f)
             apply_magnetic_boundary_conditions!(A, B, op, :g)
         end
         apply_temperature_boundary_conditions!(A, B, op)
         B_diag = diag(B)
         interior_dofs = findall(i -> abs(B_diag[i]) > 1e-14, 1:c.n)
         info = Dict("method" => "MHD tau", "size" => c.n)
         return A, B, interior_dofs, info
     end
     ```
     IMPORTANT: read the ORIGINAL lines 548–596 first and preserve the exact BC-call
     conditions (the `params.Le != 0` / magnetic-BC guards) and the exact `interior_dofs`
     / `info` construction — copy them verbatim into the wrapper rather than the
     approximation above if they differ. The wrapper's `(A,B,interior_dofs,info)` must
     match the pre-refactor return exactly.

- [ ] **Step 4: Run the partition test, verify PASS** — Step 1 command → PASS.

- [ ] **Step 5: Serial regression — MHD suite unchanged** — `$JL --project=. -e 'using Test; include("test/mhd_boundary_conditions.jl"); include("test/mhd_galerkin.jl"); include("test/mhd_stress_free_bc.jl")' 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done`.

- [ ] **Step 6: Commit (ASK USER FIRST)** — `git add src/MHD/assembly.jl test/distributed_assembly.jl` / `git commit -m "refactor(mhd): extract _assemble_mhd_coo with owned-row distributed assembly"`

---

## Task 3: Extension — distributed Mat build, fill, BCs, rewire (NOT runnable here)

**Files:** Modify `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl` (and `raw_petsc.jl` if `MatZeroRows` needs a raw ccall).

> Cannot run here. Verify: `Meta.parseall`, symbol-audit vs installed PetscWrap 0.1.5 / SlepcWrap 0.1.3, `CORE_OK`. Follow Phase-0 ccall conventions.

- [ ] **Step 1: Add `_create_dist_mat` + `_fill_dist_mat!` to `MagratheaSlepcExt.jl`**

```julia
function _create_dist_mat(n::Int)
    mat = MatCreate(MPI.COMM_WORLD)
    MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, n, n)
    MatSetFromOptions(mat)
    rstart, rend = MatGetOwnershipRange(mat)
    return mat, Int(rstart), Int(rend)
end

function _fill_dist_mat!(mat, rows, cols, vals, rstart::Int, rend::Int)
    PI = PetscWrap.PetscInt
    d, o = Magrathea._owned_coo_nnz(rows, cols, rstart, rend)
    MatMPIAIJSetPreallocation(mat, PI(0), PI.(d), PI(0), PI.(o))
    @inbounds for k in eachindex(rows)
        r0 = rows[k] - 1
        if rstart <= r0 < rend
            MatSetValue(mat, r0, cols[k] - 1, PetscScalar(vals[k]), INSERT_VALUES)
        end
    end
    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY)
    return mat
end
```

- [ ] **Step 2: Implement `_apply_dist_bcs!`** — port the serial `apply_velocity_boundary_conditions!`, `apply_magnetic_boundary_conditions!` (`:f`,`:g` when `Le != 0`), `apply_temperature_boundary_conditions!` (in `src/MHD/assembly.jl`, lines ~604–740) to the distributed Mat. READ those serial functions first; replicate the SAME tau-row indices and the SAME BC-equation values, but for each BC row only when owned, do `MatZeroRows` then `MatSetValues` the BC entries:

```julia
# Sketch — the per-BC-row logic must mirror the serial functions exactly.
function _apply_dist_bcs!(Amat, Bmat, op, rstart::Int, rend::Int)
    # For each tau/BC row `grow` (1-based) the serial code overwrites:
    #   owned = rstart <= grow-1 < rend
    #   owned && _zero_and_set_row!(Amat, grow, bc_cols_A, bc_vals_A)
    #   owned && _zero_and_set_row!(Bmat, grow, bc_cols_B, bc_vals_B)  # usually B row = 0
    # Build the (cols, vals) for each BC row from the serial formulas
    # (chebyshev boundary value/derivative rows, etc.).
    ...
end
```
Provide `_zero_and_set_row!(mat, grow, cols, vals)` that does `MatZeroRows(mat, [grow-1], 0.0)` (collective — see note) then `MatSetValues` the row's BC entries (0-based), and a final `MatAssemblyBegin/End`. If `MatZeroRows` is not wrapped in PetscWrap 0.1.5, add a raw ccall in `raw_petsc.jl`:
```julia
function _mat_zero_rows(mat, grows::Vector{Int})   # grows are 0-based
    PI = PetscWrap.PetscInt
    idx = PI.(grows)
    @assert iszero(ccall((:MatZeroRows, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
        (PetscWrap.CMat, PI, Ptr{PI}, PetscWrap.PetscScalar, PetscWrap.CVec, PetscWrap.CVec),
        mat, PI(length(idx)), idx, PetscScalar(0), C_NULL, C_NULL))
    return nothing
end
```
NOTE: `MatZeroRows` is collective — every rank must call it with its owned BC-row list (possibly empty). Structure `_apply_dist_bcs!` so all ranks reach the `MatZeroRows`/assembly calls the same number of times. This is the most intricate, cluster-only part — symbol-audit the `MatZeroRows` signature against the installed PETSc and the serial BC formulas carefully.

- [ ] **Step 3: Rewire the MHD branch of `_slepc_solve`.** Currently the MHD tau matrices reach `_slepc_solve` as already-assembled `SparseMatrixCSC` (from `assemble_mhd_matrices`) and go through Phase-0 `_to_petsc_dist`. Add an MHD-origin distributed path: when solving an MHD problem, instead build the operators distributed:
```julia
    n = op.matrix_size
    Amat, rstart, rend = _create_dist_mat(n)
    Bmat, _, _ = _create_dist_mat(n)
    coo = Magrathea._assemble_mhd_coo(op; owned_julia_rows=(rstart+1):rend)
    _fill_dist_mat!(Amat, coo.A_rows, coo.A_cols, coo.A_vals, rstart, rend)
    _fill_dist_mat!(Bmat, coo.B_rows, coo.B_cols, coo.B_vals, rstart, rend)
    _apply_dist_bcs!(Amat, Bmat, op, rstart, rend)
    # ... then EPSCreate/EPSSetOperators(eps, Amat, Bmat)/solve/gather as in Phase 0
```
This requires `_slepc_solve` (or a new MHD-specific entry the MHD solve path calls) to receive the `op::MHDStabilityOperator` rather than pre-built `(A,B)`. Decide the cleanest wiring by reading how `solve(::MHDProblem; backend=:slepc)` currently reaches `_slepc_solve` (via `solve_eigenvalue_problem(A,B; backend=:slepc)`); add an MHD-aware path that passes `op` through to the extension (e.g. a `Magrathea._SLEPC_MHD_SOLVER` hook analogous to `_SLEPC_SOLVER`, registered by the extension, used by `solve(::MHDProblem)` when `backend==:slepc`). Keep the generic `_slepc_solve(A,B;...)` (Phase 0) for non-MHD and as fallback. Report the exact wiring you chose.

- [ ] **Step 4: Parse + symbol-audit + CORE_OK** — `$JL -e 'for f in ("ext/MagratheaSlepcExt/raw_petsc.jl","ext/MagratheaSlepcExt/MagratheaSlepcExt.jl"); Meta.parseall(read(f,String)); end; println("PARSE_OK")'`; audit `MatMPIAIJSetPreallocation`, `MatGetOwnershipRange`, `MatSetValue`, `MatZeroRows` (raw), `MatAssembly*` against installed source; `$JL --project=. -e 'using Magrathea; println("CORE_OK")'` (sandbox off). Report the audit table + any symbol you could not confirm.

- [ ] **Step 5: Commit (ASK USER FIRST)** — `git add ext/MagratheaSlepcExt/` / `git commit -m "feat(mpi): distributed MHD assembly + BC application in SLEPc extension"`

---

## Task 4: Wire tests + full suite + docs

**Files:** Modify `test/runtests.jl`, `README.md`, `test/distributed_assembly.jl`.

- [ ] **Step 1: Wire serial tests** — add `include("distributed_assembly.jl")` to `test/runtests.jl` (near `include("dof_ownership.jl")`).

- [ ] **Step 2: Append a guarded cluster note-test** to `test/distributed_assembly.jl`:
```julia
@testset "Distributed MHD assembly (requires PETSc+MUMPS under MPI)" begin
    if !haskey(ENV, "PETSC_DIR")
        @info "PETSC_DIR unset — skipping distributed MHD assembly test (run under mpirun)"
        @test true
    else
        @info "Run the mpirun eigenvalue-match validation manually; see README."
        @test true
    end
end
```

- [ ] **Step 3: Full suite green** — `$JL --project=. test/runtests.jl 2>&1 | grep -iE "Error During Test|did not pass"; echo done` → only `done`; confirm `distributed interior COO partition-reassembles` appears in the log.

- [ ] **Step 4: Update README** — under the SLEPc section, note that the MHD path now assembles distributed (each rank builds only its owned rows; BCs applied on the distributed matrix); onset/biglobal/triglobal still use replicated assembly (later phases). Same `mpirun` driver pattern.

- [ ] **Step 5: Commit (ASK USER FIRST)** — `git add test/runtests.jl test/distributed_assembly.jl README.md` / `git commit -m "test+docs: distributed MHD assembly tests and notes"`

---

## Self-review notes

- **Spec coverage:** `_mhd_index_map` (T1), `_owned_coo_nnz` (T1), `_assemble_mhd_coo` extract + owned filter + block-skip + thin wrapper (T2), partition-reassemble verification vs pre-BC + serial regression (T2), `_create_dist_mat`/`_fill_dist_mat!` (T3), `_apply_dist_bcs!` distributed BC port + `MatZeroRows` (T3), MHD solve rewire (T3), suite wiring + guarded cluster test + docs (T4). All spec sections mapped.
- **Placeholder note:** T3 Step 2 `_apply_dist_bcs!` is given as a structured sketch + a hard requirement to mirror the serial BC formulas (lines ~604–740) exactly, because reproducing every BC row formula here would be guesswork against an unverifiable PETSc API — the implementer reads the serial functions and ports them. This is the one place full code isn't inlined; it is bounded (port existing functions) and cluster-validated.
- **Type consistency:** `_assemble_mhd_coo` returns the NamedTuple `(A_rows,A_cols,A_vals,B_rows,B_cols,B_vals,n,interior_dofs,info)` used by the wrapper, the partition test, and the extension. `_owned_coo_nnz(rows,cols,rstart,rend)` signature consistent T1↔T3. Owned-row convention `(rstart+1):rend` consistent throughout.
- **Verification honesty:** T1+T2 (and the partition-reassemble correctness proof) run here; T3+T4's PETSc pieces — especially `_apply_dist_bcs!` — are cluster-only, validated by the `mpirun` eigenvalue-match run.
