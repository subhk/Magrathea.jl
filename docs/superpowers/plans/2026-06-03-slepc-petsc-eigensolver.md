# SLEPc/PETSc Eigensolver Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SLEPc (via `SlepcWrap.jl`+`PetscWrap.jl`) as a `backend=:slepc` option for the generalized eigenproblem `A x = σ B x`, behind an optional package extension, leaving KrylovKit the unchanged default.

**Architecture:** Additive. Core `Magrathea` gains a `Ref` hook + `_dispatch_eigen` switch; every solve site gets a `backend` kwarg (default `:krylovkit`, byte-for-byte unchanged) that diverts the same `(A,B)` to a weak-dependency extension `MagratheaSlepcExt` when `:slepc`. The extension is the only place PETSc/SLEPc/MPI symbols appear.

**Tech Stack:** Julia 1.12, package extensions (weakdeps), PetscWrap.jl, SlepcWrap.jl, MPI.jl, KrylovKit (existing).

**Spec:** `docs/superpowers/specs/2026-06-03-slepc-petsc-eigensolver-design.md`

---

## Environment / workflow notes (read first)

- **Run Julia via the versioned binary** (juliaup launcher is broken here):
  `JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia`
  Bash tool needs `dangerouslyDisableSandbox: true`. Use Julia **1.12.x** (Manifest is v1.12).
- **No PETSc build in this environment.** Tasks 1–5 (core hook, threading, packaging, stub/error behavior) are fully verifiable here. Task 6 (the SLEPc solver body) and Task 7's equivalence test **cannot be precompiled or run here** — they are verified by the user on a PETSc/SLEPc-enabled machine. Treat Task 6's code as "best-effort against the installed PetscWrap/SlepcWrap legacy API (verified from their `example/helmholtz.jl` + `example/complex.jl`); confirm exact symbol names against the installed package version before running."
- **Commits:** the user's standing rule is *no `git commit` without explicit permission*. Each task ends with a commit step — **pause and ask the user before running it**. Do not auto-commit.

---

## File Structure

- `src/Stability/solver.jl` (modify) — add `_SLEPC_SOLVER` hook, `_solve_generalized_eigen_slepc`, `_dispatch_eigen`; add `backend` kwarg to `solve_eigenvalue_problem(A,B)`. Owns the backend contract.
- `src/Stability/linear.jl` (modify) — `backend` kwarg on `solve_eigenvalue_problem(op)` and `_krylov_eigensolve_optimized`; divert reduced pencil.
- `src/Stability/triglobal.jl` (modify) — `backend` kwarg on `solve_block_eigenvalue_problem` + its caller; divert assembled pencil.
- `src/solve.jl` (modify) — `backend` kwarg on each `solve(::*Problem)`; galerkin `eigen` path diverts when `:slepc`.
- `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl` (create) — the SLEPc solver; registers itself into the hook on load.
- `Project.toml` (modify) — `[weakdeps]`, `[extensions]`, `[compat]`.
- `test/slepc_backend.jl` (create) — runs-here tests (stub error, unknown backend, default unchanged) + a `PETSC_DIR`-guarded equivalence test.
- `test/runtests.jl` (modify) — `include("slepc_backend.jl")`.

---

## Task 1: Core backend hook + dispatcher + `solve_eigenvalue_problem(A,B)` divert

**Files:**
- Modify: `src/Stability/solver.jl` (after the `using Logging` line / before `solve_eigenvalue_problem`)
- Test: `test/slepc_backend.jl` (create)

- [ ] **Step 1: Write the failing tests** (create `test/slepc_backend.jl`)

```julia
using Test
using SparseArrays
using Magrathea

@testset "SLEPc backend dispatch (core, no PETSc)" begin
    A = sparse(ComplexF64[2 0 0; 0 3 0; 0 0 4])
    B = sparse(ComplexF64[1 0 0; 0 1 0; 0 0 1])

    # Unknown backend → ArgumentError
    @test_throws ArgumentError Magrathea.solve_eigenvalue_problem(A, B; nev=1, backend=:nope)

    # :slepc without the extension loaded → actionable error naming the packages
    err = try
        Magrathea.solve_eigenvalue_problem(A, B; nev=1, backend=:slepc)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("PetscWrap", sprint(showerror, err))
    @test occursin("SlepcWrap", sprint(showerror, err))

    # Default backend still solves (unchanged KrylovKit path)
    vals, vecs, info = Magrathea.solve_eigenvalue_problem(A, B; nev=1, sigma=0.0)
    @test eltype(vals) <: Complex
    @test info["solver"] == :krylovkit
end
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$JL --project=. -e 'using Test; @testset "t" begin include("test/slepc_backend.jl") end'`
Expected: FAIL — `:slepc` does not yet throw the right error / `backend` kwarg unknown.

- [ ] **Step 3: Add the hook + dispatcher in `src/Stability/solver.jl`** (insert after line 16, `using Logging`)

```julia
# ---------------------------------------------------------------------------
# Pluggable eigensolver backends. The SLEPc backend lives in an optional package
# extension (MagratheaSlepcExt) and registers itself here on load; core never
# references PETSc/SLEPc symbols, keeping `using Magrathea` PETSc-free.
# ---------------------------------------------------------------------------
const _SLEPC_SOLVER = Ref{Union{Nothing,Function}}(nothing)

"""Solve `A x = σ B x` with SLEPc. Requires the MagratheaSlepcExt extension (load
`PetscWrap` and `SlepcWrap`, with a complex-scalar PETSc build)."""
function _solve_generalized_eigen_slepc(A::SparseMatrixCSC, B::SparseMatrixCSC; kwargs...)
    solver = _SLEPC_SOLVER[]
    solver === nothing && error(
        "backend=:slepc requires the SLEPc extension. Load it with " *
        "`using PetscWrap, SlepcWrap` (needs a complex-scalar PETSc build with " *
        "PETSC_DIR/PETSC_ARCH and SLEPC_DIR set).")
    return solver(A, B; kwargs...)
end

"""Dispatch a sparse generalized eigensolve to the selected backend, returning the
common `(eigenvalues::Vector{Complex}, eigenvectors::Matrix{Complex}, info::Dict)`."""
function _dispatch_eigen(A::SparseMatrixCSC, B::SparseMatrixCSC;
                         backend::Symbol=:krylovkit,
                         nev::Int, sigma, which::Symbol, selection::Symbol,
                         tol::Float64, maxiter::Int,
                         krylovdim::Union{Nothing,Int}, verbosity::Int)
    if backend === :krylovkit
        return _krylov_eigensolve(A, B; nev=nev, sigma=sigma, which=which,
                                  selection=selection, tol=tol, maxiter=maxiter,
                                  krylovdim=krylovdim, verbosity=verbosity)
    elseif backend === :slepc
        return _solve_generalized_eigen_slepc(A, B; nev=nev, sigma=sigma, which=which,
                                              selection=selection, tol=tol, maxiter=maxiter,
                                              verbosity=verbosity)
    else
        throw(ArgumentError("Unknown eigensolver backend $(backend); use :krylovkit or :slepc"))
    end
end
```

- [ ] **Step 4: Thread `backend` through `solve_eigenvalue_problem(A,B)`** in `src/Stability/solver.jl`

Change the signature (line 96-104) to add `backend::Symbol=:krylovkit,` (place it just after `nev::Int=1,`), and replace the `_krylov_eigensolve(...)` call (lines 114-122) with:

```julia
    eigenvalues, eigenvectors, info = _dispatch_eigen(A, B;
                                                      backend = backend,
                                                      nev = nev,
                                                      sigma = sigma,
                                                      which = which,
                                                      selection = selection,
                                                      tol = tol,
                                                      maxiter = maxiter,
                                                      krylovdim = krylovdim,
                                                      verbosity = verbosity)
```

Also update the `@info` line (112) to interpolate the backend: change `solver="KrylovKit shift-invert"` to `solver=backend`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$JL --project=. -e 'using Test; @testset "t" begin include("test/slepc_backend.jl") end'`
Expected: PASS (3 testsets).

- [ ] **Step 6: Commit** (ASK USER FIRST)

```bash
git add src/Stability/solver.jl test/slepc_backend.jl
git commit -m "feat(solver): add pluggable eigensolver backend hook + :slepc dispatch"
```

---

## Task 2: Thread `backend` through constrained hydro solve (`linear.jl`)

**Files:**
- Modify: `src/Stability/linear.jl:606-620` (`solve_eigenvalue_problem(op)`) and `:634-702` (`_krylov_eigensolve_optimized`)

- [ ] **Step 1: Add `backend` kwarg to `solve_eigenvalue_problem(op)`** (line 606)

Add `backend::Symbol=:krylovkit,` after `nev::Int=6,`, and pass `backend=backend` into the `_krylov_eigensolve_optimized(...)` call (line 617).

- [ ] **Step 2: Add `backend` kwarg + divert in `_krylov_eigensolve_optimized`** (line 634)

Add `backend::Symbol=:krylovkit,` to its kwargs. After the reduced matrices are built (line 648-649) and the shift `σ` is selected (lines 651-667), insert the SLEPc divert **before** the `lu_factor = lu(A - σ * B)` line:

```julia
    if backend === :slepc
        vals_s, vecs_s, info_s = _solve_generalized_eigen_slepc(
            sparse(A), sparse(B); nev=nev, sigma=σ, which=which,
            selection=:maxreal, tol=tol, maxiter=maxiter, verbosity=0)
        # Reconstruct full-space eigenvectors from reduced columns (same as the
        # KrylovKit branch does via _reconstruct_full_vector).
        vecs_full = [_reconstruct_full_vector(reduction, vals_s isa Nothing ? vecs_s : vecs_s[:, j])
                     for j in 1:size(vecs_s, 2)]
        ordering = which == :LR ? sortperm(real.(vals_s); rev=true) :
                   which == :LM ? sortperm(abs.(vals_s); rev=true) :
                   collect(1:length(vals_s))
        ordering = ordering[1:min(nev, length(ordering))]
        return vals_s[ordering], vecs_full[ordering], info_s
    end
```

(Note: `A, B` here are the reduced matrices from line 648; `reduction` is its third return value. `_reconstruct_full_vector(reduction, v)` is the existing helper used at line 693.)

- [ ] **Step 3: Verify the default path is unchanged (compile + existing tests)**

Run: `$JL --project=. -e 'using Test; @testset "t" begin include("test/onset_api.jl"); include("test/mean_flow_stability.jl") end' 2>&1 | tail -5`
Expected: all PASS (KrylovKit default untouched).

- [ ] **Step 4: Add a runs-here stub test** to `test/slepc_backend.jl` (inside a new `@testset`)

```julia
@testset "SLEPc backend reaches constrained hydro path" begin
    params = OnsetParams(E=1e-3, Pr=1.0, Ra=100.0, χ=0.35, m=2, lmax=6, Nr=16)
    op = LinearStabilityOperator(params)
    err = try
        Magrathea.solve_eigenvalue_problem(op; nev=1, backend=:slepc)
        nothing
    catch e; e end
    @test err isa ErrorException && occursin("SlepcWrap", sprint(showerror, err))
end
```

- [ ] **Step 5: Run it**

Run: `$JL --project=. -e 'using Test; @testset "t" begin include("test/slepc_backend.jl") end'`
Expected: PASS.

- [ ] **Step 6: Commit** (ASK USER FIRST)

```bash
git add src/Stability/linear.jl test/slepc_backend.jl
git commit -m "feat(linear): route constrained hydro solve through backend dispatch"
```

---

## Task 3: Thread `backend` through triglobal solve

**Files:**
- Modify: `src/Stability/triglobal.jl:1899` (`solve_block_eigenvalue_problem`) and its caller `solve_triglobal_eigenvalue_problem` (find with `grep -n "solve_block_eigenvalue_problem" src/Stability/triglobal.jl`)

- [ ] **Step 1: Add `backend` kwarg + divert in `solve_block_eigenvalue_problem`** (line 1899)

Change signature to add a trailing kwarg: `function solve_block_eigenvalue_problem(A, B, σ_target, nev, verbose; backend::Symbol=:krylovkit)`. Immediately after computing `shift = Complex{T}(T(σ_target), T(1e-6))` (line 1909), insert:

```julia
    if backend === :slepc
        vals_s, vecs_s, _ = _solve_generalized_eigen_slepc(
            A, B; nev=nev, sigma=shift, which=:LR, selection=:maxreal,
            tol=T(1e-8), maxiter=200, verbosity=0)
        perm = sortperm(real.(vals_s); rev=true)
        return vals_s[perm], vecs_s[:, perm]
    end
```

(This matches the function's existing 2-tuple return `(eigenvalues, eigenvectors)`.)

- [ ] **Step 2: Thread `backend` from the caller**

In `solve_triglobal_eigenvalue_problem` (the function that calls `solve_block_eigenvalue_problem`), add `backend::Symbol=:krylovkit` to its kwargs and pass `; backend=backend` to the `solve_block_eigenvalue_problem(...)` call. (Locate exact lines via grep; mirror the kwarg-threading style.)

- [ ] **Step 3: Verify default path unchanged**

Run: `$JL --project=. -e 'using Test; @testset "t" begin include("test/triglobal.jl") end' 2>&1 | tail -5`
Expected: all PASS.

- [ ] **Step 4: Commit** (ASK USER FIRST)

```bash
git add src/Stability/triglobal.jl
git commit -m "feat(triglobal): route block eigensolve through backend dispatch"
```

---

## Task 4: Thread `backend` through `solve(::*Problem)` + galerkin dense path

**Files:**
- Modify: `src/solve.jl` — `solve(::OnsetProblem)` (line 96), `solve(::BiglobalProblem)` (131), `solve(::TriglobalProblem)` (~167), `solve(::MHDProblem)` (220), galerkin `eigen` (242-251).

- [ ] **Step 1: Add `backend::Symbol=:krylovkit` kwarg to each `solve(::*Problem)` method** and forward it to the underlying solver call:
  - OnsetProblem (line 105) → `solve_onset_problem(...; backend=backend, ...)` — and add `backend` kwarg to `solve_onset_problem` (in `onset.jl`), forwarding into its `solve_eigenvalue_problem(op; ...)` call.
  - BiglobalProblem (line 150) → same, via `solve_biglobal_problem`.
  - TriglobalProblem (~line 175) → forward into `solve_triglobal_eigenvalue_problem(...; backend=backend)`.
  - MHDProblem tau branch (line 265) → `solve_eigenvalue_problem(A, B; ..., backend=backend)`.

- [ ] **Step 2: Divert the galerkin dense `eigen` when `:slepc`** (replace lines 242-251 region)

```julia
        if backend === :slepc
            vals_s, vecs_s, _ = Magrathea._solve_generalized_eigen_slepc(
                sparse(A_gal), sparse(B_gal); nev=nev,
                sigma = sigma === nothing ? zero(Complex{T}) : Complex{T}(sigma),
                which=:LR, selection=:maxreal, tol=1e-10, maxiter=1000, verbosity=0)
            eigenvalues = vals_s
            evecs_full = [reconstruct_mhd_galerkin_full(op, layout, vecs_s[:, j])
                          for j in 1:size(vecs_s, 2)]
        else
            F = eigen(A_gal, B_gal)
            keep = findall(isfinite, F.values)
            vals = F.values[keep]
            order = sigma !== nothing ? sortperm(abs.(vals .- Complex{T}(sigma))) :
                    which === :LM      ? sortperm(abs.(vals); rev=true) :
                    which === :LI      ? sortperm(imag.(vals); rev=true) :
                                         sortperm(real.(vals); rev=true)
            sel = order[1:min(nev, length(order))]
            eigenvalues = vals[sel]
            evecs_full = [reconstruct_mhd_galerkin_full(op, layout, F.vectors[:, keep[s]]) for s in sel]
        end
        evec_matrix = _eigvecs_to_matrix(eigenvalues, evecs_full, T)
```

(Keep the surrounding `info`/`StabilityResult` return as-is.)

- [ ] **Step 3: Verify defaults unchanged**

Run: `$JL --project=. test/runtests.jl 2>&1 | grep -iE "Error During|did not pass" | head; echo done`
Expected: no failures (only testset-name false positives), exit clean.

- [ ] **Step 4: Add runs-here divert test** to `test/slepc_backend.jl`

```julia
@testset "SLEPc backend reaches solve(::MHDProblem) galerkin path" begin
    p = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1.0, ricb=0.35,
                  m=1, lmax=3, N=8, B0_type=axial, B0_amplitude=1.0)
    err = try; solve(MHDProblem(p); nev=1, backend=:slepc); nothing; catch e; e end
    @test err isa ErrorException && occursin("PetscWrap", sprint(showerror, err))
end
```

- [ ] **Step 5: Run it**, expect PASS.

- [ ] **Step 6: Commit** (ASK USER FIRST)

```bash
git add src/solve.jl src/Stability/onset.jl
git commit -m "feat(solve): thread backend kwarg through public solves + galerkin path"
```

---

## Task 5: Packaging — weakdeps, extension skeleton, compat

**Files:**
- Modify: `Project.toml`
- Create: `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl` (skeleton that registers the hook)

- [ ] **Step 1: Edit `Project.toml`** — add to `[weakdeps]` (alongside RecipesBase, Makie):

```toml
PetscWrap = "5be22e1c-01b5-4697-96eb-ef9ccdc854b8"
SlepcWrap = "c3679e3b-785e-4ccc-b734-b7685cbb935e"
```

Add to `[extensions]`:

```toml
MagratheaSlepcExt = ["PetscWrap", "SlepcWrap"]
```

Add to `[compat]` (use the versions resolved in the temp-env probe; confirm with `$JL --project=. -e 'using Pkg; Pkg.status()'` after adding):

```toml
MPI = "0.20"
PetscWrap = "0.2"
SlepcWrap = "0.2"
```

- [ ] **Step 2: Create `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl` skeleton**

```julia
module MagratheaSlepcExt

using Magrathea
using SparseArrays
using PetscWrap
using SlepcWrap

# Real solver added in Task 6. For now register a placeholder so the wiring is testable.
function _slepc_solve(A::SparseMatrixCSC, B::SparseMatrixCSC; kwargs...)
    error("MagratheaSlepcExt._slepc_solve not yet implemented")
end

function __init__()
    Magrathea._SLEPC_SOLVER[] = _slepc_solve
    return nothing
end

end # module
```

- [ ] **Step 3: Verify core still precompiles WITHOUT the extension** (no PETSc needed)

Run: `$JL --project=. -e 'using Magrathea; println("CORE_OK")'`
Expected: prints `CORE_OK` (extension not triggered because PetscWrap/SlepcWrap aren't loaded; core unaffected).

- [ ] **Step 4: Verify the full default suite still passes**

Run: `$JL --project=. test/runtests.jl 2>&1 | grep -iE "Error During|did not pass"; echo done`
Expected: no real failures.

- [ ] **Step 5: Commit** (ASK USER FIRST)

```bash
git add Project.toml ext/MagratheaSlepcExt/MagratheaSlepcExt.jl
git commit -m "feat: register MagratheaSlepcExt weak-dependency extension skeleton"
```

---

## Task 6: SLEPc solver implementation (extension body) — NOT runnable here

**Files:**
- Modify: `ext/MagratheaSlepcExt/MagratheaSlepcExt.jl`

> **Constraint:** This cannot be precompiled/run in the dev environment (no PETSc). The
> code below follows the SlepcWrap **legacy** API verified from
> `~/.julia/packages/SlepcWrap/*/example/helmholtz.jl` and `example/complex.jl`.
> On the PETSc machine, confirm each symbol exists in the installed version before running.

- [ ] **Step 1: Replace `_slepc_solve` with the full implementation**

```julia
"""
Solve `A x = σ B x` with SLEPc shift-invert. Returns the Magrathea contract
`(eigenvalues::Vector{ComplexF64}, eigenvectors::Matrix{ComplexF64}, info::Dict)`.
Serial (single MPI rank); requires a complex-scalar PETSc/SLEPc build.
"""
function _slepc_solve(A::SparseMatrixCSC, B::SparseMatrixCSC;
                      nev::Int, sigma, which::Symbol, selection::Symbol,
                      tol::Float64, maxiter::Int, verbosity::Int=0)
    size(A) == size(B) || throw(DimensionMismatch("A and B must match"))
    n = size(A, 1)

    # Shift target: mirror the KrylovKit auto-shift when sigma === nothing.
    target = sigma === nothing ?
        (which === :LR ? ComplexF64(10, 0) :
         which === :LI ? ComplexF64(0, 10) : ComplexF64(1, 0)) :
        ComplexF64(sigma)

    SlepcInitialize("-eps_nev $(nev) -st_type sinvert -eps_target_magnitude " *
                    "-eps_tol $(tol) -eps_max_it $(maxiter)")

    PetscScalar <: Real && (SlepcFinalize();
        error("PETSc/SLEPc must be built with complex scalars (--with-scalar-type=complex)"))

    Amat = _to_petsc(A, n)
    Bmat = _to_petsc(B, n)

    eps = EPSCreate()
    EPSSetOperators(eps, Amat, Bmat)
    EPSSetProblemType(eps, SlepcWrap.EPS_GNHEP)
    EPSSetDimensions(eps, nev, PETSC_DECIDE, PETSC_DECIDE)
    EPSSetTarget(eps, PetscScalar(target))
    EPSSetWhichEigenpairs(eps, SlepcWrap.EPS_TARGET_MAGNITUDE)
    EPSSetFromOptions(eps)
    EPSSetUp(eps)
    EPSSolve(eps)

    nconv = EPSGetConverged(eps)
    nout = min(nconv, nev)
    nout == 0 && (EPSDestroy(eps); MatDestroy(Amat); MatDestroy(Bmat);
                  SlepcFinalize(); error("SLEPc returned no converged eigenpairs"))

    vals = Vector{ComplexF64}(undef, nout)
    vecs = Matrix{ComplexF64}(undef, n, nout)
    vr, vi = MatCreateVecs(Amat)
    for j in 0:(nout - 1)
        vpr, vpi, vecr, veci = EPSGetEigenpair(eps, j, vr, vi)
        vals[j + 1] = ComplexF64(vpr, vpi)
        arr = VecGetArray(vecr)                 # complex build → already complex
        vecs[:, j + 1] .= ComplexF64.(arr)
        VecRestoreArray(vecr, arr)
    end

    info = Dict{String,Any}(
        "solver" => :slepc, "strategy" => :shift_invert, "target" => target,
        "nconv" => nconv, "converged_reason" => EPSGetConvergedReason(eps),
        "selection" => selection)

    EPSDestroy(eps); MatDestroy(Amat); MatDestroy(Bmat); SlepcFinalize()

    perm = _sort_indices_local(vals, selection)
    return vals[perm], vecs[:, perm], info
end

# CSC → PETSc AIJ, inserting per-column (PETSc uses 0-based indices).
function _to_petsc(M::SparseMatrixCSC, n::Int)
    mat = MatCreate()
    MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, n, n)
    MatSetFromOptions(mat); MatSetUp(mat)
    rows = rowvals(M); vals = nonzeros(M)
    for col in 1:size(M, 2)
        for k in nzrange(M, col)
            MatSetValue(mat, rows[k] - 1, col - 1, PetscScalar(vals[k]), INSERT_VALUES)
        end
    end
    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY)
    return mat
end

# Local copy of the core selection ordering (avoids depending on a non-exported core symbol).
function _sort_indices_local(ev::AbstractVector{<:Complex}, selection::Symbol)
    selection === :maxreal      ? sortperm(real.(ev); rev=true) :
    selection === :minabs       ? sortperm(abs.(ev)) :
    selection === :closest_real ? sortperm(abs.(real.(ev))) :
    error("Unknown selection strategy $(selection)")
end
```

- [ ] **Step 2: (On the PETSc machine) verify it loads and registers**

Run (with `PETSC_DIR`/`SLEPC_DIR` set, complex build):
`$JL --project=. -e 'using Magrathea, PetscWrap, SlepcWrap; println(Magrathea._SLEPC_SOLVER[] !== nothing)'`
Expected: `true`. If any symbol (`EPSSetWhichEigenpairs`, `EPS_TARGET_MAGNITUDE`, `VecGetArray`, …) is missing, adjust to the installed SlepcWrap/PetscWrap names (check that package's `src/eps.jl` / `src/Vec.jl`).

- [ ] **Step 3: Commit** (ASK USER FIRST)

```bash
git add ext/MagratheaSlepcExt/MagratheaSlepcExt.jl
git commit -m "feat(slepc): implement SLEPc shift-invert generalized eigensolver"
```

---

## Task 7: Guarded equivalence test + wiring + docs

**Files:**
- Modify: `test/slepc_backend.jl`, `test/runtests.jl`, `README`/docs

- [ ] **Step 1: Append a PETSc-guarded equivalence test** to `test/slepc_backend.jl`

```julia
@testset "SLEPc vs KrylovKit eigenvalue agreement (requires PETSc build)" begin
    if !haskey(ENV, "PETSC_DIR")
        @info "PETSC_DIR unset — skipping SLEPc runtime equivalence test"
    else
        @eval using PetscWrap, SlepcWrap
        p = MHDParams(E=1e-3, Pr=1.0, Pm=1.0, Ra=100.0, Le=1.0, ricb=0.35,
                      m=1, lmax=3, N=12, B0_type=dipole, B0_amplitude=1.0)
        op = MHDStabilityOperator(p)
        A, B, _, _ = assemble_mhd_matrices(op)
        vK, _, _ = Magrathea.solve_eigenvalue_problem(A, B; nev=4, sigma=0.0, backend=:krylovkit)
        vS, _, _ = Magrathea.solve_eigenvalue_problem(A, B; nev=4, sigma=0.0, backend=:slepc)
        @test isapprox(sort(real.(vK))[1:2], sort(real.(vS))[1:2]; rtol=1e-4)
    end
end
```

- [ ] **Step 2: Wire into the suite** — add to `test/runtests.jl` after the other includes:

```julia
include("slepc_backend.jl")
```

- [ ] **Step 3: Run full suite here (PETSc test auto-skips)**

Run: `$JL --project=. test/runtests.jl 2>&1 | grep -iE "Error During|did not pass|skipping SLEPc"; echo done`
Expected: no failures; the "skipping SLEPc" info line appears.

- [ ] **Step 4: Document** — add a short "SLEPc backend (optional)" section to the README/docs: install a complex-scalar PETSc+SLEPc, set `PETSC_DIR`/`PETSC_ARCH`/`SLEPC_DIR`, `]add PetscWrap SlepcWrap`, then `using Magrathea, PetscWrap, SlepcWrap` and pass `backend=:slepc` to `solve`/`solve_eigenvalue_problem`. Note serial-only, complex-build requirement.

- [ ] **Step 5: Commit** (ASK USER FIRST)

```bash
git add test/slepc_backend.jl test/runtests.jl README.md
git commit -m "test+docs: SLEPc backend guarded equivalence test and usage docs"
```

---

## Self-review notes

- **Spec coverage:** hook+dispatch (T1), all 4 sites — MHD tau (T1), constrained hydro/onset/biglobal (T2), triglobal (T3), galerkin dense (T4); packaging/weakdeps/extension (T5); SLEPc body w/ EPS_GNHEP+sinvert+complex guard (T6); error behavior (T1/T2/T4 stub tests); guarded equivalence + docs (T7). All spec sections mapped.
- **Type/contract consistency:** every backend path returns `(Vector{Complex}, Matrix{Complex}, info::Dict)`; `triglobal` site adapts to its 2-tuple return; `linear`/`galerkin` sites convert eigenvector matrix columns through their existing `_reconstruct_full_vector` / `reconstruct_mhd_galerkin_full` helpers. Hook name `_SLEPC_SOLVER`, core entry `_solve_generalized_eigen_slepc`, dispatcher `_dispatch_eigen` used consistently.
- **Verification honesty:** Tasks 1–5 + Task 7's skip-path run here; Task 6 + the PETSc equivalence test run only on a complex-scalar PETSc build (called out at top and in each affected step).
