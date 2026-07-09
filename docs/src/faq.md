# FAQ & Troubleshooting

<div class="magrathea-hero">
  <div class="magrathea-eyebrow">Help</div>
  <h1>FAQ &amp; troubleshooting.</h1>
  <p>Common questions and fixes for Magrathea.jl users.</p>
</div>

## Installation

### Q: Julia complains about incompatible package versions

**Solution:** Update packages and resolve dependencies:

```julia
julia> using Pkg
julia> Pkg.update()
julia> Pkg.resolve()
julia> Pkg.instantiate()
```

If conflicts persist, delete `Manifest.toml` and re-instantiate:

```bash
rm Manifest.toml
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

---

### Q: `Pkg.instantiate()` hangs or takes too long

**Possible causes:**
- Network issues or proxy settings
- Large dependencies (MKL, FFTW) downloading for the first time

**Solutions:**
1. Check your internet connection
2. Wait patiently on first install (can take several minutes)
3. Try setting `JULIA_PKG_SERVER` environment variable:
   ```bash
   export JULIA_PKG_SERVER=""
   ```

---

## Running Examples

### Q: Example script uses removed v1 constructors

**Error:**
```
UndefVarError from an old v1 setup script
```

**Solution:** Update the script to use `OnsetParams`, a problem wrapper, and `solve`, then ensure Julia is launched with the project environment:

```bash
julia --project=. example/linear_stability_demo.jl
```

Or activate manually:
```julia
using Pkg
Pkg.activate(".")
using Magrathea
```

---

### Q: The eigenvalue search diverges or oscillates

**Symptoms:**
- Growth rate jumps erratically
- Solver fails to converge

**Solutions:**

1. **Provide a better initial guess:**
   ```julia
   Ra_c, ω_c, _ = find_critical_rayleigh(..., Ra_guess=5e5)
   ```

2. **Reduce the number of eigenvalues:**
   ```julia
   result = solve(OnsetProblem(params); nev=2)
   ```

3. **Adjust tolerance:**
   ```julia
   result = solve(OnsetProblem(params); tol=1e-5, maxiter=200)
   ```

4. **Check parameter regime:** Extreme Ekman numbers (``E < 10^{-7}``) require higher resolution.

---

### Q: Results don't match expected benchmark values

**Checklist:**
1. Verify all parameters match exactly (E, Pr, Ra, m, χ)
2. Check boundary condition settings
3. Ensure sufficient resolution (increase `lmax`, `Nr`)
4. Compare against published tables (Christensen & Wicht 2015)

---

## Performance

### Q: Memory usage spikes beyond expectations

**Diagnosis:** Problem size is too large for available RAM.

**Solutions:**

1. **Reduce resolution:**
   ```julia
   params = OnsetParams(..., lmax=30, Nr=32)  # Reduced from lmax=60, Nr=64
   ```

2. **Check DOF before solving:**
   ```julia
   op = LinearStabilityOperator(params)
   println("DOF: ", op.total_dof)  # Should be < 100,000 for desktop
   ```

3. **For tri-global problems:**
   ```julia
   size_report = estimate_triglobal_problem_size(params)
   println("Total DOFs: ", size_report.total_dofs)
   println("DOFs per mode: ", size_report.dofs_per_mode)
   ```

**Memory guidelines:**

| DOFs | Approx Memory | Suitable for |
|------|---------------|--------------|
| <10,000 | <1 GB | Laptop |
| 10,000-100,000 | 1-16 GB | Desktop |
| >100,000 | >16 GB | Workstation/cluster |

---

### Q: Solver returns zero or NaN eigenvalues

**Possible causes:**
1. Matrix is singular or ill-conditioned
2. Boundary conditions over-constrained
3. Parameters at critical point

**Solutions:**

1. **Add a small shift:**
   ```julia
   eigenvalues, _, _ = solve_eigenvalue_problem(op; which = :LM)
   ```

2. **Check matrix condition:**
   ```julia
   using LinearAlgebra
   cond_A = cond(Matrix(A[interior_dofs, interior_dofs]))
   println("Condition number: ", cond_A)
   ```

3. **Verify boundary conditions:**
   ```julia
   println("Interior DOFs: ", length(interior_dofs))
   # Should be > 0
   ```

---

### Q: Computation is slow for large problems

**Optimization strategies:**

1. **Use sparse matrices** (default in Magrathea.jl)

2. **Adjust BLAS threads:**
   ```julia
   using LinearAlgebra
   BLAS.set_num_threads(4)
   ```

3. **Request fewer eigenvalues:**
   ```julia
   result = solve(OnsetProblem(params); nev=4)  # Not nev=20
   ```

4. **Lower tolerance during exploration:**
   ```julia
   # Quick exploration
   result = solve(OnsetProblem(params); tol=1e-4)

   # Final production run
   result = solve(OnsetProblem(params); tol=1e-8)
   ```

---

## Basic States

### Q: Custom basic state produces NaN values

**Checklist:**

1. **Verify grid matching:**
   ```julia
   @assert bs.Nr == params.Nr
   @assert abs(bs.r[1] - χ) < 1e-10
   ```

2. **Check dictionary completeness:**
   ```julia
   for l in 0:bs.lmax_bs
       @assert haskey(bs.theta_coeffs, l)
   end
   ```

3. **Validate derivatives:**
   ```julia
   # Derivatives should be computed from same Chebyshev operator
   bs.dtheta_dr_coeffs[l] = cd.D1 * bs.theta_coeffs[l]
   ```

4. **Start from working examples:**
   ```julia
   # Use conduction as baseline
   bs_test = conduction_basic_state(cd, χ)
   ```

---

### Q: How do I import coefficients from external codes?

**Workflow:**

1. Export spectral coefficients from source code (Rayleigh, Magic, etc.)
2. Save to portable format (HDF5, JLD2, NPY)
3. Load and transform to Magrathea.jl convention:

```julia
using JLD2, Interpolations

# Load external data
@load "external_data.jld2" theta_lm r_external

# Interpolate to Magrathea.jl grid
cd = ChebyshevDiffn(Nr, [χ, 1.0], 4)

theta_coeffs = Dict{Tuple{Int,Int}, Vector{Float64}}()
for ((l, m), data) in theta_lm
    itp = LinearInterpolation(r_external, data)
    theta_coeffs[(l, m)] = itp.(cd.x)
end
```

---

## MHD Module

### Q: MHD extension slows everything down

**Explanation:** MHD doubles the number of variables (velocity + magnetic field).

**Solutions:**

1. **Use coarser grids while prototyping:**
   ```julia
   params = MHDParams(..., lmax=10, N=16)  # Quick test
   ```

2. **Verify before production:**
   ```julia
   op = MHDStabilityOperator(params)
   println("Matrix size: ", op.matrix_size)
   ```

3. **Pre-allocate arrays** if extending the code.

---

### Q: Magnetic boundary conditions confuse me

**Quick reference:**

| Value | Type | Physical Meaning |
|-------|------|------------------|
| 0 | Insulating | No currents outside (Earth's mantle) |
| 1 | Conducting | Finite conductivity boundary |
| 2 | Perfect conductor | Infinite conductivity (Earth's inner core) |

**Earth-like configuration:**
```julia
params = MHDParams(
    ...,
    bci_magnetic = 2,  # Perfect conductor at ICB
    bco_magnetic = 0,  # Insulating at CMB
)
```

---

### Q: Results don't match Kore (Python implementation)

**Common differences:**

1. **Parameter definitions:** Verify E, Pr, Pm, Ra, Le match exactly
2. **Boundary conditions:** Magrathea.jl uses numerical codes (0, 1, 2)
3. **Approximation:** Magrathea.jl uses Boussinesq (no anelastic effects)
4. **Resolution:** May need higher `lmax` or `N` for convergence

**Validation approach:**
```julia
# Reproduce known benchmark
params = MHDParams(
    E = 4.734e-5, Pr = 1.0, Pm = 1.0,
    Ra = 1.6e6, Le = 0.0,  # Hydrodynamic benchmark
    ...
)
# Compare against Christensen & Wicht (2015) Table 1
```

---

## Documentation

### Q: Documenter build fails or `make.jl` errors

**Solution:**

1. Instantiate the docs environment (once):
   ```bash
   julia --project=docs -e '
       using Pkg
       Pkg.develop(PackageSpec(path=pwd()))
       Pkg.instantiate()'
   ```

2. Build the site:
   ```bash
   julia --project=docs docs/make.jl
   ```

3. Open `docs/build/index.html` in your browser to preview.

---

### Q: How do I publish on GitHub Pages?

**Steps:**

1. Enable GitHub Pages in repository settings
2. Choose "GitHub Actions" as source
3. The `.github/workflows/docs.yml` in this repo uses Documenter's `deploydocs` — push to `main` triggers automatic deployment.

---

## General Tips

### Debugging Eigenvalue Problems

```julia
# 1. Check matrix properties
using LinearAlgebra, SparseArrays

op = LinearStabilityOperator(params)
A, B = op.A, op.B

println("A size: ", size(A))
println("A nnz: ", nnz(A), " (", 100*nnz(A)/prod(size(A)), "% fill)")
println("A symmetric: ", issymmetric(A))

# 2. Check for NaN/Inf
@assert !any(isnan, A.nzval) "A contains NaN"
@assert !any(isinf, A.nzval) "A contains Inf"

# 3. Test with dense solver for small problems
if size(A, 1) < 1000
    λ_dense = eigen(Matrix(A), Matrix(B)).values
    println("Dense eigenvalues: ", sort(λ_dense, by=real, rev=true)[1:5])
end
```

### Performance Profiling

```julia
using BenchmarkTools

# Time matrix assembly
@btime op = LinearStabilityOperator($params)

# Time eigenvalue solve
@btime solve_eigenvalue_problem($op; nev=4)

# Profile memory
@allocated solve(OnsetProblem(params); nev=4)
```

---

## Getting Help

If you're still stuck:

1. **Check existing issues:** [GitHub Issues](https://github.com/subhk/Magrathea.jl/issues)

2. **Open a new issue** with:
   - Julia version (`VERSION`)
   - Platform (Linux, macOS, Windows/WSL)
   - Minimal reproducing script
   - Complete error message

3. **Enable verbose output:**
   ```julia
   ENV["MAGRATHEA_VERBOSE"] = "1"
   # Run your code
   ```

---

!!! tip "Quick Diagnostics"
    Run this snippet to gather system info for bug reports:
    ```julia
    using Pkg
    println("Julia: ", VERSION)
    println("OS: ", Sys.KERNEL)
    Pkg.status("Magrathea")
    ```
