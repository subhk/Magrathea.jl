# Getting Started

<div class="magrathea-hero">
  <div class="magrathea-eyebrow">Install &amp; first run</div>
  <h1>From a fresh clone to your first eigenvalue.</h1>
  <p>
    Install Magrathea.jl on Linux, macOS, or Windows (WSL), then assemble and solve a
    rotating-convection onset problem &mdash; follow the steps in order.
  </p>
  <div class="magrathea-actions">
    <a class="magrathea-button primary" href="../problem_setup/">First problem</a>
    <a class="magrathea-button secondary" href="../examples/">Examples</a>
  </div>
</div>

## Prerequisites

### Required

- **Julia 1.10 or newer** - Magrathea.jl is developed and tested against Julia 1.10.x and 1.11.x
- **Git** - For cloning and pulling updates

### Optional but Recommended

- **Documenter.jl** - For building the documentation locally (`julia --project=docs docs/make.jl`)
- **VS Code with Julia extension** - For a richer REPL and plot experience
- **C/Fortran toolchain** - For building dependencies such as MKL or FFTW if Julia requests them

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/subhk/Magrathea.jl.git
cd Magrathea.jl
```

### Step 2: Instantiate the Julia Environment

Open Julia inside the project folder and instantiate the dependencies:

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
```

The first run downloads packages including:

| Package | Purpose |
|---------|---------|
| `LinearAlgebra` | Standard Julia linear algebra |
| `SparseArrays` | Sparse matrix operations |
| `LinearMaps` | Linear operator abstractions |
| `SpecialFunctions` | Special mathematical functions |
| `JLD2` | HDF5-based file I/O |
| `WignerSymbols` | Wigner 3j symbols for mode coupling |
| `Parameters` | Keyword argument macros |

!!! tip "Subsequent Sessions"
    Once instantiated, subsequent Julia sessions only need `Pkg.activate(".")` to use the pre-compiled dependencies.

### Step 3: Run the Test Suite

Before creating new problems, verify your installation passes the regression tests:

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.test()
```

The tests assemble small eigenvalue problems to verify that matrix blocks and solver wrappers agree with stored benchmarks.

### Step 4: Verify an Example Script

For a basic smoke test, run the linear stability demo:

```bash
julia --project=. example/linear_stability_demo.jl
```

You should see output similar to:

```
m    Re(λ₁)          Im(λ₁)          iterations
------------------------------------------------
 1  -1.23456e-02   5.67890e-01      24
 2  -8.76543e-03   6.12345e-01      28
...
```

## Package Structure

After installation, the project has the following structure:

```
Magrathea.jl/
├── src/                      # Source code
│   ├── Magrathea.jl              # Main module entry point
│   ├── validation.jl         # Input validation  [v2.0]
│   ├── types.jl              # Problem/result types, estimate_size  [v2.0]
│   ├── solve.jl              # Unified solve() API  [v2.0]
│   ├── show.jl               # Pretty-printing  [v2.0]
│   ├── Spectral/             # Chebyshev, ultraspherical, Galerkin discretization
│   ├── BasicStates/          # Basic states, SH transforms, coupling operators
│   ├── Stability/            # Onset / biglobal / triglobal + eigensolver
│   ├── Operators/            # Sparse operators + boundary conditions
│   └── MHD/                  # MHD extension (see Codebase Structure for the full tree)
├── example/                  # Example scripts
│   ├── linear_stability_demo.jl
│   ├── mhd_dynamo_example.jl
│   ├── triglobal_analysis_demo.jl
│   ├── basic_state_onset_example.jl
│   └── ...
├── test/                     # Test suite
├── docs/                     # Documentation
├── Project.toml              # Package dependencies
└── Manifest.toml             # Locked dependency versions
```

!!! note "Public API"
    Magrathea.jl uses `OnsetParams`, typed problem wrappers such as `OnsetProblem`, `solve`, and `estimate_size` as the public stability-analysis interface.

## Julia Configuration

### Recommended Startup Configuration

Add the following to `~/.julia/config/startup.jl` for a better REPL experience:

```julia
atreplinit() do repl
    try
        @eval using Revise
    catch err
        @warn "Revise not available" err
    end
end
```

This enables [Revise.jl](https://github.com/timholy/Revise.jl) to pick up changes as you edit source files - essential for iterative development.

### Environment Variables

Magrathea.jl recognizes several environment variables:

| Variable | Purpose | Example |
|----------|---------|---------|
| `MAGRATHEA_VERBOSE` | Enable verbose output | `"1"` |
| `MAGRATHEA_THETA_POINTS` | Default meridional resolution | `"96"` |

## Building Documentation Locally

The documentation uses [Documenter.jl](https://documenter.juliadocs.org/). To preview locally:

### Step 1: Instantiate the docs environment

```bash
julia --project=docs -e '
    using Pkg
    Pkg.develop(PackageSpec(path=pwd()))
    Pkg.instantiate()'
```

### Step 2: Build the site

```bash
julia --project=docs docs/make.jl
```

Open `docs/build/index.html` in your browser to see the rendered site.

## Your First Calculation

Let's compute the growth rate for a rotating convection problem:

```julia
using Magrathea

# 1. Define parameters
params = OnsetParams(E=1e-3, Pr=1.0, Ra=1e5, χ=0.35,
                     m=4, lmax=20, Nr=32)

# 2. Create and solve
problem = OnsetProblem(params)
result = solve(problem; nev=6)

# 3. View results
println("Growth rate: ", result.growth_rate)
println("Frequency: ", result.frequency)
```

### Understanding the Output

The eigenvalues ``\lambda = \sigma + i\omega`` represent:

- **Growth rate (``\sigma``)**: Positive values indicate instability (convection onset)
- **Drift frequency (``\omega``)**: Rate at which the pattern rotates azimuthally

For Earth-like parameters at the onset of convection:

- ``\sigma \approx 0`` (marginal stability)
- ``\omega > 0`` (prograde drift with rotation)

## Troubleshooting

### Package Refuses to Precompile

```julia
julia> Pkg.update()
julia> Pkg.instantiate()
```

If issues persist, delete `Manifest.toml` and re-instantiate.

### Out-of-Memory Errors

Reduce `lmax` or `Nr` in the examples. Start with smaller values:

```julia
params = OnsetParams(
    ...,
    lmax = 30,    # Reduced from 60
    Nr = 32,      # Reduced from 64
)
```

### Solver Doesn't Converge

Try adjusting solver parameters:

```julia
result = solve(OnsetProblem(params);
    nev = 4,
    tol = 1e-5,       # Relaxed tolerance
    maxiter = 200,    # More iterations
)
```

### Julia Version Mismatch

Ensure you're using Julia 1.10 or newer:

```julia
julia> VERSION
v"1.10.0"
```

## Next Steps

Now that Magrathea.jl is installed and working, proceed to:

1. **[Setting Up Your First Problem](problem_setup.md)** - Learn to configure and solve onset problems
2. **[Basic States](basic_states.md)** - Create custom background temperature and flow profiles
3. **[Examples](examples.md)** - Explore the example scripts in the `example/` directory

---

!!! success "Installation Complete"
    You're ready to start computing convection onset in rotating spherical shells!
