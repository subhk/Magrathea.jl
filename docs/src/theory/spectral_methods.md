# Spectral Methods

<div class="magrathea-hero">
  <div class="magrathea-eyebrow">Theory</div>
  <h1>Sparse spectral discretization.</h1>
  <p>The Chebyshev and ultraspherical spectral methods Magrathea.jl uses for high accuracy at high sparsity.</p>
</div>

## Overview

Magrathea.jl combines two spectral approaches:

1. **Spherical Harmonics** for angular directions (``\theta, \phi``)
2. **Chebyshev/Ultraspherical Polynomials** for radial direction (``r``)

This choice provides:
- Spectral accuracy (exponential convergence for smooth solutions)
- Natural handling of spherical geometry
- Sparse operator matrices

## Chebyshev Spectral Method

### Chebyshev Polynomials

Chebyshev polynomials ``T_n(x)`` are defined on ``[-1, 1]`` by:

```math
T_n(\cos\theta) = \cos(n\theta)
```

With the recurrence relation:
```math
T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
```

And orthogonality:
```math
\int_{-1}^{1} \frac{T_m(x) T_n(x)}{\sqrt{1-x^2}} dx = \begin{cases} \pi & m = n = 0 \\ \pi/2 & m = n \neq 0 \\ 0 & m \neq n \end{cases}
```

### Collocation Points

Magrathea.jl uses Chebyshev-Gauss-Lobatto points:

```math
x_j = \cos\left(\frac{\pi j}{N-1}\right), \quad j = 0, 1, \ldots, N-1
```

These points cluster near the boundaries, providing enhanced resolution where boundary layers form.

### Domain Mapping

For a physical domain ``[r_i, r_o]``, we map from computational domain ``[-1, 1]``:

```math
r = \frac{r_o + r_i}{2} + \frac{r_o - r_i}{2} x
```

```math
\frac{d}{dr} = \frac{2}{r_o - r_i} \frac{d}{dx}
```

### Differentiation Matrices

The Chebyshev differentiation matrix ``D`` computes derivatives at collocation points:

```math
\left(\frac{df}{dx}\right)_j = \sum_{k=0}^{N-1} D_{jk} f_k
```

The matrix elements are:
```math
D_{jk} = \begin{cases}
\frac{c_j}{c_k} \frac{(-1)^{j+k}}{x_j - x_k} & j \neq k \\
-\frac{x_j}{2(1-x_j^2)} & 0 < j = k < N-1 \\
\frac{2(N-1)^2 + 1}{6} & j = k = 0 \\
-\frac{2(N-1)^2 + 1}{6} & j = k = N-1
\end{cases}
```

Where ``c_j = 2`` for ``j = 0, N-1`` and ``c_j = 1`` otherwise.

### ChebyshevDiffn Structure

```julia
struct ChebyshevDiffn{T<:AbstractFloat}
    n::Int              # Number of points
    domain::Tuple{T,T}  # Physical domain [a, b]
    max_order::Int      # Highest derivative order
    x::Vector{T}        # Collocation points
    D1::Matrix{T}       # First derivative matrix
    D2::Matrix{T}       # Second derivative matrix
    D3::Matrix{T}       # Third derivative matrix
    D4::Matrix{T}       # Fourth derivative matrix
end
```

Higher derivatives are computed by matrix multiplication: ``D_2 = D_1 \times D_1``, etc.

## Ultraspherical Spectral Method

The key innovation in Magrathea.jl is using the Olver-Townsend ultraspherical method for sparse operator construction.

### Gegenbauer Polynomials

Gegenbauer (ultraspherical) polynomials ``C_n^{(\lambda)}(x)`` generalize Chebyshev polynomials:
- ``C_n^{(0)}(x) = T_n(x)`` (Chebyshev of first kind)
- ``C_n^{(1/2)}(x) = P_n(x)`` (Legendre)
- ``C_n^{(1)}(x) = U_n(x)`` (Chebyshev of second kind)

Orthogonality:
```math
\int_{-1}^{1} C_m^{(\lambda)}(x) C_n^{(\lambda)}(x) (1-x^2)^{\lambda-1/2} dx = h_n^{(\lambda)} \delta_{mn}
```

### The Ultraspherical Chain

The fundamental insight: differentiation raises the ultraspherical index ``\lambda``:

```math
\frac{d}{dx} C_n^{(\lambda)}(x) = 2\lambda C_{n-1}^{(\lambda+1)}(x)
```

This means:
- First derivative: ``C^{(0)} \to C^{(1)}``
- Second derivative: ``C^{(0)} \to C^{(1)} \to C^{(2)}``
- And so on...

### Sparse Differentiation

The differentiation operator in coefficient space is **banded**:

```math
D^{(\lambda)}_{n,n'} = 2\lambda \delta_{n', n-1}
```

This is a superdiagonal matrix with just one nonzero diagonal!

### Sparse Conversion

The conversion operator ``S^{(\lambda)}`` transforms between bases:
```math
C_n^{(\lambda)} = \sum_{k} S^{(\lambda)}_{n,k} C_k^{(\lambda+1)}
```

This is also banded (tridiagonal).

### Sparse Multiplication

Multiplication by ``x`` in coefficient space:
```math
x C_n^{(\lambda)} = \alpha_{n-1}^{(\lambda)} C_{n-1}^{(\lambda)} + \alpha_n^{(\lambda)} C_{n+1}^{(\lambda)}
```

Where:
```math
\alpha_n^{(\lambda)} = \frac{n+2\lambda}{2(n+\lambda+1)}
```

Also tridiagonal!

### Radial Operator Construction

For operators like ``r^p \frac{d^n}{dr^n}``, Magrathea.jl:

1. Converts Chebyshev coefficients through the ultraspherical chain
2. Applies multiplication matrices for ``r^p``
3. Applies differentiation matrices
4. Results in a **sparse** matrix

```julia
# Example: r² d²/dr² operator
op = sparse_radial_operator(2, 2, N, ri, ro)
# Returns sparse matrix with ~O(p+n) bandwidth
```

### Sparsity Analysis

For an ``N \times N`` operator:

| Operation | Dense | Sparse | Bandwidth |
|-----------|-------|--------|-----------|
| ``d/dr`` | ``N^2`` | ``N`` | 1 |
| ``d^2/dr^2`` | ``N^2`` | ``2N`` | 2 |
| ``r \cdot d/dr`` | ``N^2`` | ``3N`` | 3 |
| ``r^2 d^2/dr^2`` | ``N^2`` | ``6N`` | 6 |

**Sparsity for ``N = 64``:**
- Dense: 4,096 entries
- Sparse: ~100-200 entries
- **Sparsity: 95-98%**

## Spherical Harmonics

### Definition

Spherical harmonics ``Y_\ell^m(\theta, \phi)`` are eigenfunctions of the angular Laplacian:

```math
Y_\ell^m(\theta, \phi) = \sqrt{\frac{2\ell+1}{4\pi} \frac{(\ell-m)!}{(\ell+m)!}} P_\ell^m(\cos\theta) e^{im\phi}
```

Where ``P_\ell^m`` are associated Legendre functions.

### Properties

**Orthonormality:**
```math
\int Y_\ell^m Y_{\ell'}^{m'*} d\Omega = \delta_{\ell\ell'} \delta_{mm'}
```

**Angular Laplacian:**
```math
\mathcal{L} Y_\ell^m = \ell(\ell+1) Y_\ell^m
```

## Boundary Condition Implementation

### Tau Method

Magrathea.jl uses the Tau method for boundary conditions:

1. Replace the last few rows of the operator matrix with BC constraints
2. The replaced rows correspond to highest polynomial degrees
3. Interior accuracy is preserved

For example, no-slip velocity BCs on poloidal potential:
- Replace row ``N-1`` with: ``P(r_i) = 0``
- Replace row ``N`` with: ``P(r_o) = 0``
- Replace row ``N+1`` with: ``P'(r_i) = 0``
- Replace row ``N+2`` with: ``P'(r_o) = 0``

### BC Matrix Form

Boundary condition evaluation at ``r = r_b``:

```math
P(r_b) = \sum_{n=0}^{N-1} a_n T_n(x_b) = \sum_{n=0}^{N-1} \mathcal{B}^{(0)}_n a_n
```

```math
P'(r_b) = \sum_{n=0}^{N-1} a_n T_n'(x_b) = \sum_{n=0}^{N-1} \mathcal{B}^{(1)}_n a_n
```

Where ``\mathcal{B}^{(k)}`` is the BC evaluation row for the ``k``-th derivative.

## Error Analysis

### Spectral Convergence

For smooth solutions, the error decreases exponentially with ``N``:

```math
\|u - u_N\| \sim e^{-\alpha N}
```

Where ``\alpha`` depends on solution smoothness.

### Resolution Guidelines

| Feature | Minimum Resolution |
|---------|-------------------|
| Smooth profiles | ``N \geq 16`` |
| Boundary layers | ``N \geq 32`` |
| Turbulent structures | ``N \geq 64`` |
| Low Ekman (``E < 10^{-6}``) | ``N \geq 96`` |

For spherical harmonics:
- ``\ell_{max} \geq m + 10`` for adequate mode resolution
- ``\ell_{max} \geq 3m`` for well-resolved patterns

## Implementation Details

### sparse_radial_operator Function

```julia
"""
Construct r^power * d^deriv_order/dr^deriv_order operator
using ultraspherical spectral method.

Returns sparse matrix in Chebyshev coefficient space.
"""
function sparse_radial_operator(power, deriv_order, N, ri, ro)
    # 1. Build ultraspherical differentiation chain
    D_chain = build_differentiation_chain(deriv_order, N)

    # 2. Build r^power multiplication operator
    M = build_multiplication_operator(power, N, ri, ro)

    # 3. Combine: M * D_chain
    # Result is sparse with bandwidth ~O(power + deriv_order)

    return sparse(M * D_chain)
end
```

### Memory Comparison

For a problem with ``\ell_{max} = 60``, ``N_r = 64``, ``m = 10``:

| Method | Matrix Storage | Assembly Time |
|--------|---------------|---------------|
| Dense | ~50 MB | ~10 s |
| Sparse (traditional) | ~5 MB | ~2 s |
| **Ultraspherical** | **~1 MB** | **~0.5 s** |

## Verification

### Manufactured Solutions

Magrathea.jl includes tests using manufactured solutions:

1. Choose a known solution ``u_{exact}(r)``
2. Compute ``f = \mathcal{L}[u_{exact}]`` analytically
3. Solve ``\mathcal{L}[u] = f`` numerically
4. Compare ``u`` to ``u_{exact}``

### Convergence Tests

```julia
# Test spectral convergence
errors = Float64[]
for N in [16, 32, 64, 128]
    u_num = solve_problem(N)
    push!(errors, norm(u_num - u_exact))
end

# Should see exponential decrease
```

---

## References

1. Olver, S. and Townsend, A. (2013). *A fast and well-conditioned spectral method*. SIAM Review.

2. Boyd, J.P. (2001). *Chebyshev and Fourier Spectral Methods*. Dover.

3. Trefethen, L.N. (2000). *Spectral Methods in MATLAB*. SIAM.

4. Glatzmaier, G.A. (2014). *Introduction to Modeling Convection in Planets and Stars*. Princeton University Press.
