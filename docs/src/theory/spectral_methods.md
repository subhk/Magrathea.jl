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

The hydrodynamic onset, biglobal and triglobal solvers use Chebyshev collocation
on `Nr` Gauss–Lobatto points. The MHD solver uses ultraspherical coefficient
operators, with `N` the maximum Chebyshev degree (`N + 1` coefficients).

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
`ChebyshevDiffn` stores them in ascending order, so `cd.x[1]` is the inner wall.

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

See [`ChebyshevDiffn`](@ref) for the current fields. `x` contains the radial
nodes; `D1` through `D4` contain collocation differentiation matrices when
requested by the constructor.

The implementation differentiates Chebyshev coefficients and transforms back
to values; see `src/Spectral/chebyshev.jl` for the construction.

## Ultraspherical Spectral Method

The MHD solver, and the coefficient-space `SparseStabilityOperator`, use the
Olver–Townsend ultraspherical method for sparse operator construction.

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
\frac{d}{dx} C_n^{(\lambda)}(x) = 2\lambda C_{n-1}^{(\lambda+1)}(x), \quad \lambda > 0,
\qquad \frac{d}{dx} T_n(x) = n\, C_{n-1}^{(1)}(x)
```

This means:
- First derivative: ``C^{(0)} \to C^{(1)}``
- Second derivative: ``C^{(0)} \to C^{(1)} \to C^{(2)}``
- And so on...

### Sparse Differentiation

The differentiation operator in coefficient space is **banded**. It maps
``C^{(\lambda)}`` coefficients ``a`` to ``C^{(\lambda+1)}`` coefficients
``b_n = D^{(\lambda)}_{n,n+1} a_{n+1}``:

```math
D^{(\lambda)}_{n,n'} = 2\lambda\, \delta_{n', n+1} \quad (\lambda > 0), \qquad
D^{(0)}_{n,n'} = (n+1)\, \delta_{n', n+1}
```

This is a superdiagonal matrix with just one nonzero diagonal!

### Sparse Conversion

The conversion operator ``S^{(\lambda)}`` transforms between bases:
```math
C_n^{(\lambda)} = \frac{\lambda}{n+\lambda}\left(C_n^{(\lambda+1)} - C_{n-2}^{(\lambda+1)}\right),
\qquad T_n = \tfrac{1}{2}\left(C_n^{(1)} - C_{n-2}^{(1)}\right)\ (n \ge 1)
```

Its matrix is banded, with nonzeros on the main diagonal and the second superdiagonal.

### Sparse Multiplication

Multiplication by ``x`` follows the three-term recurrence (``\lambda > 0``):
```math
x\, C_n^{(\lambda)} = \frac{n+2\lambda-1}{2(n+\lambda)}\, C_{n-1}^{(\lambda)} + \frac{n+1}{2(n+\lambda)}\, C_{n+1}^{(\lambda)}
```

with ``x T_n = (T_{n-1} + T_{n+1})/2`` (``n \ge 1``) for Chebyshev polynomials. Multiplication
by a polynomial of degree ``p``, such as ``r^p``, is therefore banded with
bandwidth ``p`` (`Magrathea.multiplication_matrix`).

### Radial Operator Construction

For an operator ``r^p \frac{d^n}{dr^n}`` acting on Chebyshev coefficients,
Magrathea.jl:

1. Applies ``n`` differentiation matrices, moving from ``C^{(0)}`` to ``C^{(n)}``
2. Multiplies by ``r^p`` in the ``C^{(n)}`` basis
3. Converts up to the residual basis of the equation
4. Results in a **banded** matrix

```julia
# Example: r² d²/dr² from Chebyshev coefficients to C^(2) coefficients
op = Magrathea.banded_radial_term(Float64, 2, 2, 2, N, ri, ro)
```

The energy-conserving MHD Galerkin assembly uses these banded terms. The tau
assembly stores each block as a Chebyshev-to-Chebyshev operator,
`Magrathea.sparse_radial_operator(p, n, N, ri, ro)`. Its derivative part is upper
triangular rather than banded, because it converts back from ``C^{(n)}``. The
assembly then multiplies every fluid row block by the conversion chain to
``C^{(4)}`` (poloidal velocity) or ``C^{(2)}`` (the other fields).

### Sparsity Analysis

Nonzeros of `banded_radial_term` for ``N = 64`` (a ``65 \times 65`` matrix, 4,225
entries), mapping to ``C^{(n)}``:

| Operation | Stored nonzeros | Nonzero diagonals |
|-----------|-----------------|-------------------|
| ``d/dr`` | 64 | 1 |
| ``d^2/dr^2`` | 63 | 1 |
| ``r \cdot d/dr`` | 191 | 3 |
| ``r^2 d^2/dr^2`` | 312 | 5 |
| ``r^4 d^4/dr^4`` | 539 | 9 |

These are 87–98.5% sparse.

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

### Boundary enforcement paths

Hydrodynamic collocation replaces boundary equations at the endpoint-related
rows, then reduces the pencil with a basis satisfying those constraints.
MHD with insulating or perfectly conducting magnetic walls, for any background
field, uses an energy-conserving Galerkin assembly: boundary-recombined trial
bases, with each equation tested against its own trial basis. A finite-conductivity
core or mantle uses coefficient-tau constraints: the highest residual rows are
replaced, while all unknown coefficient columns remain.
The tau poloidal velocity residual is in ``C^{(4)}`` and the other residuals are
in ``C^{(2)}``; these are equation coefficients, not radial grid points.

No-slip constrains the poloidal potential and its derivative at both walls,
and the toroidal potential at both walls. Exact row locations and derivative
scalings depend on the representation. See the
[boundary equations](mathematical_foundations.md#Boundary-Conditions) and
[source map](../codebase_structure.md) for the corresponding implementations.

### BC Matrix Form

Boundary condition evaluation at ``r = r_b`` for a coefficient expansion of
maximum degree ``N``:

```math
P(r_b) = \sum_{n=0}^{N} a_n T_n(x_b) = \sum_{n=0}^{N} \mathcal{B}^{(0)}_n a_n
```

```math
P'(r_b) = \sum_{n=0}^{N} a_n \frac{2}{r_o-r_i}T_n'(x_b) = \sum_{n=0}^{N} \mathcal{B}^{(1)}_n a_n
```

Where ``\mathcal{B}^{(k)}`` is the BC evaluation row for the ``k``-th derivative.
At the walls ``x_b = \pm 1``, ``T_n(\pm1) = (\pm1)^n`` and
``T_n'(\pm1) = (\pm1)^{n+1} n^2``.

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

### Radial operator functions

Both functions live in `src/Spectral/` and are internal (not exported):

```julia
# src/Spectral/galerkin.jl: Chebyshev coefficients -> C^(q_out) coefficients, banded
Magrathea.banded_radial_term(T, power, deriv, q_out, N, ri, ro)

# src/Spectral/ultraspherical.jl: Chebyshev -> Chebyshev coefficients, used by MHDStabilityOperator
Magrathea.sparse_radial_operator(power, deriv, N, ri, ro)
```

`banded_radial_term` composes `ultraspherical_derivative`, `multiplication_matrix`
and `ultraspherical_conversion` without any back-solve. `sparse_radial_operator`
applies the same derivative chain, solves with the conversion chain to return to the
Chebyshev basis, and then multiplies by ``r^p``. On resolved inputs, its output
converted up to ``C^{(n)}`` agrees with `banded_radial_term`
(`test/galerkin_radial.jl`).

### Memory Comparison

`estimate_size(problem)` prints the matrix size and a dense-storage estimate before
solving. As a measured example, the MHD tau pencil for `lmax = 15`, `N = 32`,
`m = 2` with an axial field has ``n = 1155``. Its `A` stores about 87,000 nonzeros
(6.5% of ``n^2``), roughly 2 MB, compared with 21 MB for one dense complex matrix.

## Verification

### Manufactured Solutions

Magrathea.jl includes tests using manufactured solutions (for example, the thermal
boundary values in `test/boundary_physics.jl` and the mean flows in
`test/nonlinear_mean_flow.jl`):

1. Choose a known solution ``u_{exact}(r)``
2. Compute ``f = \mathcal{L}[u_{exact}]`` analytically
3. Solve ``\mathcal{L}[u] = f`` numerically
4. Compare ``u`` to ``u_{exact}``

`test/galerkin_radial.jl` also compares Galerkin radial operators with analytic
Laplacian and beam spectra.

### Convergence Tests

```julia
using Magrathea

# Spectral convergence of the collocation derivative on r ∈ [0.35, 1]
f(r) = exp(r) * sin(3r)
df(r) = exp(r) * (sin(3r) + 3cos(3r))
for Nr in (8, 12, 16, 24)
    cd = ChebyshevDiffn(Nr, [0.35, 1.0], 1)
    err = maximum(abs, cd.D1 * f.(cd.x) .- df.(cd.x))
    println("Nr = $Nr: max error = $err")
end
```

The error falls from about ``10^{-5}`` at `Nr = 8` to round-off (about
``10^{-13}``) by `Nr = 16`.

---

## References

1. Olver, S. and Townsend, A. (2013). *A fast and well-conditioned spectral method*. SIAM Review.

2. Boyd, J.P. (2001). *Chebyshev and Fourier Spectral Methods*. Dover.

3. Trefethen, L.N. (2000). *Spectral Methods in MATLAB*. SIAM.

4. Glatzmaier, G.A. (2014). *Introduction to Modeling Convection in Planets and Stars*. Princeton University Press.
