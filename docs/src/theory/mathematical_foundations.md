# Mathematical Foundations

<div class="magrathea-hero">
  <div class="magrathea-eyebrow">Theory</div>
  <h1>The equations behind Magrathea.jl.</h1>
  <p>The mathematical framework: governing equations, non-dimensionalization, and the eigenvalue-problem formulation.</p>
</div>

## Governing Equations

### Boussinesq Convection in Rotating Shells

Magrathea.jl solves the linearized equations for thermal convection in a rotating spherical shell under the Boussinesq approximation. The dimensional equations are:

**Momentum (Navier-Stokes):**
```math
\rho_0 \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) + 2\rho_0 \boldsymbol{\Omega} \times \mathbf{u} = -\nabla p + \rho_0 \nu \nabla^2 \mathbf{u} + \rho \mathbf{g}
```

**Continuity (incompressibility):**
```math
\nabla \cdot \mathbf{u} = 0
```

**Energy:**
```math
\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \kappa \nabla^2 T
```

**Equation of State (Boussinesq):**
```math
\rho = \rho_0 \left[ 1 - \alpha (T - T_0) \right]
```

Where:
- ``\mathbf{u}`` = velocity field
- ``p`` = pressure
- ``T`` = temperature
- ``\rho_0`` = reference density
- ``\nu`` = kinematic viscosity
- ``\kappa`` = thermal diffusivity
- ``\alpha`` = thermal expansion coefficient
- ``\boldsymbol{\Omega} = \Omega \hat{\mathbf{z}}`` = rotation vector
- ``\mathbf{g} = -g \hat{\mathbf{r}}`` = gravity (radially inward)

### Non-Dimensionalization

The radial operators are built on ``r \in [\chi, 1]``, i.e. the **outer radius ``r_o`` is the length unit**. We non-dimensionalize using:

| Quantity | Scale | Description |
|----------|-------|-------------|
| Length | ``r_o`` | Outer radius |
| Time | ``1/\Omega`` | Rotation period |
| Velocity | ``\Omega r_o`` | Rotational velocity |
| Temperature | ``\Delta T`` | Temperature contrast |
| Pressure | ``\rho_0 \Omega^2 r_o^2`` | Rotational pressure |

This yields the dimensionless parameters:

| Parameter | Definition | Physical Meaning |
|-----------|------------|------------------|
| Ekman number | ``E = \nu / (\Omega r_o^2)`` | Viscous/Coriolis forces (``r_o``-based) |
| Prandtl number | ``Pr = \nu / \kappa`` | Momentum/thermal diffusivity |
| Rayleigh number | ``Ra = \alpha g \Delta T\, (r_o - r_i)^3 / (\nu \kappa)`` | Buoyancy forcing (shell-thickness based; see note) |
| Radius ratio | ``\chi = r_i / r_o`` | Geometric parameter |

!!! warning "Length-scale convention (read before benchmarking)"
    Magrathea.jl mixes two length scales, which is the single most common source of
    confusion when comparing against the literature:

    - The **Ekman number is outer-radius (``r_o``) based**: ``E = \nu/(\Omega r_o^2)``.
      This is the value you pass and the value used directly in the operators.
    - The **Rayleigh number you pass is shell-thickness (``d = r_o - r_i``) based**:
      ``Ra = \alpha g \Delta T\, d^3/(\nu\kappa)``. It is converted internally to the
      ``r_o``-based value the operators need, ``Ra_{\text{int}} = Ra/(1-\chi)^3``.

    Much rotating-convection literature (e.g. Barik et al. 2023 and the **Kore**
    code Magrathea.jl is ported from, the weakly-nonlinear onset analyses, the
    geodynamo benchmark) non-dimensionalizes **both** ``E`` and ``Ra`` by the shell
    thickness ``d``, defining ``Ek_d = \nu/(\Omega d^2) = E/(1-\chi)^2``.

    **To reproduce a ``d``-based literature value at radius ratio ``\chi``:**

    | Their quantity | Pass to Magrathea.jl |
    |----------------|------------------|
    | ``Ek_d`` (their Ekman) | `E = Ek_d * (1 - χ)^2` |
    | ``Ra_d`` (their Rayleigh) | `Ra = Ra_d` (already ``d``-based) |

    The returned critical value `Ra_c` is the ``d``-based Rayleigh number; the
    *modified* Rayleigh number commonly tabulated is ``\widetilde{Ra} = Ra_c \cdot Ek_d``.
    This mapping is validated to four significant figures below.

### Non-Dimensional Equations

For linearization about the conductive state, the implemented gravity profile
is proportional to radius, ``g(r)=g_o r/r_o``. Using the public shell-thickness
Rayleigh number gives:

**Momentum:**
```math
\frac{\partial \mathbf{u}}{\partial t} + 2 \hat{\mathbf{z}} \times \mathbf{u} = -\nabla p + E \nabla^2 \mathbf{u} + \frac{Ra \cdot E^2}{Pr(1-\chi)^3} r\Theta \hat{\mathbf{r}}
```

**Continuity:**
```math
\nabla \cdot \mathbf{u} = 0
```

**Energy:**
```math
\frac{\partial \Theta}{\partial t} + \mathbf{u} \cdot \nabla T_0 = \frac{E}{Pr} \nabla^2 \Theta
```

Where ``T = T_0(r) + \Theta(r, \theta, \phi, t)`` and ``T_0`` is the basic state temperature profile.

## Toroidal-Poloidal Decomposition

Since ``\nabla \cdot \mathbf{u} = 0``, the velocity can be decomposed:

```math
\mathbf{u} = \nabla \times \nabla \times (P \hat{\mathbf{r}}) + \nabla \times (T \hat{\mathbf{r}})
```

Where:
- ``P(r, \theta, \phi, t)`` = poloidal scalar
- ``T(r, \theta, \phi, t)`` = toroidal scalar

### Velocity Components

In spherical coordinates ``(r, \theta, \phi)``:

```math
u_r = \frac{1}{r^2} \mathcal{L} P
```

```math
u_\theta = \frac{1}{r} \frac{\partial^2 P}{\partial r \partial \theta} + \frac{1}{r \sin\theta} \frac{\partial T}{\partial \phi}
```

```math
u_\phi = \frac{1}{r \sin\theta} \frac{\partial^2 P}{\partial r \partial \phi} - \frac{1}{r} \frac{\partial T}{\partial \theta}
```

Where ``\mathcal{L}`` is the angular Laplacian:
```math
\mathcal{L} = -\frac{1}{\sin\theta} \frac{\partial}{\partial \theta} \left( \sin\theta \frac{\partial}{\partial \theta} \right) - \frac{1}{\sin^2\theta} \frac{\partial^2}{\partial \phi^2}
```

### Coordinates stored by the stability solver

The formulas above describe the physical potentials accepted by
`potentials_to_velocity`. The onset matrices store rescaled potentials
``P_{\rm physical}=rP_{\rm onset}``, ``T_{\rm physical}=rT_{\rm onset}``, and use
angular basis ``Y_{\ell m}/\sqrt{2\ell+1}``, where ``Y`` is complex orthonormal.
Thus one onset poloidal mode gives
``u_r=\ell(\ell+1)P_{\rm onset}Y/(r\sqrt{2\ell+1})``.
`eigenvector_to_velocity` and the triglobal reconstruction apply these conventions
automatically and evaluate angular derivatives analytically.

The native mean-flow potentials use a different convention:
``p=rP_{\rm onset}/\sqrt{2\ell+1}`` and
``t=-rT_{\rm onset}/\sqrt{2\ell+1}`` after the real/complex phase conversion.
The minus sign comes from using ``\hat r\times\nabla_hY`` for the native toroidal
basis. The coupling assembly converts the physical fields before projection.

## Spherical Harmonic Expansion

Fields are expanded in spherical harmonics:

```math
\psi(r, \theta, \phi, t) = \sum_{\ell=0}^{L} \sum_{m=-\ell}^{\ell} \psi_\ell^m(r, t) Y_\ell^m(\theta, \phi)
```

Where ``Y_\ell^m`` are the spherical harmonics satisfying:
```math
\mathcal{L} Y_\ell^m = \ell(\ell+1) Y_\ell^m
```

### Single Mode Analysis

For stability analysis, we consider perturbations with a single azimuthal wavenumber ``m``:
```math
\psi(r, \theta, \phi, t) = e^{im\phi} e^{\sigma t} \sum_\ell \psi_\ell(r) Y_\ell^m(\theta, 0)
```

The eigenvalue ``\sigma = \sigma_r + i\omega`` determines:
- ``\sigma_r > 0``: unstable (growing perturbation)
- ``\sigma_r = 0``: marginally stable (onset of convection)
- ``\sigma_r < 0``: stable (decaying perturbation)
- ``\omega``: drift frequency (pattern rotation rate)

## Linearized Equations in Spectral Form

Taking the curl of the momentum equation eliminates pressure. The linearized equations for each ``(\ell, m)`` mode become:

### Poloidal Equation (2× curl of momentum)

```math
\sigma \nabla^2 \mathcal{L} P_\ell = E \nabla^4 \mathcal{L} P_\ell - 2im \frac{\partial}{\partial z}(z \cdot \nabla^2 P_\ell) + C_\ell^{(PT)} T_{\ell\pm1} + \frac{Ra \cdot E^2}{Pr} \ell(\ell+1) \Theta_\ell
```

### Toroidal Equation (1× curl of momentum)

```math
\sigma \mathcal{L} T_\ell = E \nabla^2 \mathcal{L} T_\ell + 2im \frac{\partial}{\partial z}(z \cdot T_\ell) + C_\ell^{(TP)} P_{\ell\pm1}
```

### Temperature Equation

```math
\sigma \Theta_\ell = \frac{E}{Pr} \nabla^2 \Theta_\ell - u_r \frac{dT_0}{dr}
```

Where the Coriolis coupling terms ``C_\ell^{(PT)}`` and ``C_\ell^{(TP)}`` couple modes with ``\Delta\ell = \pm 1``.

## Eigenvalue Problem

The spectral equations lead to the generalized eigenvalue problem:

```math
\mathbf{A} \mathbf{x} = \sigma \mathbf{B} \mathbf{x}
```

Where:
- ``\mathbf{x} = [P_\ell(r_j), T_\ell(r_j), \Theta_\ell(r_j)]_{j,\ell}`` = state vector
- ``\mathbf{A}`` = spatial operator (Coriolis, diffusion, buoyancy)
- ``\mathbf{B}`` = mass matrix (time derivative weighting)
- ``\sigma`` = complex eigenvalue

### Matrix Structure

For a single azimuthal mode ``m``, the matrices have block structure:

```math
\mathbf{A} = \begin{pmatrix}
A^{PP} & A^{PT} & A^{P\Theta} \\
A^{TP} & A^{TT} & 0 \\
A^{\Theta P} & 0 & A^{\Theta\Theta}
\end{pmatrix}
```

Where:
- ``A^{PP}``: Poloidal diffusion + Coriolis diagonal
- ``A^{PT}, A^{TP}``: Coriolis coupling (off-diagonal in ``\ell``)
- ``A^{P\Theta}``: Buoyancy coupling
- ``A^{\Theta P}``: Temperature advection

## Boundary Conditions

### Mechanical Conditions

Both walls are impermeable. No-slip also sets both tangential velocities to zero;
stress-free sets both tangential viscous tractions to zero. The radial formulas
must use the potential convention of the solver:

| Potentials | No-slip at each wall | Stress-free at each wall |
|---|---|---|
| Onset/MHD, with ``rP`` and ``rT`` inside the curls | ``P=P'=T=0`` | ``P=P''=0``, ``T'-T/r=0`` |
| Native mean flow, with unit-vector potentials ``p,t`` | ``p=p'=t=0`` | ``p=0``, ``p''-2p'/r=0``, ``t'-2t/r=0`` |

Primes denote physical radial derivatives. For example, the onset poloidal
radial velocity is ``\ell(\ell+1)P/r`` and its tangential strain amplitude is
``P''+[\ell(\ell+1)-2]P/r^2``. Impermeability reduces the latter to ``P''``.
The additional radius factors in the mean-flow table follow from ``p=rP``
(up to angular normalization). These conditions agree with the
[standard spherical mechanical conditions](https://magic-sph.github.io/numerics.html#mechanical-boundary-conditions)
after converting potentials.

### Thermal Conditions

Perturbations satisfy ``\Theta=0`` for fixed temperature, or ``\Theta'=0`` for
fixed flux. Nonzero `flux` values in the boundary APIs mean the **radial temperature
derivative**, at either wall. Physical outward heat flux is
``q_n=-\kappa n_r\Theta'``, with ``n_r=-1`` at the inner wall and ``n_r=+1`` at the
outer wall. Thus a negative outer `flux` corresponds to positive outward heat loss.

The steady `solve_poisson_mode` helper rejects degree-zero problems with flux
specified at both walls: these require compatible forcing/fluxes and a temperature
gauge. Higher-degree modes can have two flux conditions.

### Magnetic Conditions

For the MHD convention ``\mathbf b=\nabla\times\nabla\times(rf\hat r)+\nabla\times(rg\hat r)``:

| Wall model | Radial conditions |
|---|---|
| Insulating inner cavity | ``rf'-\ell f=0``, ``g=0`` |
| Insulating exterior | ``rf'+(\ell+1)f=0``, ``g=0`` |
| Stationary perfect conductor, no-slip fluid | ``f=0``, ``g'+g/r=0`` |
| Equal-diffusivity/permeability stationary conducting core, no-slip fluid | Continuous ``f,f',g,g'`` with diffusion solved in the core |

When the fluid slips, use the full tangential electric field
``\mathbf E_t=[E_m\nabla\times\mathbf b-\mathbf u\times\mathbf B_0]_t``:
it vanishes at a perfect conductor and is continuous at a conducting-core
interface. The toroidal wall equation includes this motional EMF; the other
component follows from the poloidal induction equation and converges with radial
resolution. See the [MHD guide](../mhd_user_guide.md#Magnetic-Boundary-Conditions)
for the core assumptions and supported options.

`test/boundary_physics.jl` independently evaluates polynomial derivatives,
physical velocity/strain, and electric fields of computed eigenmodes. It checks
mixed mechanical/thermal walls, both radial orderings, 2D/3D mean flows,
insulating matching, and radial convergence of the retained magnetic boundary
harmonics. Conditions on unretained harmonics require angular convergence too.

## Critical Rayleigh Number

The critical Rayleigh number ``Ra_c`` is defined as the value where the leading eigenvalue has zero growth rate:

```math
Ra_c = \min_{m, \ell} \{ Ra : \max(\text{Re}(\sigma)) = 0 \}
```

At onset:
- Convection first appears at ``(Ra_c, m_c)``
- The critical frequency ``\omega_c = \text{Im}(\sigma)`` gives the drift rate
- For rotating convection, typically ``\omega_c > 0`` (prograde drift)

## MHD Extension

For MHD problems, the magnetic field is similarly decomposed:

```math
\mathbf{B} = \nabla \times \nabla \times (rf \hat{\mathbf{r}}) + \nabla \times (rg \hat{\mathbf{r}})
```

Adding the Lorentz force to momentum:
```math
\mathbf{F}_{Lorentz} = Le^2 (\nabla \times \mathbf{B}) \times \mathbf{B}_0
```

And the induction equation:
```math
\frac{\partial \mathbf{B}}{\partial t} = \nabla \times (\mathbf{u} \times \mathbf{B}_0) + E_m \nabla^2 \mathbf{B}
```

This adds two magnetic field variables: ``(P, T, f, g, \Theta)``.

---

## References

1. Christensen, U.R. and Wicht, J. (2015). *Numerical Dynamo Simulations*. Treatise on Geophysics, Vol. 8.

2. Zhang, K. and Liao, X. (2017). *Theory and Modeling of Rotating Fluids*. Cambridge University Press.

3. Jones, C.A. (2011). *Planetary Magnetic Fields and Fluid Dynamos*. Annual Review of Fluid Mechanics.

4. Dormy, E. and Soward, A.M. (2007). *Mathematical Aspects of Natural Dynamos*. CRC Press.
