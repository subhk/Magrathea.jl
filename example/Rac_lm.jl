using LinearAlgebra, SparseArrays, Arpack

# critical Ra for rotating spherical shell, fixed m
function critical_Ra_shell(m;
        N=40, lmax=80,
        E=1e-4, Pr=1.0,
        ri=0.35, ro=1.0)

    # radial grid and diff. matrices on [ri,ro]
    D0, x0 = cheb(N)
    # map x∈[-1,1] → r∈[ri,ro]
    r = (ro+ri)/2 .+ ((ro-ri)/2)*x0
    D = (2/(ro-ri))*D0
    D2 = D*D
    Np = N+1
    Iden = I(Np)

    # helper for the angular Laplacian term
    function LTT(ℓ)
        # build (D2 - ℓ(ℓ+1)/r^2)
        return D2 .- Diagonal(ℓ*(ℓ+1) ./ (r.^2))
    end

    best = (ℓ=0, Ra=Inf)
    # loop over spherical‐harmonic degree
    for ℓ in m:lmax
        # 3×3 blocks of A0*X + Ra*B*X = 0
        L = LTT(ℓ)
        C = Diagonal(2im*m ./ (r.^2))    # Coriolis coupling
        LTP = ℓ*(ℓ+1)

        # build A0 blocks
        A11 =           L               # T‐eqn
        A12 =                C
        A21 = C
        A22 = E*(L*L)        # P‐eqn
        A23 = zeros(Np,Np)
        A31 = Iden           # Θ‐eqn
        A32 = zeros(Np,Np)
        A33 =           L

        # assemble into big matrices
        A0 = [  A22   A21   A23;
                A12   A11   zeros(Np,Np);
                A31   zeros(Np,Np)  A33 ]
        B  = spzeros(3Np, 3Np)
        # only P‐eqn (block A22) carries Ra coupling via Θ:
        B[1:Np, 2Np+1:3Np] .= -LTP*Iden

        # enforce BCs via tau‐rows
        # stress‐free: P=0 & P''=0  at r=ri,r0:
        #   replace rows 1,2 and rows Np-1,Np of A0 and B
        function tau!(M)
            # P(ri)=0
            M[1, :] .= 0;    M[1, 1] = 1
            # P''(ri)=0  →  D2 row
            M[2, :] .= D2[1, :]
            # P''(ro)=0
            M[Np-1, :] .= D2[end, :]
            # P(ro)=0
            M[Np,   :] .= 0;  M[Np, end] = 1
            # T=0 at r=ri,ro
            M[Np+1, :] .= 0;       M[Np+1, Np+1] = 1
            M[2Np,  :] .= 0;       M[2Np, 2Np]    = 1
            # θ=0 at r=ri,ro
            M[2Np+1,:] .= 0;  M[2Np+1,2Np+1] = 1
            M[3Np,  :] .= 0;  M[3Np,  3Np]   = 1
        end

        tau!(A0);  tau!(B)   # impose on both

        # solve generalized eigenproblem for smallest Ra
        vals, _ = eigs(A0, B; nev=1, which=:SR)
        Ra_ℓ = real(vals[1])
        if Ra_ℓ < best.Ra
            best = (ℓ=ℓ, Ra=Ra_ℓ)
        end
    end

    return best  # (ℓ_crit, Ra_c)
end

# Example: critical Ra for m=6, Ekman=1e-4
ℓc, Rac = critical_Ra_shell(6; N=50, lmax=100, E=1e-4, Pr=1.0, ri=0.7, ro=1.0)
println("m=6 → ℓ₍c₎=$(ℓc),  Ra₍c₎=$(Rac)")
