module PoleVelocityChecks

using Test, Magrathea, LinearAlgebra

# Independently differentiated trigonometric interpolation, exact for the
# sectoral l=1,2 fields below. Endpoints deliberately retain sin(pi) roundoff.
function pole_grid(::Type{T}) where {T}
    r = T[.4, .6, .8, 1]
    θ = collect(range(zero(T), T(pi); length=5))
    s, c = sin.(θ), cos.(θ)
    V = hcat(ones(T, 5), c, s, cos.(2 .* θ), sin.(2 .* θ))
    Vθ = hcat(zeros(T, 5), -s, c, -2 .* sin.(2 .* θ), 2 .* cos.(2 .* θ))
    R = hcat(ones(T, 4), r, r.^2, r.^3)
    Rr = hcat(zeros(T, 4), ones(T, 4), 2 .* r, 3 .* r.^2)
    (; r, θ, s, c, Dr=Rr/R, Dθ=Vθ/V)
end

function reconstruct(g, P, Tor, m; costheta=nothing, l=max(1, abs(m)))
    potentials_to_velocity(P, Tor; Dr=g.Dr, Dθ=g.Dθ,
        Lθ=-l*(l+1)*Matrix{eltype(g.r)}(I, 5, 5),
        r=g.r, sintheta=g.s, m=m, costheta=costheta)
end

@testset "Axisymmetric pole reconstruction and zero fields" begin
    for T in (Float32, Float64, BigFloat)
        g = pole_grid(T)
        P = (g.r.^2 ./ 2) * g.c'
        Tor = (g.r.^3) * g.c'
        ur, uθ, uφ = reconstruct(g, P, Tor, 0)
        tol = 256eps(T)
        @test all(isfinite, ur) && all(isfinite, uθ) && all(isfinite, uφ)
        @test ur ≈ ones(T, 4) * g.c' atol=tol
        @test uθ ≈ -ones(T, 4) * g.s' atol=tol
        @test uφ ≈ (g.r.^2) * g.s' atol=tol
        @test all(eltype(u) === Complex{T} for u in (ur, uθ, uφ))

        # The operator API has always supported real axisymmetric inputs with
        # no angular-coordinate property, and must retain their real dtype.
        op = (; Dr=g.Dr, Dθ=g.Dθ, Lθ=-2Matrix{T}(I,5,5), r=g.r, m=0)
        wrapped = Magrathea.velocity_from_potentials(op, P, Tor)
        @test all(eltype(u) === T for u in wrapped)
        @test all(isapprox(a,b; atol=tol) for (a,b) in zip(wrapped,(ur,uθ,uφ)))

        Z = zeros(T, 2, 2)
        for m in (0, 1, -1, 2, -2)
            # There is intentionally no information distinguishing north and
            # south here: exact zero fields do not require that information.
            vel = potentials_to_velocity(Z, Z; Dr=Z, Dθ=Z, Lθ=Z,
                r=T[.5, 1], sintheta=zeros(T, 2), m=m)
            @test all(iszero, vel[1]) && all(iszero, vel[2]) && all(iszero, vel[3])
        end
    end
end

@testset "Signed first-order harmonics have physical polar limits" begin
    for T in (Float32, Float64, BigFloat), m in (-1, 1)
        g = pole_grid(T)
        P = (g.r.^2 ./ 2) * g.s'
        Tor = g.r.^3 * g.s'
        tol = 1024eps(T)
        expected = (ones(T,4)*g.s',
                    ones(T,4)*g.c' .+ (im*m .* g.r.^2)*ones(T,5)',
                    fill(Complex{T}(im*m),4,5) .- (g.r.^2)*g.c')
        for c in (nothing, g.c)
            vel = reconstruct(g, P, Tor, m; costheta=c)
            @test all(all(isfinite,u) for u in vel)
            @test all(isapprox(a,b; atol=tol, rtol=tol) for (a,b) in zip(vel,expected))
            @test all(eltype(u) === Complex{T} for u in vel)
        end
        # A pure poloidal field reconstructs the constant Cartesian vector
        # (1, i*m, 0), even though spherical components depend on φ at the poles.
        vel = reconstruct(g, P, zero(Tor), m)
        for j in (1,5), φ in T[0,.37,1.2]
            phase = cis(m*φ)
            ur, ut, up = (v[1,j]*phase for v in vel)
            x = ur*g.s[j]*cos(φ) + ut*g.c[j]*cos(φ) - up*sin(φ)
            y = ur*g.s[j]*sin(φ) + ut*g.c[j]*sin(φ) + up*cos(φ)
            z = ur*g.c[j] - ut*g.s[j]
            @test x ≈ one(T) atol=tol
            @test y ≈ Complex{T}(im*m) atol=tol
            @test abs(z) < tol
        end
        op = (; Dr=g.Dr, Dθ=g.Dθ, Lθ=-2Matrix{T}(I,5,5), r=g.r, theta=g.θ, m)
        vel = Magrathea.velocity_from_potentials(op,P,Tor)
        @test all(isapprox(a,b; atol=tol,rtol=tol) for (a,b) in zip(vel,expected))
        @test all(eltype(u) === Complex{T} for u in vel)
        # Retain operators exposing only precomputed reciprocal geometry.
        inv_r = inv.(g.r)
        op_inv = (; Dr=g.Dr, Dθ=g.Dθ, Lθ=op.Lθ, inv_r,
                    inv_r_sinθ=inv_r*inv.(g.s)', im_m=im*m)
        vel_inv = Magrathea.velocity_from_potentials(op_inv,P,Tor)
        @test all(isapprox(a,b; atol=tol,rtol=tol) for (a,b) in zip(vel_inv,expected))
    end
end

@testset "Mixed coordinate and field precision at poles" begin
    g32 = pole_grid(Float32)
    g = merge(g32, (;r=Float64.(g32.r), Dr=Float64.(g32.Dr)))
    P = (g.r.^2 ./ 2) * g.s'
    Tor = (g.r.^3) * g.s'
    expected = (ones(4)*g.s', ones(4)*g.c' .+ (im .* g.r.^2)*ones(5)',
        fill(im,4,5) .- (g.r.^2)*g.c')
    for c in (nothing,g.c), m in (-1,1)
        vel = reconstruct(g,P,Tor,m;costheta=c)
        target = m==1 ? expected : conj.(expected)
        @test all(all(isfinite,v) for v in vel)
        @test all(isapprox(a,b;rtol=2e-5,atol=2e-5) for (a,b) in zip(vel,target))
        @test all(eltype(v) === ComplexF64 for v in vel)
    end
    # The axisymmetric wrapper must still promote real field data when its
    # geometric factors have higher precision.
    op = (;Dr=g32.Dr,Dθ=g32.Dθ,Lθ=-2Matrix{Float32}(I,5,5),r=g.r,m=0)
    vel = Magrathea.velocity_from_potentials(op,Float32.(P),Float32.(Tor))
    @test all(eltype(v) === Float64 for v in vel)
end

@testset "Higher-order limits and irregular polar data" begin
    for T in (Float32, Float64), m in (-2, 2)
        g = pole_grid(T)
        P = (g.r.^2 ./ 2) * (g.s.^2)'
        Tor = (g.r.^3) * (g.s.^2)'
        vel = reconstruct(g, P, Tor, m)
        @test all(iszero, vcat((vec(v[:,[1,5]]) for v in vel)...))
        @test all(all(isfinite,u) for u in vel)
        @test vel[1][:,2:4] ≈ 3ones(T,4)*(g.s[2:4].^2)' rtol=256eps(T)
        @test vel[2][:,2:4] ≈ 2ones(T,4)*(g.s[2:4].*g.c[2:4])' .+
            (im*m .* g.r.^2)*g.s[2:4]' rtol=256eps(T)
        @test_throws ArgumentError reconstruct(g,P,ones(T,4,5),m)
        @test_throws ArgumentError reconstruct(g,P,(g.r.^3)*g.s',m)
    end
    g = pole_grid(Float64)
    P = (g.r.^2 ./ 2) * g.s'
    @test_throws DimensionMismatch reconstruct(g,P,zero(P),1;costheta=[1.])
    @test_throws ArgumentError reconstruct(g,P,zero(P),1;costheta=zeros(5))
    # Contradictory derivative data cannot determine which pole is present.
    D = copy(g.Dθ)
    D[1,:] .= 0
    D[1,2] = 1
    D[1,4] = -1
    s = copy(g.s); s[[1,5]] .= 0; s[4] = s[2]
    ga = merge(g, (;s, Dθ=D))
    Tor = g.r * (s .* g.c)'
    @test_throws ArgumentError reconstruct(ga,zero(P),Tor,1)
    @test all(all(isfinite,v) for v in reconstruct(ga,zero(P),Tor,1;costheta=g.c))

    # A genuinely off-axis point above the roundoff threshold still uses the
    # ordinary division, even close to either pole.
    s = [1e-7, .4, 1., .4, 1e-7]
    gn = merge(g,(;s))
    Pn = (g.r.^2 ./ 2)*s'
    Tn = (g.r.^3)*s'
    vel = reconstruct(gn,Pn,Tn,1)
    dr = gn.Dr*Pn
    expected = (2Pn ./ gn.r.^2,
        (dr*gn.Dθ') ./ gn.r .+ im.*Tn ./ (gn.r*s'),
        im.*dr ./ (gn.r*s') .- (Tn*gn.Dθ') ./ gn.r)
    @test all(isapprox(a,b;rtol=1e-12,atol=1e-12) for (a,b) in zip(vel,expected))
end

end # module
