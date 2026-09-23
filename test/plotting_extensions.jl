using Test
using LinearAlgebra
using Logging
using Magrathea

# The plotting integrations are package extensions triggered by optional
# weak dependencies. Each block below is skipped when its trigger package
# (Makie, RecipesBase) is not installed in the active environment. `import`
# loads the extension without bringing the packages' exports into scope.
plotting_has_makie = Base.find_package("Makie") !== nothing
plotting_has_recipesbase = Base.find_package("RecipesBase") !== nothing
plotting_has_makie && @eval import Makie
plotting_has_recipesbase && @eval import RecipesBase

# Small results for each eigenvector layout, built with dense eigensolves so no
# PETSc/SLEPc installation is needed.
function _plotting_leading(values, nev)
    keep = findall(isfinite, values)
    return keep[sortperm(real.(values[keep]); rev=true)[1:min(nev, length(keep))]]
end

function _plotting_onset_result()
    params = OnsetParams(E=1e-2, Pr=1.0, Ra=2e4, χ=0.35, m=2, lmax=6, Nr=16)
    op = LinearStabilityOperator(params)
    A, B, idofs, bdofs = assemble_matrices(op)
    Ar, Br, reduction = Magrathea._constrained_reduced_matrices(A, B, op, idofs, bdofs)
    F = eigen(Ar, Br)
    sel = _plotting_leading(F.values, 3)
    vecs = reduce(hcat, [Magrathea._reconstruct_full_vector(reduction, F.vectors[:, j]) for j in sel])
    return StabilityResult(F.values[sel], vecs, OnsetProblem(params); extra=(operator=op,))
end

function _plotting_mhd_result(B0_type)
    field = B0_type != no_field
    params = MHDParams(E=1e-2, Pr=1.0, Pm=1.0, Ra=100.0, Le=field ? 0.01 : 0.0,
                       B0_amplitude=field ? 1.0 : 0.0, ricb=0.35, m=2, lmax=6, N=16,
                       B0_type=B0_type)
    return solve(MHDProblem(params); nev=3, backend=:dense)
end

function _plotting_triglobal_result()
    params = OnsetParams(E=1e-2, Pr=1.0, Ra=2e4, χ=0.35, m=0, lmax=4, Nr=16)
    bs3d = basic_state(params; mode=:nonaxisymmetric, amplitude=0.01, mmax_bs=1, lmax_bs=3)
    problem = TriglobalProblem(params, bs3d, -1:1)
    tparams = TriglobalParams(E=params.E, Pr=params.Pr, Ra=params.Ra, χ=params.χ,
                              m_range=-1:1, lmax=params.lmax, Nr=params.Nr,
                              basic_state_3d=bs3d)
    coupled = setup_coupled_mode_problem(tparams)
    single = Magrathea.build_single_mode_operators(coupled, false)
    coupling = Magrathea.build_mode_coupling_operators(coupled, single, false)
    A, B = Magrathea.assemble_block_matrices(coupled, single, coupling, false)
    F = eigen(Matrix(A), Matrix(B))
    sel = _plotting_leading(F.values, 3)
    result = StabilityResult(F.values[sel], F.vectors[:, sel], problem;
                             extra=(coupled_modes=collect(-1:1),))
    return result, coupled
end

if plotting_has_makie || plotting_has_recipesbase
    plotting_onset = _plotting_onset_result()
    # Silence operator-construction logs and the negative-m validation warning.
    plotting_mhd, plotting_mhd_nofield, (plotting_tri, plotting_coupled) =
        with_logger(NullLogger()) do
            (_plotting_mhd_result(axial), _plotting_mhd_result(no_field),
             _plotting_triglobal_result())
        end
end

makie_ext = plotting_has_makie ? Base.get_extension(Magrathea, :MagratheaMakieExt) : nothing
if makie_ext === nothing
    @info "Makie extension not loaded; skipping Makie plotting tests"
else
    @testset "Makie extension: radial profiles follow each result layout" begin
        ext = makie_ext
        res = plotting_onset
        op = res.extra.operator
        evec = res.eigenvectors[:, 1]
        P, Tor, Θ = Magrathea.extract_eigenvector_coefficients(evec, op)
        for (field, coeffs) in ((:poloidal, P), (:toroidal, Tor), (:temperature, Θ))
            r_grid, profiles = ext._ext_radial_profiles(ext._ext_layout(res), evec, field)
            @test r_grid == op.r
            @test [name for (name, _) in profiles] == ["l=$l" for l in sort(collect(keys(coeffs)))]
            @test all(profile == coeffs[l] for ((_, profile), l) in
                      zip(profiles, sort(collect(keys(coeffs)))))
        end
        @test_throws ArgumentError ext._ext_radial_profiles(op, evec, :magnetic_poloidal)

        # MHD: Chebyshev coefficient blocks evaluated on the reconstruction grid
        res = plotting_mhd
        op = res.extra.operator
        evec = res.eigenvectors[:, 1]
        idx = Magrathea._mhd_index_map(op)
        rg = Magrathea._mhd_radial_grid(op)
        for (field, section, ls) in ((:poloidal, :u, op.ll_u), (:toroidal, :v, op.ll_v),
                                     (:temperature, :h, op.ll_h),
                                     (:magnetic_poloidal, :f, op.ll_f),
                                     (:magnetic_toroidal, :g, op.ll_g))
            r_grid, profiles = ext._ext_radial_profiles(op, evec, field)
            @test r_grid == rg
            @test length(profiles) == length(ls)
            for ((name, profile), l) in zip(profiles, ls)
                @test name == "l=$l"
                @test profile ≈ Magrathea._mhd_radial_eval(evec[idx[(l, section)]],
                                                           op.params.ricb, rg)
            end
        end
        @test_throws ArgumentError ext._ext_radial_profiles(
            plotting_mhd_nofield.extra.operator, plotting_mhd_nofield.eigenvectors[:, 1],
            :magnetic_poloidal)

        # Triglobal: every coupled m block, matching the package's own extraction
        res = plotting_tri
        layout = ext._ext_layout(res)
        @test layout isa Magrathea.CoupledModeProblem
        @test layout.block_indices == plotting_coupled.block_indices
        evec = res.eigenvectors[:, 1]
        for (field, pick) in ((:poloidal, first), (:toroidal, last))
            _, profiles = ext._ext_radial_profiles(layout, evec, field)
            expected = Tuple{String,Vector{ComplexF64}}[]
            for m in plotting_coupled.m_range
                coeffs = pick(Magrathea._extract_mode_coefficients(evec, plotting_coupled, m))
                append!(expected, [("m=$m, l=$l", coeffs[l]) for l in sort(collect(keys(coeffs)))])
            end
            @test first.(profiles) == first.(expected)
            @test all(p ≈ q for (p, q) in zip(last.(profiles), last.(expected)))
        end
        _, θprofiles = ext._ext_radial_profiles(layout, evec, :temperature)
        @test length(θprofiles) == sum(length(Magrathea._mode_reconstruction(
            plotting_coupled, abs(m)).op.l_sets[:Θ]) for m in plotting_coupled.m_range)
        @test_throws ArgumentError ext._ext_radial_profiles(layout, evec, :magnetic_toroidal)
    end

    @testset "Makie extension: meridional fields use package reconstructions" begin
        ext = makie_ext
        npts = 12
        for res in (plotting_onset, plotting_mhd)
            evec = res.eigenvectors[:, 1]
            op = res.extra.operator
            ur, uθ, uφ, rg, grid = perturbation_velocity(evec, op; Nθ=npts)
            for (field, U) in ((:ur, ur), (:utheta, uθ), (:uphi, uφ))
                r_grid, θ, values = ext._ext_meridional_field(op, evec, field, npts)
                @test r_grid == rg
                @test θ == grid.θ
                @test values == U
            end
            θfield, _, _ = perturbation_temperature(evec, op; Nθ=npts)
            @test ext._ext_meridional_field(op, evec, :temperature, npts)[3] == θfield
        end
        evec = plotting_mhd.eigenvectors[:, 1]
        Br, Bθ, Bφ, _, _ = perturbation_magnetic(evec, plotting_mhd.extra.operator; Nθ=npts)
        for (field, Bc) in ((:Br, Br), (:Btheta, Bθ), (:Bphi, Bφ))
            @test ext._ext_meridional_field(plotting_mhd.extra.operator, evec, field, npts)[3] == Bc
        end
        @test_throws ArgumentError ext._ext_meridional_field(
            plotting_onset.extra.operator, plotting_onset.eigenvectors[:, 1], :Br, npts)
        @test_throws ArgumentError ext._ext_meridional_field(
            plotting_mhd_nofield.extra.operator, plotting_mhd_nofield.eigenvectors[:, 1],
            :Bphi, npts)

        # Triglobal: coupled-mode velocity at φ = 0
        evec = plotting_tri.eigenvectors[:, 1]
        ur, uθ, uφ = Magrathea.eigenvector_to_velocity_triglobal(
            evec, plotting_coupled; Nθ=npts, φ_slice=0.0)
        layout = ext._ext_layout(plotting_tri)
        for (field, U) in ((:ur, ur), (:utheta, uθ), (:uphi, uφ))
            r_grid, θ, values = ext._ext_meridional_field(layout, evec, field, npts)
            @test length(r_grid) == size(values, 1) && length(θ) == size(values, 2)
            @test issorted(r_grid) && first(r_grid) ≈ 0.35 && last(r_grid) ≈ 1.0
            @test values ≈ U
        end
        @test_throws ArgumentError ext._ext_meridional_field(layout, evec, :temperature, npts)
        @test_throws ArgumentError ext._ext_meridional_field(layout, evec, :Br, npts)
    end

    @testset "Makie extension: public plotting functions" begin
        ax_plots(fig) = fig.content[1].scene.plots
        for res in (plotting_onset, plotting_mhd, plotting_tri)
            @test eigenspectrum(res) isa Makie.Figure
            fig = plot_radial(res, 1; field=:toroidal)
            @test fig isa Makie.Figure
            _, profiles = makie_ext._ext_radial_profiles(
                makie_ext._ext_layout(res), res.eigenvectors[:, 1], :toroidal)
            @test count(p -> p isa Makie.Lines, ax_plots(fig)) == length(profiles)
            fig = plot_meridional(res, 1; field=:uφ, npoints=12)   # Unicode alias
            @test count(p -> p isa Makie.Heatmap, ax_plots(fig)) == 1
        end
        @test plot_meridional(plotting_onset, 1; npoints=12) isa Makie.Figure
        @test plot_meridional(plotting_mhd, 1; field=:Btheta, npoints=12) isa Makie.Figure
        @test plot_radial(plotting_mhd, 1; field=:magnetic_poloidal) isa Makie.Figure

        @test_throws ArgumentError plot_radial(plotting_onset, 1; field=:bogus)
        @test_throws ArgumentError plot_radial(plotting_onset, 1; field=:magnetic_toroidal)
        @test_throws ArgumentError plot_meridional(plotting_onset, 1; field=:poloidal)
        @test_throws ArgumentError plot_meridional(plotting_onset, 1; field=:Br)
        @test_throws ArgumentError plot_meridional(plotting_tri, 1)   # no triglobal temperature

        # Results built without a stored operator rebuild the layout from the problem.
        bare = StabilityResult(plotting_onset.eigenvalues, plotting_onset.eigenvectors,
                               plotting_onset.problem)
        @test plot_radial(bare, 1; field=:temperature) isa Makie.Figure
        # Eigenvectors that do not match the problem layout are rejected.
        short = StabilityResult(plotting_onset.eigenvalues,
                                plotting_onset.eigenvectors[1:end-1, :], plotting_onset.problem)
        @test_throws DimensionMismatch plot_radial(short, 1)
        @test_throws DimensionMismatch plot_meridional(short, 1; field=:ur)
        short_mhd = StabilityResult(plotting_mhd.eigenvalues,
                                    plotting_mhd.eigenvectors[1:end-1, :], plotting_mhd.problem;
                                    extra=plotting_mhd.extra)
        @test_throws DimensionMismatch plot_radial(short_mhd, 1)
        short_tri = StabilityResult(plotting_tri.eigenvalues,
                                    plotting_tri.eigenvectors[1:end-1, :], plotting_tri.problem)
        @test_throws DimensionMismatch plot_meridional(short_tri, 1; field=:ur)

        fig = @test_logs (:warn, r"out of range") plot_radial(plotting_onset, 99)
        @test fig isa Makie.Figure
    end
end

recipes_ext = plotting_has_recipesbase ?
    Base.get_extension(Magrathea, :MagratheaRecipesBaseExt) : nothing
if recipes_ext === nothing
    @info "RecipesBase extension not loaded; skipping RecipesBase plotting tests"
else
    # Plots normally defines this backend hook; RecipesBase's own tests stub it too.
    hasmethod(RecipesBase.is_key_supported, Tuple{Symbol}) ||
        (RecipesBase.is_key_supported(::Symbol) = true)

    @testset "RecipesBase extension recipes" begin
        attrs(pairs...) = Dict{Symbol,Any}(pairs...)
        for res in (plotting_onset, plotting_mhd, plotting_tri)
            series = only(RecipesBase.apply_recipe(attrs(), res))
            @test series.args == (real.(res.eigenvalues), imag.(res.eigenvalues))
        end

        series = only(RecipesBase.apply_recipe(attrs(), [plotting_onset, plotting_tri]))
        @test series.args[1] == [plotting_onset.problem.params.Ra, plotting_tri.problem.params.Ra]
        @test series.args[2] == [plotting_onset.growth_rate, plotting_tri.growth_rate]
        series = only(RecipesBase.apply_recipe(attrs(:sweep_param => :Le), [plotting_mhd]))
        @test series.args[1] == [plotting_mhd.problem.params.Le]

        # MHDParams names differ (ricb, N); triglobal results couple a whole m_range.
        @test_throws ArgumentError RecipesBase.apply_recipe(attrs(:sweep_param => :Nr), [plotting_mhd])
        @test_throws ArgumentError RecipesBase.apply_recipe(attrs(:sweep_param => :m), [plotting_tri])
        series = only(RecipesBase.apply_recipe(attrs(:sweep_param => :m), [plotting_onset]))
        @test series.args[1] == [plotting_onset.problem.params.m]
    end
end
