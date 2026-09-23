module MagratheaRecipesBaseExt

using Magrathea
using RecipesBase

# Eigenvalue spectrum scatter plot
@recipe function f(r::Magrathea.StabilityResult)
    xlabel --> "Growth rate (σᵣ)"
    ylabel --> "Frequency (σᵢ)"
    seriestype --> :scatter
    markersize --> 6
    markershape --> :circle
    label --> "Eigenvalues ($(length(r.eigenvalues)))"
    real.(r.eigenvalues), imag.(r.eigenvalues)
end

# Value of the swept parameter for one result. Parameter names differ between
# problem types (e.g. MHDParams has `ricb` and `N` instead of `χ` and `Nr`), and a
# triglobal result couples every m in `m_range`, so its `params.m` is not the
# azimuthal order of the eigenmode.
function _sweep_value(problem, sweep_param::Symbol)
    if problem isa Magrathea.TriglobalProblem && sweep_param === :m
        throw(ArgumentError(
            "sweep_param=:m is not defined for TriglobalProblem results, which couple " *
            "m_range=$(problem.m_range)"))
    end
    params = problem.params
    hasproperty(params, sweep_param) || throw(ArgumentError(
        "Parameter :$sweep_param not found on $(typeof(params)). " *
        "Available fields: $(fieldnames(typeof(params)))"))
    return getproperty(params, sweep_param)
end

# Growth rate vs parameter sweep
@recipe function f(results::Vector{<:Magrathea.StabilityResult}; sweep_param=:Ra)
    xlabel --> string(sweep_param)
    ylabel --> "Growth rate"
    seriestype --> :line
    markershape --> :circle
    markersize --> 4
    label --> "Growth rate vs $(sweep_param)"
    xs = [_sweep_value(r.problem, Symbol(sweep_param)) for r in results]
    ys = [r.growth_rate for r in results]
    xs, ys
end

end # module
