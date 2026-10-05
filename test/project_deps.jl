using Test
using TOML
using Magrathea

@testset "Project dependencies" begin
    project = TOML.parsefile(joinpath(dirname(@__DIR__), "Project.toml"))
    deps = project["deps"]

    @test haskey(deps, "Logging")
    # `solve` extends CommonSolve.solve so it does not clash with SciML exports.
    @test haskey(deps, "CommonSolve")
    @test nameof(parentmodule(Magrathea.solve)) === :CommonSolve
end
