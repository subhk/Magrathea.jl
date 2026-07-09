using Test
using TOML
using Magrathea

@testset "Legacy v1 API is removed" begin
    @test !isdefined(Magrathea, :ShellParams)
    @test !isdefined(Magrathea, :leading_modes)
    @test !isdefined(Magrathea, :_arnoldi_eigensolve)
    @test !isdefined(Magrathea, :build_thermal_wind)
    @test !isdefined(Magrathea, :build_thermal_wind_3d)

    project = TOML.parsefile(joinpath(dirname(@__DIR__), "Project.toml"))
    @test !haskey(project["deps"], "ArnoldiMethod")
    @test !haskey(project["compat"], "ArnoldiMethod")
end
