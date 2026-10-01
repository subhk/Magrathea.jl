module BoundaryAPIGuards
using Test, Magrathea, LinearAlgebra

@testset "Stationary mechanical walls agree across public mean-state APIs" begin
    cd=ChebyshevDiffn(16,[.35,1.],4)
    config=(E=.1,Pr=1.,Ra=30.,χ=.35,Nr=16,lmax=4,m=1)
    for wall in (:no_slip,:stress_free)
        other=wall===:no_slip ? :stress_free : :no_slip
        state=meridional_basic_state(cd,.35,.1,30.,1.,4,.1;mechanical_bc=wall)
        matching=OnsetParams(;config...,mechanical_bc=wall)
        incompatible=OnsetParams(;config...,mechanical_bc=other)
        @test BiglobalProblem(matching,state).basic_state === state
        @test OnsetParams(;config...,mechanical_bc=wall,basic_state=state).basic_state === state
        @test BiglobalParams(;config...,mechanical_bc=wall,basic_state=state).basic_state === state
        @test_throws ArgumentError BiglobalProblem(incompatible,state)
        @test_throws ArgumentError OnsetParams(;config...,mechanical_bc=other,basic_state=state)
        @test_throws ArgumentError BiglobalParams(;config...,mechanical_bc=other,basic_state=state)
        @test_throws ArgumentError build_basic_state_operators(state,LinearStabilityOperator(incompatible),1)

        # An m=2-only wall violation cannot disappear in the m=0 projection
        # used to construct the diagonal triglobal blocks.
        three=nonaxisymmetric_basic_state(cd,.35,.1,30.,1.,4,2,
            Dict((2,2)=>.1);mechanical_bc=wall)
        @test TriglobalProblem(matching,three,0:2).basic_state === three
        @test_throws ArgumentError TriglobalProblem(incompatible,three,0:2)
        tri_config=(E=.1,Pr=1.,Ra=30.,χ=.35,Nr=16,lmax=4,m_range=0:2,basic_state_3d=three)
        @test setup_coupled_mode_problem(TriglobalParams(;tri_config...,mechanical_bc=wall)) isa Magrathea.CoupledModeProblem
        @test_throws ArgumentError setup_coupled_mode_problem(TriglobalParams(;tri_config...,mechanical_bc=other))
    end

    # Compatibility is determined from the physical field, not a label saying
    # which constructor was used. Motionless conduction satisfies both choices.
    conduction=conduction_basic_state(cd,.35,4)
    conduction3=nonaxisymmetric_basic_state(cd,.35,.1,30.,1.,4,2,
        Dict{Tuple{Int,Int},Float64}())
    for wall in (:no_slip,:stress_free)
        p=OnsetParams(;config...,mechanical_bc=wall)
        @test BiglobalProblem(p,conduction).basic_state === conduction
        @test TriglobalProblem(p,conduction3,0:2).basic_state === conduction3
    end
end

end
