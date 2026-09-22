using Test, Magrathea, SparseArrays, LinearAlgebra

@testset "MHD validated parameter and model scope" begin
    base=(;E=.01,Pr=1.,Pm=1.,Ra=1.,Le=.1,ricb=.35,m=1,lmax=3,N=8,B0_type=axial)
    for kw in ((;Le=-.1),(;Le=NaN),(;Le=Inf),(;E=Inf),(;m=-1),
               (;bci=3),(;bco=3),(;bci_thermal=2),(;bco_thermal=-1),
               (;bci_magnetic=3),(;bco_magnetic=1),(;forcing_frequency=1.))
        @test_throws ArgumentError MHDParams(;merge(base,kw)...)
    end
    p=MHDParams(;base...)
    @test_throws ArgumentError MHDProblem{Float64,Symbol}(p,:unsupported_mean_state)
end

@testset "MHD boundary helpers match coefficient-space assembly" begin
    for T in (Float32,Float64), inner in (0,1,2), outer in (0,2), mechanical in (0,1)
        p=MHDParams(E=T(.01),Pr=one(T),Pm=one(T),Ra=one(T),Le=T(.1),ricb=T(.35),
                    m=1,lmax=3,N=8,B0_type=axial,B0_amplitude=zero(T),bci_magnetic=inner,bco_magnetic=outer,
                    bci=mechanical,bco=mechanical,bci_thermal=1,bco_thermal=1)
        op=MHDStabilityOperator(p); c=Magrathea._assemble_mhd_coo(op)
        A=sparse(c.A_rows,c.A_cols,c.A_vals,c.n,c.n)
        B=sparse(c.B_rows,c.B_cols,c.B_vals,c.n,c.n)
        Magrathea.apply_velocity_boundary_conditions!(A,B,op)
        Magrathea.apply_temperature_boundary_conditions!(A,B,op)
        Magrathea.apply_magnetic_boundary_conditions!(A,B,op,:f)
        Magrathea.apply_magnetic_boundary_conditions!(A,B,op,:g)
        assembledA,assembledB,interior,_=assemble_mhd_matrices(op)
        @test A ≈ assembledA
        @test B == assembledB
        @test eltype(A) === Complex{T}
        index=Magrathea._mhd_index_map(op)
        bc=setdiff(1:op.matrix_size,interior)
        @test length(bc) == 4length(op.ll_u)+2(length(op.ll_v)+length(op.ll_h)+length(op.ll_f)+length(op.ll_g)) +
                            (inner==1 ? length(op.ll_f)+length(op.ll_g) : 0)
        @test iszero(B[bc,:])
        if inner==1
            for (section,core,ls) in ((:f,:fi,op.ll_f),(:g,:gi,op.ll_g)), l in ls
                shell=index[(l,section)]; solid=index[(l,core)]
                @test !iszero(A[last(shell),solid])
                @test !iszero(A[last(solid),shell])
                @test !iszero(B[solid[1:end-1],solid])
            end
        end
    end
end
