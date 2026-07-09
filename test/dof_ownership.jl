using Test
using Magrathea

@testset "row_to_dof / dof_to_row round-trip" begin
    im = Dict((1,:P)=>1:4, (2,:P)=>5:8, (1,:Θ)=>9:12)   # Nr=4, 3 blocks, 12 rows
    for grow in 1:12
        key, loc = Magrathea.row_to_dof(im, grow)
        @test Magrathea.dof_to_row(im, key, loc) == grow
    end
    @test Magrathea.row_to_dof(im, 6) == ((2,:P), 2)
    @test Magrathea.dof_to_row(im, (1,:Θ), 1) == 9
    @test_throws ErrorException Magrathea.row_to_dof(im, 0)
    @test_throws ErrorException Magrathea.row_to_dof(im, 13)
    @test_throws ErrorException Magrathea.dof_to_row(im, (9,:P), 1)
    @test_throws ErrorException Magrathea.dof_to_row(im, (1,:P), 5)
end

@testset "owned_block_ranges" begin
    im = Dict((1,:P)=>1:4, (2,:P)=>5:8, (1,:Θ)=>9:12)
    @test Magrathea.owned_block_ranges(im, 0, 12) ==
          [((1,:P),1:4), ((2,:P),1:4), ((1,:Θ),1:4)]
    @test Magrathea.owned_block_ranges(im, 0, 6) ==
          [((1,:P),1:4), ((2,:P),1:2)]
    @test Magrathea.owned_block_ranges(im, 6, 12) ==
          [((2,:P),3:4), ((1,:Θ),1:4)]
    @test Magrathea.owned_block_ranges(im, 4, 8) == [((2,:P),1:4)]
    @test Magrathea.owned_block_ranges(im, 4, 4) == []
end
