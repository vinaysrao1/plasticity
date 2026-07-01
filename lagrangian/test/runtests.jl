using PlasticityFEM
using Test

@testset "PlasticityFEM" begin
    @testset "Materials" begin
        include("test_material.jl")
    end
    @testset "Elements" begin
        include("test_element.jl")
    end
    @testset "Assembly" begin
        include("test_assembly.jl")
    end
    @testset "Solver" begin
        include("test_solver.jl")
    end
    @testset "Allocations" begin
        include("test_allocations.jl")
    end
    @testset "FiniteStrain" begin
        include("test_finite_strain.jl")
    end
    @testset "Visualization" begin
        include("test_visualization.jl")
    end
    @testset "HardValidation" begin
        include("hard_validation.jl")
    end
    @testset "Scaling" begin
        include("test_scaling.jl")
    end
end
