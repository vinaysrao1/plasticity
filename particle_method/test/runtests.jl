using ParticlePlasticity
using Test

@testset "ParticlePlasticity" begin
    @testset "G0 BSpline" begin
        include("test_G0_bspline.jl")
    end
    @testset "G1 Kernel" begin
        include("test_G1_kernel.jl")
    end
    @testset "G1b Drift" begin
        include("test_G1b_drift.jl")
    end
    @testset "G2 Elastodynamics" begin
        include("test_G2_elastodynamics.jl")
    end
    @testset "G3 Tension" begin
        include("test_G3_tension.jl")
    end
    @testset "G4 Fbar" begin
        include("test_G4_fbar.jl")
    end
    @testset "G5 Cantilever" begin
        include("test_G5_cantilever.jl")
    end
    @testset "G6 Necking" begin
        include("test_G6_necking.jl")
    end
    @testset "Unit (misc, DESIGN §12)" begin
        include("test_unit_misc.jl")
    end
end
