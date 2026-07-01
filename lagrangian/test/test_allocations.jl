# Allocation tests for the hot kernels (DESIGN §7.1, T8, T20).
# These MUST be measured inside a function so the @allocated macro does not
# capture global-variable boxing.

using PlasticityFEM
using PlasticityFEM.Materials
using PlasticityFEM.Elements
using StaticArrays
using Test

@testset "return_map zero allocation" begin
    function f()
        mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)
        Z6 = zero(SVector{6,Float64})
        ε = SVector{6,Float64}(0.01, 0, 0, 0, 0, 0)
        return_map(mat, ε, Z6, Z6, 0.0)               # warmup
        a1 = @allocated return_map(mat, ε, Z6, Z6, 0.0)   # plastic branch
        εe = SVector{6,Float64}(1e-5, 0, 0, 0, 0, 0)
        return_map(mat, εe, Z6, Z6, 0.0)
        a2 = @allocated return_map(mat, εe, Z6, Z6, 0.0)  # elastic branch
        return a1, a2
    end
    a1, a2 = f()
    @test a1 == 0
    @test a2 == 0
end

@testset "element_force_tangent! zero allocation" begin
    function f()
        mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)
        mesh = box_mesh(1.0, 1.0, 1.0, 1, 1, 1)
        cache = precompute_cache(mesh.nodes, mesh.elements)
        ngp = 8
        εp = zeros(6,ngp); β = zeros(6,ngp); ᾱ = zeros(ngp); σ = zeros(6,ngp)
        ue = SVector{24,Float64}(ntuple(i -> 0.001*i, 24))
        B1, Jw1 = element_geometry(cache, 1)
        element_force_tangent!(mat, B1, Jw1, ue, εp, β, ᾱ, 1, σ, Val(false))  # warmup
        a = @allocated element_force_tangent!(mat, B1, Jw1, ue, εp, β, ᾱ, 1, σ, Val(false))
        ac = @allocated element_force_tangent!(mat, B1, Jw1, ue, εp, β, ᾱ, 1, σ, Val(true))
        return a, ac
    end
    a, ac = f()
    @test a == 0
    @test ac == 0
end
