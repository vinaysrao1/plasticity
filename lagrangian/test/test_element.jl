# Element tests (DESIGN §8.2, T9–T12).

using PlasticityFEM
using PlasticityFEM.Elements
using PlasticityFEM.Materials
using StaticArrays
using LinearAlgebra
using Test

# helper: element node coords (8×3 SMatrix) from mesh
function elem_coords(mesh, e)
    SMatrix{8,3,Float64,24}(ntuple(24) do k
        a = (k - 1) % 8 + 1
        j = (k - 1) ÷ 8 + 1
        mesh.nodes[j, mesh.elements[a, e]]
    end)
end

@testset "T9 partition of unity" begin
    mesh = box_mesh(2.0, 1.5, 1.0, 2, 2, 2)
    Xe = elem_coords(mesh, 1)
    for ξ in PlasticityFEM.Elements.GAUSS_PTS
        N = hex8_shape(ξ)
        @test sum(N) ≈ 1.0 atol=1e-13
        dN = hex8_dshape(ξ)
        J = PlasticityFEM.Elements.jacobian(Xe, dN)
        dNdx = dN * inv(J)
        # Σ_a ∂N_a/∂x_j = 0
        for j in 1:3
            @test sum(@view dNdx[:, j]) ≈ 0.0 atol=1e-13
        end
    end
end

@testset "T10 jacobian / volume" begin
    lx, ly, lz = 2.0, 3.0, 4.0
    nx, ny, nz = 2, 3, 4
    mesh = box_mesh(lx, ly, lz, nx, ny, nz)
    cache = precompute_cache(mesh.nodes, mesh.elements)
    vol_elem = (lx/nx) * (ly/ny) * (lz/nz)
    for e in 1:mesh.nelem
        _, detJw = element_geometry(cache, e)
        # each GP detJ = element volume / 8
        for g in 1:8
            @test detJw[g] ≈ vol_elem / 8 rtol=1e-12
            @test detJw[g] > 0
        end
        # Σ detJ·w over element = element volume
        @test sum(detJw) ≈ vol_elem rtol=1e-12
    end
end

@testset "T11 rigid body motion" begin
    mesh = box_mesh(1.0, 1.0, 1.0, 1, 1, 1)
    cache = precompute_cache(mesh.nodes, mesh.elements)
    # uniform translation
    t = SVector{3,Float64}(0.3, -0.2, 0.5)
    ue_trans = SVector{24,Float64}(ntuple(c -> t[(c-1)%3+1], 24))
    Bs, detJw = element_geometry(cache, 1)
    for g in 1:8
        ε = Bs[g] * ue_trans
        @test norm(ε) <= 1e-12
    end
    # infinitesimal rotation: u = ω × (x - x0)
    ω = SVector{3,Float64}(1e-6, -2e-6, 3e-6)
    ue_rot = SVector{24,Float64}(ntuple(24) do c
        a = (c - 1) ÷ 3 + 1
        comp = (c - 1) % 3 + 1
        x = SVector{3,Float64}(mesh.nodes[1, mesh.elements[a,1]],
                               mesh.nodes[2, mesh.elements[a,1]],
                               mesh.nodes[3, mesh.elements[a,1]])
        cross(ω, x)[comp]
    end)
    for g in 1:8
        ε = Bs[g] * ue_rot
        @test norm(ε) <= 1e-10   # zero to linear order
    end
    # element internal force from rigid translation = 0
    mat = J2Material(E=210e3, ν=0.3, σy0=1e9)
    ngp = 8
    εp = zeros(6,ngp); β = zeros(6,ngp); ᾱ = zeros(ngp); σ = zeros(6,ngp)
    Fe, _ = element_force_tangent!(mat, Bs, detJw, ue_trans,
                                   εp, β, ᾱ, 1, σ, Val(false))
    @test norm(Fe) <= 1e-9
end

@testset "T12 single-element patch test (constant strain)" begin
    mesh = box_mesh(1.3, 0.9, 1.1, 1, 1, 1)
    mat = J2Material(E=210e3, ν=0.3, σy0=1e9)   # stay elastic
    model = Model(mesh, mat)
    # impose a uniform strain field ε_xx = c via displacements u_x = c·x on all nodes
    c = 1e-3
    allnodes = collect(1:mesh.nnodes)
    # prescribe every boundary node consistent with u = (c x, 0, 0)
    for n in allnodes
        x = mesh.nodes[1, n]
        prescribe!(model, [n], :x, c * x; ramp=false)
        prescribe!(model, [n], :y, 0.0; ramp=false)
        prescribe!(model, [n], :z, 0.0; ramp=false)
    end
    solve!(model; nsteps=1, tol=1e-10)
    σ = gauss_stress(model)
    # exact constant stress = ℂ : ε with ε = [c,0,0,0,0,0]
    σ_exact = mat.Cmat * SVector{6,Float64}(c, 0, 0, 0, 0, 0)
    for g in 1:8
        @test σ[:, g] ≈ Vector(σ_exact) atol=1e-7 * mat.E * c
    end
end
