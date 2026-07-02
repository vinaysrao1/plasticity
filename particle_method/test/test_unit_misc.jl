# Additional unit tests called out in DESIGN §12 ("Unit tests additionally
# cover..."): grid indexing round-trip, particle→cell binning,
# `ParticleInversionError` on a deliberately inverted F, APIC exactness on
# affine velocity fields, F̄ dilation algebra, and (added here) `load!`
# nodal-force correctness.

using ParticlePlasticity
using StaticArrays
using LinearAlgebra
using Test

const GridMod = ParticlePlasticity.GridMod
const ParticlesMod = ParticlePlasticity.ParticlesMod
const Transfer = ParticlePlasticity.Transfer
const Constitutive = ParticlePlasticity.Constitutive
const BC = ParticlePlasticity.BoundaryConditions

@testset "unit: grid indexing round-trip" begin
    grid = GridMod.Grid(SVector(-1.0, 2.0, 0.5), 0.3, (7, 9, 5))
    for ix in 1:7, iy in 1:9, iz in 1:5
        i = GridMod.node_index(grid, ix, iy, iz)
        x = GridMod.node_coords(grid, i)
        @test isapprox(x[1], -1.0 + (ix - 1) * 0.3; atol=1e-12)
        @test isapprox(x[2], 2.0 + (iy - 1) * 0.3; atol=1e-12)
        @test isapprox(x[3], 0.5 + (iz - 1) * 0.3; atol=1e-12)
    end
end

@testset "unit: particle -> cell binning (F-bar)" begin
    h = 1.0
    grid = GridMod.Grid(SVector(0.0, 0.0, 0.0), h, (5, 5, 5))
    # cell (0,0,0): x in [0,1)^3 ; cell (1,0,0): x in [1,2)x[0,1)x[0,1)
    c1 = Constitutive._cell_id(grid, SVector(0.4, 0.4, 0.4))
    c2 = Constitutive._cell_id(grid, SVector(0.6, 0.4, 0.4))
    c3 = Constitutive._cell_id(grid, SVector(1.4, 0.4, 0.4))
    @test c1 == c2                 # same cell
    @test c1 != c3                 # different cell
    @test Constitutive.ncells(grid) == 4 * 4 * 4
end

@testset "unit: ParticleInversionError on inverted F" begin
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0)
    pts = ParticlesMod.Particles([SVector(0.0, 0.0, 0.0)], 1.0, 1.0)
    # force an inverted (negative-determinant) F directly, bypassing the
    # normal F-update, to exercise the guard deliberately.
    pts.F[1] = SMatrix{3,3,Float64,9}(-1, 0, 0, 0, 1, 0, 0, 0, 1)
    pts.C[1] = zero(SMatrix{3,3,Float64,9})   # Lp=0 so Ftrial = pts.F[1] itself
    @test_throws Constitutive.ParticleInversionError Constitutive.particle_stress_update!(pts, 1, mat, 0.0)
end

@testset "unit: APIC exactness on an affine velocity field" begin
    h = 0.5
    grid = GridMod.Grid(SVector(-2.0, -2.0, -2.0), h, (17, 17, 17))
    pts = ParticlesMod.sample_box(SVector(-1.0, -1.0, -1.0), SVector(1.0, 1.0, 1.0), h; ppc=2, ρ=1.0)
    np = length(pts.x)
    A = SMatrix{3,3,Float64,9}(0.1, 0.4, -0.2, -0.3, 0.05, 0.6, 0.2, -0.1, 0.25)
    x0 = SVector(0.13, -0.07, 0.02)
    v0 = SVector(1.0, -2.0, 0.5)
    for p in 1:np
        pts.v[p] = v0 + A * (pts.x[p] - x0)
        pts.C[p] = A
    end
    fill!(grid.m, 0.0)
    fill!(grid.p, zero(SVector{3,Float64}))
    fill!(grid.f, zero(SVector{3,Float64}))
    fill!(grid.v, zero(SVector{3,Float64}))
    Transfer.p2g!(grid, pts)
    for i in eachindex(grid.m)
        grid.m[i] > 0 && (grid.v[i] = grid.p[i] / grid.m[i])
    end
    Transfer.g2p!(grid, pts, 0.0)
    for p in 1:np
        @test isapprox(pts.v[p], v0 + A * (pts.x[p] - x0); atol=1e-9)
        @test isapprox(pts.C[p], A; atol=1e-9)
    end
end

@testset "unit: F-bar dilation algebra (det F̄ₚ = J̄_c)" begin
    h = 1.0
    grid = GridMod.Grid(SVector(0.0, 0.0, 0.0), h, (4, 4, 4))
    xs = [SVector(0.3, 0.3, 0.3), SVector(0.7, 0.3, 0.3), SVector(0.3, 0.7, 0.3)]
    pts = ParticlesMod.Particles(xs, 1.0, 1.0)
    pts.F[1] = SMatrix{3,3,Float64,9}(1.2, 0, 0, 0, 1.1, 0, 0, 0, 0.9)
    pts.F[2] = SMatrix{3,3,Float64,9}(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0)
    pts.F[3] = SMatrix{3,3,Float64,9}(1.5, 0, 0, 0, 0.8, 0, 0, 0, 1.0)
    Ftrial = copy(pts.F)
    Jnum = zeros(Constitutive.ncells(grid))
    Jden = zeros(Constitutive.ncells(grid))
    Constitutive.cell_Jbar!(Jnum, Jden, grid, pts, Ftrial)
    c = Constitutive._cell_id(grid, xs[1])
    Jbar = Jnum[c] / Jden[c]
    expected_Jbar = (det(pts.F[1]) + det(pts.F[2]) + det(pts.F[3])) / 3   # equal V0
    @test isapprox(Jbar, expected_Jbar; atol=1e-12)
    for p in 1:3
        scale = cbrt(Jbar / det(Ftrial[p]))
        Fbar = scale * Ftrial[p]
        @test isapprox(det(Fbar), Jbar; atol=1e-10)
    end
end

@testset "unit: load! nodal force" begin
    h = 0.5
    grid = GridMod.Grid(SVector(0.0, 0.0, 0.0), h, (5, 5, 5))
    loads = BC.ForceBC[]
    # face x=1.0 within the 2x2 grid of y,z in [0.5,1.5]
    BC.load!(loads, grid, x -> abs(x[1] - 1.0) < 1e-9 && 0.4 < x[2] < 1.6 && 0.4 < x[3] < 1.6,
             :z, -90.0; distribute=true)
    fill!(grid.f, zero(SVector{3,Float64}))
    BC.apply_loads!(grid, loads, 0.0)
    total = sum(grid.f)
    @test isapprox(total[3], -90.0; atol=1e-10)   # total force conserved under distribute=true
    @test isapprox(total[1], 0.0; atol=1e-12)
end
