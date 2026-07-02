# G0 — B-spline partition of unity (DESIGN §12).
#
# Σᵢ wᵢₚ = 1 and Σᵢ ∇wᵢₚ = 0 for random particle positions (to 1e-12); linear
# field reproduction Σᵢ wᵢₚ xᵢ = xₚ. Pure unit test, no physics.

using ParticlePlasticity
using StaticArrays
using Random
using Test

const GridMod = ParticlePlasticity.GridMod

@testset "G0: B-spline partition of unity" begin
    h = 0.37
    grid = GridMod.Grid(SVector(0.0, 0.0, 0.0), h, (12, 12, 12))

    Random.seed!(20260702)
    ntrials = 500
    for _ in 1:ntrials
        # keep particles well inside the grid interior (away from the stencil
        # bounds-check edge) so the 27-node stencil is always fully valid.
        xp = SVector(3.0 * h + 4.0 * h * rand(),
                     3.0 * h + 4.0 * h * rand(),
                     3.0 * h + 4.0 * h * rand())
        st = GridMod.bspline_stencil(grid, xp)

        wsum = sum(st.w)
        @test isapprox(wsum, 1.0; atol=1e-12)

        gsum = sum(st.gradw)
        @test isapprox(gsum[1], 0.0; atol=1e-12)
        @test isapprox(gsum[2], 0.0; atol=1e-12)
        @test isapprox(gsum[3], 0.0; atol=1e-12)

        # linear field reproduction: Σᵢ wᵢₚ xᵢ = xₚ
        xrecon = sum(st.w[k] * st.pos[k] for k in 1:27)
        @test isapprox(xrecon[1], xp[1]; atol=1e-12)
        @test isapprox(xrecon[2], xp[2]; atol=1e-12)
        @test isapprox(xrecon[3], xp[3]; atol=1e-12)
    end

    # ParticleOutOfBoundsError for a particle whose stencil escapes the grid
    @test_throws GridMod.ParticleOutOfBoundsError GridMod.bspline_stencil(grid, SVector(0.001, 0.001, 0.001), 7)
end
