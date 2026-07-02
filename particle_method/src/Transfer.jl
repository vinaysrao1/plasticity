"""
    Transfer

Particle↔grid APIC transfers, `p2g!` and `g2p!` (DESIGN §4). Single transfer
(no MUSL remap): `p2g!` scatters mass/momentum/internal-force, `g2p!` gathers
the updated grid velocity back (PIC velocity + APIC affine `Cₚ`) and advects
particles. Allocation-free hot loops (27-node stencil per particle).
"""
module Transfer

using StaticArrays
using ..GridMod: Grid, bspline_stencil
using ..ParticlesMod: Particles
using PlasticityFEM.FiniteStrain: voigt_to_sym3

export p2g!, g2p!

"""
    p2g!(grid, particles)

Particles → grid (DESIGN §4): for each particle, scatter to its 27-node
stencil
- `mᵢ += wᵢₚ mₚ`
- `pᵢ += wᵢₚ mₚ (vₚ + Cₚ(xᵢ − xₚ))` (APIC affine momentum)
- `fᵢ += −Vₚ σₚ ∇wᵢₚ` (internal force from the **current** Cauchy stress
  `σₚ = τₚ/Jₚ` and current volume `Vₚ = Jₚ Vₚ⁰`, both already committed from
  the previous step's constitutive update).

Assumes the grid buffers (`m`, `p`, `f`) were zeroed by the caller (`Step.step!`).
Does not finalize `vᵢ = pᵢ/mᵢ` — that happens in the grid momentum update.
"""
function p2g!(grid::Grid, pts::Particles)
    @inbounds for p in eachindex(pts.x)
        xp = pts.x[p]
        mp = pts.m[p]
        vp = pts.v[p]
        Cp = pts.C[p]
        Jp = pts.J[p]
        Vp = Jp * pts.V0[p]
        σp = voigt_to_sym3(pts.τ[p]) / Jp
        st = bspline_stencil(grid, xp, p)
        for k in 1:27
            i = st.idx[k]
            w = st.w[k]
            gw = st.gradw[k]
            xi = st.pos[k]
            grid.m[i] += w * mp
            grid.p[i] += (w * mp) * (vp + Cp * (xi - xp))
            grid.f[i] += -(Vp * (σp * gw))
        end
    end
    return grid
end

"""
    g2p!(grid, particles, dt)

Grid → particles (DESIGN §4), after the grid momentum update (BCs already
applied to `grid.v`):
- `vₚ = Σᵢ wᵢₚ vᵢ*` (PIC velocity)
- `Cₚ = (4/h²) Σᵢ wᵢₚ vᵢ* (xᵢ − xₚ)ᵀ` (APIC affine; exact `4/h²` for the
  quadratic B-spline's constant inertia tensor `Dₚ = ¼h²I`, DESIGN §5)
- `xₚ += Δt vₚ` (advect)

Does **not** update `F`/stress — that is `Constitutive.update_particles!`.
"""
function g2p!(grid::Grid, pts::Particles, dt::Float64)
    h = grid.h
    inv_D = 4.0 / (h * h)
    @inbounds for p in eachindex(pts.x)
        xp = pts.x[p]
        st = bspline_stencil(grid, xp, p)
        vp = zero(SVector{3,Float64})
        Bp = zero(SMatrix{3,3,Float64,9})
        for k in 1:27
            i = st.idx[k]
            w = st.w[k]
            vi = grid.v[i]
            dx = st.pos[k] - xp
            vp += w * vi
            Bp += (w * vi) * dx'
        end
        pts.v[p] = vp
        pts.C[p] = inv_D * Bp
        pts.x[p] = xp + dt * vp
    end
    return pts
end

end # module
