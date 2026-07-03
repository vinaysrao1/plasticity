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
    g2p!(grid, particles, dt; flip=0.0)

Grid → particles (DESIGN §4), after the grid momentum update (BCs already
applied to `grid.v`):
- PIC velocity `vₚᴾᴵᶜ = Σᵢ wᵢₚ vᵢ*`
- FLIP velocity `vₚꟳᴸᴵᴾ = vₚᵒˡᵈ + Σᵢ wᵢₚ (vᵢ* − vᵢᴾ²ᴳ)`, where `vᵢᴾ²ᴳ = pᵢ/mᵢ` is
  the mass-weighted grid velocity BEFORE the force/damping update (`grid.p`,
  `grid.m` still hold the P2G values here). FLIP carries the particle's own
  velocity history instead of re-interpolating it, so it does not bleed the
  sub-affine velocity field to numerical dissipation each step.
- stored velocity is the blend `vₚ = (1−flip)·vₚᴾᴵᶜ + flip·vₚꟳᴸᴵᴾ`
  (`flip=0` ⇒ pure APIC/PIC, unchanged; `flip→1` ⇒ FLIP, least dissipative).
- `Cₚ = (4/h²) Σᵢ wᵢₚ vᵢ* (xᵢ − xₚ)ᵀ` (APIC affine — always from the grid field)
- `xₚ += Δt vₚᴾᴵᶜ` (advect with the grid/PIC velocity, as in standard FLIP)

Does **not** update `F`/stress — that is `Constitutive.update_particles!`.
"""
function g2p!(grid::Grid, pts::Particles, dt::Float64; flip::Float64=0.0)
    h = grid.h
    inv_D = 4.0 / (h * h)
    @inbounds for p in eachindex(pts.x)
        xp = pts.x[p]
        st = bspline_stencil(grid, xp, p)
        vpic = zero(SVector{3,Float64})     # Σ w vᵢ*        (PIC gather)
        dvflip = zero(SVector{3,Float64})   # Σ w (vᵢ* − vᵢᴾ²ᴳ)  (FLIP increment)
        Bp = zero(SMatrix{3,3,Float64,9})
        for k in 1:27
            i = st.idx[k]
            w = st.w[k]
            vi = grid.v[i]
            dx = st.pos[k] - xp
            vpic += w * vi
            Bp += (w * vi) * dx'
            mi = grid.m[i]
            vi_p2g = mi > 0.0 ? grid.p[i] / mi : zero(SVector{3,Float64})
            dvflip += w * (vi - vi_p2g)
        end
        vflip = pts.v[p] + dvflip
        pts.v[p] = (1.0 - flip) * vpic + flip * vflip
        pts.C[p] = inv_D * Bp
        pts.x[p] = xp + dt * vpic
    end
    return pts
end

end # module
