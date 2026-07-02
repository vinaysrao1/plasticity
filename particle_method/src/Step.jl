"""
    Step

Orchestration: `MPMModel` (grid + particles + material + BCs + settings,
DESIGN §9), `step!` (one explicit MPM step, DESIGN §4 exact order), `run!`
(driver loop with the `KE/IE` quasi-static diagnostic, DESIGN §6), and the
postprocessing helpers `kinetic_energy`, `particle_cauchy`,
`equivalent_plastic_strain`.

**Settings are plain fields, not `Val`-dispatched** (DESIGN §9): `fbar`,
`damping`, `mass_scale` are ordinary `Bool`/`Float64` fields branched on once
per step, outside the 27-node P2G/G2P inner loops.
"""
module Step

using StaticArrays
using ..GridMod: Grid
using ..ParticlesMod: Particles
using ..Transfer: p2g!, g2p!
using ..Constitutive: update_particles!, ncells
using ..BoundaryConditions: VelocityBC, ForceBC, apply_bcs!, apply_loads!
import ..BoundaryConditions: fix!, symmetry!, prescribe!, load!
using PlasticityFEM.Materials: J2Material

export MPMModel, fix!, symmetry!, prescribe!, load!, gravity!, step!, run!,
       kinetic_energy, particle_cauchy, equivalent_plastic_strain

"""
    MPMModel(grid, particles, material; dt, fbar=false, damping=0.0,
             mass_scale=1.0, gravity=zero(SVector{3,Float64}))

The single object examples drive (DESIGN §9): bundles the background `Grid`,
the `Particles`, the (reused) `J2Material`, an accumulated BC list, and the
run settings. `mass_scale` is the informational `ρ_eff/ρ_true` ratio (DESIGN
§6) recorded for diagnostics — the actual scaling is baked into `particles.m`
and `dt` at construction time by the caller (`sample_box`/`sample_region` with
a scaled `ρ`, and `dt` from the §4.3 CFL formula with the same `ρ_eff`); the
model does not re-derive either.
"""
mutable struct MPMModel
    grid::Grid
    particles::Particles
    material::J2Material
    bcs::Vector{VelocityBC}
    loads::Vector{ForceBC}
    gravity::SVector{3,Float64}
    dt::Float64
    fbar::Bool
    damping::Float64
    mass_scale::Float64
    t::Float64
    step_count::Int
    IE::Float64                                   # accumulated internal energy (§6 diagnostic)
    # reusable per-step buffers (Constitutive.update_particles!) — no per-step alloc
    Ftrial::Vector{SMatrix{3,3,Float64,9}}
    Jnum::Vector{Float64}
    Jden::Vector{Float64}
end

function MPMModel(grid::Grid, particles::Particles, material::J2Material;
                   dt::Float64, fbar::Bool=false, damping::Float64=0.0,
                   mass_scale::Float64=1.0,
                   gravity::SVector{3,Float64}=zero(SVector{3,Float64}))
    np = length(particles)
    nc = ncells(grid)
    return MPMModel(grid, particles, material, VelocityBC[], ForceBC[], gravity, dt, fbar,
                     damping, mass_scale, 0.0, 0, 0.0,
                     Vector{SMatrix{3,3,Float64,9}}(undef, np),
                     zeros(Float64, nc), zeros(Float64, nc))
end

# --- BC / loading builders on the model (DESIGN §7, §9) ---

"""
    fix!(model, pred, comp=:all)

Zero grid-node velocity component(s) `comp` for nodes satisfying `pred`
(DESIGN §7). Extends `BoundaryConditions.fix!` with an `MPMModel` method.
"""
fix!(model::MPMModel, pred, comp::Symbol=:all) = (fix!(model.bcs, model.grid, pred, comp); model)

"""
    symmetry!(model, pred, axis)

Roller / symmetry-plane BC (alias of `fix!` for one normal component).
"""
symmetry!(model::MPMModel, pred, axis::Symbol) = (symmetry!(model.bcs, model.grid, pred, axis); model)

"""
    prescribe!(model, pred, comp, v)

Displacement-control velocity BC (DESIGN §7): `v` is a constant or `t -> Float64` ramp.
"""
prescribe!(model::MPMModel, pred, comp::Symbol, v) = (prescribe!(model.bcs, model.grid, pred, comp, v); model)

"""
    load!(model, pred, comp, value; distribute=true)

Nodal force / traction load (DESIGN §7: "Traction/body loads enter as grid
forces `fᵢ += ...`", mirroring `lagrangian`'s `load!`). `value` is the total
force in direction `comp` over the nodes satisfying `pred`, divided evenly
across them when `distribute=true` (default).
"""
load!(model::MPMModel, pred, comp::Symbol, value; distribute::Bool=true) =
    (load!(model.loads, model.grid, pred, comp, value; distribute=distribute); model)

"""
    gravity!(model, g)

Set the uniform body-force acceleration (`SVector{3,Float64}`) added as a
grid force each step (DESIGN §4: `fᵢ += mᵢ g`).
"""
gravity!(model::MPMModel, g::SVector{3,Float64}) = (model.gravity = g; model)

# --- the explicit MPM step (DESIGN §4, exact order) ---

"""
    step!(model) -> model

One explicit MPM step, in the exact order of DESIGN §4:
P2G (accumulate mass/momentum/internal force) → finalize grid velocity →
body force → explicit momentum update → **damping** → **BCs (applied last)**
→ G2P (PIC velocity, APIC `Cₚ`, advect) → particle F-update + (optional F̄) +
reused constitutive kernel → grid reset (buffers zeroed at the *start* of the
next call, matching "reset the grid" at the end of DESIGN §4's pseudocode).
"""
function step!(model::MPMModel)
    g = model.grid
    pts = model.particles
    dt = model.dt

    # --- reset the grid (DESIGN §4: zero mᵢ, pᵢ, fᵢ, vᵢ) ---
    fill!(g.m, 0.0)
    fill!(g.p, zero(SVector{3,Float64}))
    fill!(g.f, zero(SVector{3,Float64}))
    fill!(g.v, zero(SVector{3,Float64}))

    # --- P2G ---
    p2g!(g, pts)

    # --- external nodal loads (DESIGN §7: fᵢ += ...), added before the
    # explicit momentum update alongside gravity ---
    apply_loads!(g, model.loads, model.t)

    # --- grid momentum update (explicit): finalize v, gravity, Δt f/m, damping ---
    mmax = isempty(g.m) ? 0.0 : maximum(g.m)
    mtol = mmax > 0 ? mmax * 1e-10 : 0.0
    @inbounds for i in eachindex(g.m)
        if g.m[i] > mtol
            v = g.p[i] / g.m[i]
            f = g.f[i] + g.m[i] * model.gravity
            vstar = v + dt * f / g.m[i]
            if model.damping > 0
                vstar = (1.0 - model.damping) * vstar
            end
            g.v[i] = vstar
        else
            g.v[i] = zero(SVector{3,Float64})
        end
    end

    # --- BCs, applied LAST (after damping) — DESIGN §4/§7's corrected order ---
    apply_bcs!(g, model.bcs, model.t)

    # --- G2P ---
    g2p!(g, pts, dt)

    # --- particle constitutive update (F-update, F̄, reused kernel) ---
    update_particles!(pts, model.material, dt, g; fbar=model.fbar,
                       Ftrial=model.Ftrial, Jnum=model.Jnum, Jden=model.Jden)

    # --- KE/IE diagnostic (DESIGN §6): dIE = Σ_p V_p (σ_p : Dₚ) dt, Dₚ=sym(Cₚ) ---
    dIE_rate = 0.0
    @inbounds for p in eachindex(pts.x)
        σ = pts.τ[p] / pts.J[p]                      # Cauchy, 6-Voigt (physical shear)
        Cp = pts.C[p]
        D = 0.5 * (Cp + Cp')
        power = σ[1] * D[1, 1] + σ[2] * D[2, 2] + σ[3] * D[3, 3] +
                2 * (σ[4] * D[1, 2] + σ[5] * D[2, 3] + σ[6] * D[1, 3])
        Vp = pts.J[p] * pts.V0[p]
        dIE_rate += Vp * power
    end
    model.IE += dt * dIE_rate

    model.t += dt
    model.step_count += 1
    return model
end

"""
    run!(model, nsteps; diagnostics_every=0, callback=nothing) -> model

Drive `step!` `nsteps` times. If `diagnostics_every > 0`, prints `t` and the
`KE/IE` quasi-static diagnostic (DESIGN §6) every that many steps.
`callback(model, step)`, if given, runs after every step (e.g. for `.vtu`
snapshots or custom logging).
"""
function run!(model::MPMModel, nsteps::Int; diagnostics_every::Int=0, callback=nothing)
    for s in 1:nsteps
        step!(model)
        if diagnostics_every > 0 && (s % diagnostics_every == 0 || s == nsteps)
            KE = kinetic_energy(model)
            ratio = model.IE > 0 ? KE / model.IE : NaN
            println("  step $(model.step_count)  t=$(round(model.t, digits=6))  ",
                    "KE=$(KE)  IE=$(model.IE)  KE/IE=$(ratio)")
        end
        callback !== nothing && callback(model, s)
    end
    return model
end

# --- postprocessing (DESIGN §9 curated exports) ---

"""
    kinetic_energy(model) -> Float64

`Σₚ ½ mₚ ‖vₚ‖²`.
"""
function kinetic_energy(model::MPMModel)
    pts = model.particles
    KE = 0.0
    @inbounds for p in eachindex(pts.x)
        v = pts.v[p]
        KE += 0.5 * pts.m[p] * (v[1]^2 + v[2]^2 + v[3]^2)
    end
    return KE
end

"""
    particle_cauchy(pts_or_model, p) -> SVector{6,Float64}

Cauchy stress `σₚ = τₚ/Jₚ` (6-Voigt, physical shear) of particle `p` from the
last committed constitutive update.
"""
particle_cauchy(pts::Particles, p::Int) = pts.τ[p] / pts.J[p]
particle_cauchy(model::MPMModel, p::Int) = particle_cauchy(model.particles, p)

"""
    equivalent_plastic_strain(model) -> Vector{Float64}

Per-particle equivalent plastic strain `ᾱ` (a copy of the committed history).
"""
equivalent_plastic_strain(model::MPMModel) = copy(model.particles.ᾱ)

end # module
