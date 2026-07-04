# Taylor bar impact — a copper cylinder fired end-on into a rigid wall, mushrooming
# plastically. The classic large-deformation dynamic plasticity benchmark (G.I. Taylor,
# 1948), and a showcase for MPM: the impact face flattens and flares to several times its
# radius with plastic strains well over 100% — exactly where a Lagrangian FEM mesh inverts,
# while MPM's resetting background grid handles it (and any self-contact of the folding
# mushroom) naturally.
#
# Unlike the necking examples this is a genuine DYNAMIC transient — the deformation is
# inertially driven by the impact, so there is NO quasi-static damping to tune and no
# localization-suppression problem. Explicit MPM is made for this.
#
# SETUP
#   • geometry : solid copper cylinder, radius R0, length L0, axis = x. QUARTER model
#     (y≥0, z≥0 quadrant) with y=0/z=0 symmetry rollers — exact for the axisymmetric problem.
#   • wall     : rigid, at x=0, FRICTIONLESS (fix the grid x-velocity on the x≤0 half-space;
#     the face slides radially but cannot penetrate). The bar starts touching it.
#   • loading  : every particle is given an initial velocity v = (−v0, 0, 0) toward the wall.
#   • material : OFHC copper with saturation (Voce) hardening. RATE- and TEMPERATURE-
#     INDEPENDENT idealization (an effective flow curve) — so the mushroom shape and
#     length ratio are right in character, not calibrated to 1%.
#
# Report: final length L_f (and L_f/L0), mushroom footprint radius, peak ᾱ, and the
# deformed radius profile along the axis.
#
# FINE mesh by default (h=0.15 ⇒ ~25 cells across the radius, ~660k particles). This is a
# heavy run: ≈2 hours single-threaded (MPM's step! is serial) on an Apple M4 Max. For a
# ~30 s coarse sanity pass, override `H=0.8`. Cost scales as (1/h)⁴.
# Run:        julia --project=. examples/taylor_impact.jl          # fine, ~2 h
# Quick look: H=0.8 julia --project=. examples/taylor_impact.jl    # coarse, ~30 s
# Output: taylor_impact.vtu — color by EqPlasticStrain (mushroom), J (det F), VonMises.

using ParticlePlasticity
using StaticArrays
using Printf

# --- geometry ---
L0, R0 = 25.4, 3.8                          # 1 in long, 7.6 mm diameter (classic Taylor specimen)

# --- material: OFHC copper, effective rate-independent Voce hardening ---
E, ν, ρ = 117.0e3, 0.35, 8.96e-9            # MPa, -, tonne/mm³
σy0, σsat, δ, Hiso = 250.0, 550.0, 4.0, 100.0
mat = J2Material(E = E, ν = ν, σy0 = σy0, σsat = σsat, δ = δ, Hiso = Hiso)

# --- impact velocity (mm/s); 2.5e5 mm/s = 250 m/s (ductile mushroom regime) ---
v0 = parse(Float64, get(ENV, "V0", "2.5e5"))

# --- discretization (FINE by default ≈2 h; set H=0.8 for a ~30 s coarse pass) ---
h = parse(Float64, get(ENV, "H", "0.15"))
flip = parse(Float64, get(ENV, "FLIP", "0.9"))   # PIC/FLIP blend (dynamics benefit from FLIP)
damping = parse(Float64, get(ENV, "DAMP", "0.0"))# DYNAMIC: no quasi-static damping

# grid: wall + support below x=0; room for the bar length and the radial mushroom (~2.5 R0)
pad_x = 2.0
Rgrid = 2.6 * R0
nx = Int(round((L0 + 2pad_x) / h)) + 1
nyz = Int(round((Rgrid + h) / h)) + 1
grid = Grid(SVector(-pad_x, -h, -h), h, (nx, nyz, nyz))

# quarter cylinder (y≥0, z≥0), bar face at x=0
incyl(p) = (p[2]^2 + p[3]^2) <= R0^2
pts = sample_region(incyl, SVector(0.0, 0.0, 0.0), SVector(L0, R0, R0), h; ppc = 2, ρ = ρ)
np = length(pts.x)
x0 = copy(pts.x)

# initial velocity: whole bar moving at −v0 into the wall
for p in 1:np
    pts.v[p] = SVector(-v0, 0.0, 0.0)
end

c_p = sqrt((mat.K + 4mat.G / 3) / ρ)
dt = 0.2 * h / c_p
Tend = parse(Float64, get(ENV, "TEND_US", "60.0")) * 1e-6   # simulated time (default 60 µs)
nsteps = Int(round(Tend / dt))

KE0 = 0.5 * sum(pts.m) * v0^2
@printf("taylor_impact: %d particles, %d steps, dt=%.3e s, v0=%.0f m/s, T=%.0f µs\n",
        np, nsteps, dt, v0 / 1e3, Tend * 1e6)
@printf("copper c_p=%.0f m/s  (impact Mach %.3f);  yield σy: ᾱ=0→%.0f, ᾱ=0.5→%.0f, ᾱ=1→%.0f MPa\n",
        c_p / 1e3, v0 / c_p, σy0,
        σy0 + (σsat - σy0) * (1 - exp(-δ * 0.5)) + Hiso * 0.5,
        σy0 + (σsat - σy0) * (1 - exp(-δ * 1.0)) + Hiso * 1.0)

model = MPMModel(grid, pts, mat; dt = dt, fbar = true, damping = damping, mass_scale = 1.0, flip = flip)
fix!(model, p -> p[1] < 1e-9, :x)           # rigid frictionless wall at x=0 (no penetration; radial slip)
fix!(model, p -> p[2] < 1e-9, :y)           # y=0 symmetry roller
fix!(model, p -> p[3] < 1e-9, :z)           # z=0 symmetry roller

# --- run the transient ---
stopped = false
for s in 1:nsteps
    global stopped
    try
        step!(model)
    catch e
        tn = string(nameof(typeof(e)))
        if tn in ("ParticleInversionError", "ParticleOutOfBoundsError", "FBarCellOutOfBoundsError")
            println("stopped at step $s (numerical limit): ", e); stopped = true; break
        else
            rethrow()
        end
    end
    if s % max(1, nsteps ÷ 20) == 0 || s == nsteps
        ke = kinetic_energy(model)
        @printf("  step %d  t=%.2f µs  KE/KE0=%.3e\n", s, model.t * 1e6, ke / KE0)
    end
end

# --- deformed metrics ---
xc = [model.particles.x[p][1] for p in 1:np]
rc = [sqrt(model.particles.x[p][2]^2 + model.particles.x[p][3]^2) for p in 1:np]
ᾱc = model.particles.ᾱ
L_f = maximum(xc)                                       # deformed length (wall at x=0)
R_face = maximum(rc[xc .< 2.0])                         # mushroom footprint radius (slab near wall)
αmax = maximum(ᾱc)

nbin = 12
edges = range(0, L0; length = nbin + 1)
Rbin = fill(0.0, nbin); αbin = fill(0.0, nbin)
for p in 1:np
    b = clamp(searchsortedlast(edges, xc[p]), 1, nbin)  # bin by DEFORMED axial position
    Rbin[b] = max(Rbin[b], rc[p])
    αbin[b] = max(αbin[b], ᾱc[p])
end

println("\n=== Taylor copper bar impact (MPM, finite strain, F̄) ===")
@printf("material          : OFHC copper (rate/temp-independent Voce)  E=117 GPa, ν=0.35\n")
@printf("final KE/KE0      : %.3e%s\n", kinetic_energy(model) / KE0, stopped ? "  (stopped early)" : "")
@printf("final length L_f  : %.3f mm   vs L0 = %.3f  ⇒ L_f/L0 = %.3f  (%.1f%% shortening)\n",
        L_f, L0, L_f / L0, 100 * (1 - L_f / L0))
@printf("mushroom radius   : %.3f mm   vs R0 = %.3f  ⇒ %.2f× flare at the impact face\n",
        R_face, R0, R_face / R0)
@printf("peak eq. plastic strain : %.3f\n", αmax)
@printf("\nbin | x(def) | outer R | ᾱ     (x=0 = impact face / wall)\n")
for b in 1:nbin
    @printf("  %2d | %5.2f | %.4f | %.4f\n", b, 0.5 * (edges[b] + edges[b+1]), Rbin[b], αbin[b])
end

out = write_particles_vtu(joinpath(@__DIR__, "taylor_impact"), model.particles)
println("\nwrote ", out)
