# Bent rod: a slender cantilever bent under self-weight into a plastic hinge,
# with the Material Point Method (MPM).
#
# Same L=8, 1×1 cross-section, clamped-at-x=0 geometry validated against the
# `lagrangian` FEM solver in G5 (test/test_G5_cantilever.jl), with a real
# elastoplastic material (steel-like, finite hardening) instead of G5's
# purely-elastic test material.
#
# WHY GRAVITY, NOT A PRESCRIBED TIP DISPLACEMENT (root-caused, not hacked
# around). An earlier version of this example drove the tip with a
# `prescribe!` velocity ramp, the pattern validated in G5. With an ELASTIC
# material (G5) that works well (mid-span-to-tip match FEM to ~7-10%). But
# once the material can actually YIELD (this example), the SAME setup put
# the peak plastic strain at the driven tip, not the clamped root — the
# physically wrong location. Root cause, confirmed directly: a Dirichlet
# velocity BC forced onto a single grid-node plane creates a genuinely steep
# LOCAL velocity gradient between that plane and the nearest free node one
# cell inboard (the "prescribed vs free" transition happens over one cell
# width, ~0.5 units, regardless of how gently or how far the tip is driven —
# confirmed by both a 5x longer ramp and a smaller target displacement,
# neither of which moved the artifact). For an elastic material that local
# strain concentration is recoverable and invisible; for a real
# elastoplastic one it exceeds yield and "eats" real, permanent, irreversible
# plastic strain right at the boundary — a numerical artifact, not physics.
#
# The fix: don't impose a kinematic (velocity) constraint at the free end at
# all. Drive the bend with GRAVITY instead — a smooth body force with no
# node-to-node discontinuity anywhere — which DESIGN.md's own example
# description offers as the alternative to a prescribed tip motion. The tip
# is free; the beam bends under its own weight, and the bending moment (and
# hence the plastic hinge) naturally peaks at the clamped root, exactly where
# physics says it should.
#
# RESULT (one representative run, g≈1.66e9 mm/s², ~10x the first-yield
# bending-stress estimate): reaches quasi-static equilibrium (KE/IE≈0.29%,
# under the 1% gate, still gently settling from a genuine slow bending-mode
# ringdown — accepted for a demonstration, not a strict validation gate) with
# a 1.5-unit (≈19%) tip sag. Peak equivalent plastic strain ᾱ=0.085 lands at
# a particle whose UNDEFORMED position is x≈0.4 — right at the clamped root —
# with ᾱ=0 at every other sampled station (x=2,4,6,8): a clean, physically
# correct plastic hinge confined to the highest-bending-moment site, not
# smeared or misplaced.
#
# Run:  julia --project=. examples/bent_rod.jl
# Output: bent_rod.vtu — open in ParaView; particles already carry their
# deformed positions (no "warp by vector" needed — MPM particles ARE the
# deformed configuration). Color by `EqPlasticStrain` to see the hinge; by
# `VonMises` for the stress field.

using ParticlePlasticity
using StaticArrays
using Printf
using Statistics

L, w = 8.0, 1.0
E, ν = 210e3, 0.3
mat = J2Material(E=E, ν=ν, σy0=250.0, Hiso=1000.0)   # steel-like, finite hardening
ρ = 7.85e-9   # tonne/mm^3 (N, mm, tonne, s, MPa units, matching `lagrangian`'s steel examples)

h = 0.5
padx, pady, padz = 1.0, 1.5, 6.0   # generous z-padding: the tip swings far down
nx = Int(round((L + 2padx) / h)) + 1
ny = Int(round((w + 2pady) / h)) + 1
nz = Int(round((w + 2padz) / h)) + 1
grid = Grid(SVector(-padx, -pady, -padz), h, (nx, ny, nz))
pts = sample_box(SVector(0.0, 0.0, 0.0), SVector(L, w, w), h; ppc=2, ρ=ρ)
np = length(pts.x)
x0 = copy(pts.x)   # reference (undeformed) positions, for the diagnostics below

K, G = mat.K, mat.G
c_p = sqrt((K + 4G / 3) / ρ)
dt = 0.2 * h / c_p
Tnat = L / c_p

# Body-force magnitude: chosen (order-of-magnitude, then tuned empirically) so
# the root bending stress clears σy0 comfortably. Elementary beam theory for a
# uniformly-loaded cantilever (distributed load w_ld = ρ·g·A, cross-section
# A=s², s=1): M_root = w_ld·L²/2, σ_root = M_root·(s/2)/(s⁴/12) = 3ρgL²/s.
# Solving σ_root ≈ 10·σy0 (well past first yield, so a real, visually clear
# hinge develops, not just incipient yielding) for g (an initial 4x estimate
# gave only a modest ~4% sag — visually unconvincing as a "bent rod" — so
# this was tuned up empirically):
g_mag = 10 * mat.σy0 / (3 * ρ * L^2 / w)
@printf("gravity magnitude = %.3e mm/s^2 (order-of-magnitude estimate for ~10x first-yield bending stress)\n", g_mag)

N_periods = 500
T_ramp = N_periods * Tnat
T_hold = N_periods * Tnat
gfun = t -> -g_mag * (t <= T_ramp ? sin(pi * t / (2T_ramp))^2 : 1.0)   # smooth ease-in, then hold

cross(x) = -1e-9 <= x[2] <= w + 1e-9 && -1e-9 <= x[3] <= w + 1e-9   # root cross-section (fixed, never moves)

model = MPMModel(grid, pts, mat; dt=dt, fbar=true, damping=0.02, mass_scale=1.0)
fix!(model, x -> x[1] < 1e-9 && cross(x), :all)
# no prescribe! at the tip — the tip is free; gravity does the bending (see the
# root-cause note above for why a kinematic tip BC is the wrong tool here)

nsteps = Int(round((T_ramp + T_hold) / dt))
@printf("bent_rod: %d particles, %d steps, dt=%.3e\n", np, nsteps, dt)

navg = max(1, Int(round(0.1 * nsteps)))
accum_dz = zeros(np)
accum_alpha = zeros(np)
nacc = Ref(0)
let nsteps=nsteps, np=np, navg=navg
    for s in 1:nsteps
        model.gravity = SVector(0.0, 0.0, gfun(model.t))
        step!(model)
        if s % (nsteps ÷ 10) == 0 || s == nsteps
            ke = kinetic_energy(model)
            @printf("  step %d  t=%.6f  KE/IE=%.3e\n", s, model.t, model.IE > 0 ? ke / model.IE : NaN)
        end
        if s > nsteps - navg
            for p in 1:np
                accum_dz[p] += model.particles.x[p][3] - x0[p][3]
                accum_alpha[p] += model.particles.ᾱ[p]
            end
            nacc[] += 1
        end
    end
end
dz_avg = accum_dz ./ nacc[]
alpha_avg = accum_alpha ./ nacc[]

ke_ie = kinetic_energy(model) / model.IE
@printf("\nfinal KE/IE = %.3e (quasi-static gate: should be << 1%%)\n", ke_ie)

αmax, imax = findmax(alpha_avg)
xhinge = x0[imax][1]
@printf("peak equivalent plastic strain ᾱ = %.4f, at a particle whose UNDEFORMED x = %.2f (root at x=0)\n",
        αmax, xhinge)

tipmask = [x0[p][1] > L - h for p in 1:np]
tip_dz = sum(dz_avg[p] for p in 1:np if tipmask[p]) / count(tipmask)
@printf("tip-band mean z-displacement (self-weight sag) = %.3f\n", tip_dz)

stations = (0.0, 2.0, 4.0, 6.0, 8.0)
@printf("\naxial station | mean z-displacement | mean ᾱ\n")
for xs in stations
    mm = [abs(x0[p][1] - xs) < 0.3 for p in 1:np]
    @printf("  x=%.1f        |  %8.4f          |  %.4f\n",
            xs, mean(dz_avg[p] for p in 1:np if mm[p]), mean(alpha_avg[p] for p in 1:np if mm[p]))
end

out = write_particles_vtu("bent_rod", model.particles)
println("\nwrote ", out)
