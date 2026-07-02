# Extreme necking: a round-ish bar pulled well past where the `lagrangian`
# FEM mesh solver can follow, with the Material Point Method (MPM).
#
# Same setup as G6 (test/test_G6_necking.jl) — a 10×1×1 bar, ~2% mid-length
# cross-section imperfection to seed the neck, steel (E=210e3, ν=0.3,
# σy0=250, Hiso=2000 MPa), symmetry rollers on the three min faces, F̄ for
# the fully-plastic isochoric flow — but pulled to a MUCH larger elongation.
# `lagrangian/examples/finite_necking_bar.jl` documents its own ceiling: past
# early localization, the load maximum is a SOFTENING instability, and a
# plain full-Newton solver (no arc-length continuation) overshoots into an
# invalid (inverted-element) configuration and diverges — a hard wall for
# the mesh. MPM's background grid resets every step, so there is no mesh to
# invert; only individual PARTICLES can (§3.3), and deep necking pulls the
# cross-section down, not any one particle's local F, so it is far more
# resistant to this failure mode.
#
# A SECOND BOUNDARY-CONDITION FIX, DISTINCT FROM bent_rod.jl's. G6's driven
# face (`x -> x[1] > L-1e-9`) is fine at G6's small elongation (0.25, under
# the B-spline support radius 1.5h), but at THIS example's much larger
# elongation the driven grid-node plane — fixed in space, since grid nodes
# never move — would be left behind by the tip material exactly as in
# bent_rod.jl's original bug, except here the axis of travel (x) is the SAME
# axis used to SELECT the driven nodes, so bent_rod's fix (drop the
# transverse restriction) does not apply: dropping the x-restriction would
# select the wrong nodes entirely. The correct fix for a driven face moving
# ALONG its own selection axis: widen the selected region into a "grip" band
# near the end (physically a real tensile-test grip: a rigid, translating
# clamp far from the gauge section, not a knife-edge at the exact tip) AND
# extend it, unbounded, into the padding beyond the original tip position —
# so wherever the grip material ends up after however much translation, it
# remains inside the driven node set. The grip band is narrow relative to
# the gauge length so it does not interfere with the imperfection's necking.
#
# Run:  julia --project=. examples/necking_extreme.jl
# Output: necking_extreme.vtu — color by `EqPlasticStrain` for the localized
# neck, `J` (det F) to see the volume-preservation of the isochoric flow.

using ParticlePlasticity
using StaticArrays
using Printf
using Statistics

L, w = 10.0, 1.0
E, ν, σy0, Hiso = 210e3, 0.3, 250.0, 2000.0
amp, x0i, xwid = 0.02, L / 2, L / 5
elong = 2.5                          # 25% nominal elongation — far past G6's 2.5%
mat = J2Material(E=E, ν=ν, σy0=σy0, Hiso=Hiso)
ρ = 7.85e-9

h = 1.0 / 3
grip = 1.0                           # rigid-grip band width near each end
pad_x, pad_yz = elong + 1.0, 1.0     # x-padding must cover the full elongation
nx = Int(round((L + pad_x + 1.0) / h)) + 1
ny = Int(round((w + 2pad_yz) / h)) + 1
nz = Int(round((w + 2pad_yz) / h)) + 1
grid = Grid(SVector(-1.0, -pad_yz, -pad_yz), h, (nx, ny, nz))

pts = sample_box(SVector(0.0, 0.0, 0.0), SVector(L, w, w), h; ppc=2, ρ=ρ)
np = length(pts.x)
for p in 1:np
    x = pts.x[p][1]
    s = 1.0 - amp * exp(-((x - x0i) / xwid)^2)
    pts.x[p] = SVector(pts.x[p][1], pts.x[p][2] * s, pts.x[p][3] * s)
    # rescale V0/m by s^2 to match the tapered local geometry (G6's fix, see
    # test_G6_necking.jl for the full derivation)
    pts.V0[p] *= s^2
    pts.m[p] = ρ * pts.V0[p]
end
x0 = copy(pts.x)

K, G = mat.K, mat.G
c_p = sqrt((K + 4G / 3) / ρ)
dt = 0.2 * h / c_p
Tnat = L / c_p
N_periods = 600                      # long, gentle ramp -- large plastic strain accumulates
T_ramp = N_periods * Tnat
T_hold = N_periods * Tnat
vfun = t -> t <= T_ramp ? elong * (pi / (2T_ramp)) * sin(pi * t / T_ramp) : 0.0

model = MPMModel(grid, pts, mat; dt=dt, fbar=true, damping=0.02, mass_scale=1.0)
fix!(model, x -> x[1] < grip - 1e-9, :x)              # rigid grip band, root end
fix!(model, x -> x[2] < 1e-9, :y)
fix!(model, x -> x[3] < 1e-9, :z)
prescribe!(model, x -> x[1] > L - grip - 1e-9, :x, vfun)   # rigid grip band, driven end,
                                                            # unbounded above -- covers the
                                                            # padding the grip translates into

nsteps = Int(round((T_ramp + T_hold) / dt))
@printf("necking_extreme: %d particles, %d steps, dt=%.3e, target elongation=%.2f (%.0f%%)\n",
        np, nsteps, dt, elong, 100elong / L)

nbin = 10
edges = range(0, L; length=nbin + 1)
navg = max(1, Int(round(0.1 * nsteps)))
accum_y = zeros(np)
accum_alpha = zeros(np)
nacc = Ref(0)
inverted = Ref(false)
let nsteps=nsteps
    for s in 1:nsteps
        try
            step!(model)
        catch e
            # Only treat the known "legitimate physical/numerical limit"
            # exceptions (DESIGN §3.3, §8) as a graceful stop -- pushing to
            # genuinely extreme deformation until a particle inverts or
            # escapes the grid is a real outcome worth reporting, not a bug
            # to hide. Anything else propagates normally (a real bug).
            tn = string(nameof(typeof(e)))
            if tn in ("ParticleInversionError", "ParticleOutOfBoundsError", "FBarCellOutOfBoundsError")
                println("stopped at step $s (reached a numerical limit): ", e)
                inverted[] = true
                break
            else
                rethrow()
            end
        end
        if s % (nsteps ÷ 20) == 0 || s == nsteps
            ke = kinetic_energy(model)
            @printf("  step %d  t=%.6f  KE/IE=%.3e\n", s, model.t, model.IE > 0 ? ke / model.IE : NaN)
        end
        if s > nsteps - navg
            for p in 1:np
                accum_y[p] += model.particles.x[p][2]
                accum_alpha[p] += model.particles.ᾱ[p]
            end
            nacc[] += 1
        end
    end
end

if nacc[] > 0
    y_avg = accum_y ./ nacc[]
    alpha_avg = accum_alpha ./ nacc[]

    halfw = fill(0.0, nbin)
    αbin = fill(0.0, nbin)
    for p in 1:np
        b = clamp(searchsortedlast(edges, x0[p][1]), 1, nbin)
        halfw[b] = max(halfw[b], y_avg[p])
        αbin[b] = max(αbin[b], alpha_avg[p])
    end
    wmin, imin = findmin(halfw)
    αmax, imax = findmax(αbin)
    contraction = 100 * (1 - wmin / maximum(halfw))

    @printf("\nfinal KE/IE = %.3e\n", model.IE > 0 ? kinetic_energy(model) / model.IE : NaN)
    @printf("neck at bin %d (x≈%.2f), contraction = %.1f%% (half-width %.4f vs ends ~%.4f)\n",
            imin, 0.5 * (edges[imin] + edges[imin+1]), contraction, wmin, maximum(halfw))
    @printf("peak ᾱ = %.4f at bin %d\n", αmax, imax)
    @printf("\nbin | half-width | ᾱ\n")
    for b in 1:nbin
        xm = 0.5 * (edges[b] + edges[b+1])
        @printf("  x=%5.2f | %.4f | %.4f\n", xm, halfw[b], αbin[b])
    end
end

out = write_particles_vtu("necking_extreme", model.particles)
println("\nwrote ", out)
