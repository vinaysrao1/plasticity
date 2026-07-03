# Necking of a notched titanium round bar — MPM counterpart of the FEM example
# `lagrangian/examples/necking_titanium_symmetric.jl`. Both solvers run this identical
# problem so the neck compares apples-to-apples (% radius reduction + peak ᾱ).
#
# PERFECTLY SYMMETRIC setup:
#   • geometry : round bar, length L=50, with a SMOOTH PARABOLIC profile — center `taper`
#                thinner than the end radius R0=5 (default 12%) — so mid-length is the
#                global-min section that localizes the neck.
#   • material : Ti-6Al-4V — E=113.8 GPa, ν=0.342, σy0=880 MPa, Hiso=1000.
#   • TENSION on BOTH ends: each grip band is pulled ±δ/2, so the motion is symmetric
#     about the mid-plane (no one-sided dynamic bias). Ends are driven AXIALLY ONLY —
#     free to contract (no lateral grip ⇒ no grip stress raiser).
#   • in-plane rigid-body modes removed by SYMMETRY-PLANE ROLLERS: u_y=0 on y=0, u_z=0
#     on z=0. We sample only the y≥0, z≥0 QUADRANT (the model IS one quadrant of the
#     bar), so these rollers are the quadrant's symmetry BCs — they remove y/z translation
#     and rotation about the axis with zero over-constraint (the neck is axisymmetric).
#
# DYNAMIC relaxation (not Newton): a slow half-sine velocity ramp over N_periods natural
# periods with light damping drives KE/IE ≪ 1 (quasi-static gate). The background grid
# resets each step, so there is no mesh to invert.
#
# Residual MPM-specific detail: grid nodes are fixed in space, so the axial drive uses a
# short grip BAND (≥ B-spline support), unbounded into the x-padding so it stays with the
# grip material as it translates. Its inner edge is a mild stress raiser — kept short and
# far from the notch. (This is the one thing the FEM end-face BC does more cleanly.)
#
# COARSE quick-validation resolution (h=1.25 ⇒ ~4k particles); reduce `h` / raise
# `N_periods` for a converged run.
#
# Run:  julia --project=. examples/necking_titanium_bar.jl

using ParticlePlasticity
using StaticArrays
using Printf

# --- geometry / material / loading (all MATCHED to the FEM example) ---
R0, L = 5.0, 50.0                          # R0 = END radius (bar is thickest at the ends)
taper = parse(Float64, get(ENV, "TAPER", "0.12"))   # center is `taper` THINNER than the ends
# smooth parabolic profile: R(x) = R0·(1 − taper·(1 − s²)), s = (x−L/2)/(L/2) ∈ [−1,1].
sprofile(x) = (sx = (x - L / 2) / (L / 2); 1 - taper * (1 - sx^2))   # radius factor / R0
E, ν, σy0, Hiso = 113.8e3, 0.342, 880.0, 1000.0
mat = J2Material(E = E, ν = ν, σy0 = σy0, Hiso = Hiso)
ρ = 4.43e-9                                # Ti-6Al-4V density (tonne/mm³)
elong = 0.09 * L                           # 9% nominal elongation (= 4.5 mm)

# --- discretization (COARSE quick pass; env-overridable for sweeps) ---
h = parse(Float64, get(ENV, "H", "1.25"))
grip = parse(Float64, get(ENV, "GRIP", "3.0"))   # axial drive-band width at each end
# SYMMETRIC drive: each end moves ±elong/2, so the x-padding must cover the pull on BOTH
# sides (the low end travels to x = -elong/2, the high end to x = L + elong/2).
half = elong / 2
pad_yz = 1.0
xlo = -(half + 1.0)
xhi = L + half + 1.0
nx = Int(round((xhi - xlo) / h)) + 1
ny = Int(round((R0 + 2pad_yz) / h)) + 1
nz = Int(round((R0 + 2pad_yz) / h)) + 1
grid = Grid(SVector(xlo, -pad_yz, -pad_yz), h, (nx, ny, nz))

# Sample a STRAIGHT quarter-cylinder (radius R0 = end radius), then imprint the smooth
# parabolic taper by continuously scaling each particle's (y,z) by s(x)=Rprofile/R0 and its
# volume/mass by s². Continuous scaling imprints the profile at any resolution (a shallow
# taper is far smaller than the particle spacing, so an include/exclude predicate could not).
straight(p) = (p[2]^2 + p[3]^2) <= R0^2
pts = sample_region(straight, SVector(0.0, 0.0, 0.0), SVector(L, R0, R0), h; ppc = 2, ρ = ρ)
np = length(pts.x)
for p in 1:np
    x = pts.x[p][1]
    s = sprofile(x)                        # parabolic radius factor (min at center)
    pts.x[p] = SVector(x, pts.x[p][2] * s, pts.x[p][3] * s)
    pts.V0[p] *= s^2                       # tapered local geometry ⇒ rescale V0/m by s²
    pts.m[p] = ρ * pts.V0[p]
end
x0 = copy(pts.x)                           # reference positions (for binning)

# --- quasi-static half-sine velocity ramp (∫ v dt over ramp = elong) ---
c_p = sqrt((mat.K + 4mat.G / 3) / ρ)
dt = 0.2 * h / c_p
Tnat = L / c_p
N_periods = parse(Int, get(ENV, "NPERIODS", "60"))   # ramp length in natural periods (quasi-static)
T_ramp = N_periods * Tnat
T_hold = N_periods * Tnat                  # hold to let transients decay
# half-sine grip SPEED; each end moves ±elong/2, so ∫(vhalf) over the ramp = elong/2.
vhalf = t -> t <= T_ramp ? half * (pi / (2T_ramp)) * sin(pi * t / T_ramp) : 0.0

damping = parse(Float64, get(ENV, "DAMP", "0.005"))  # viscous grid-velocity damping (env-overridable)
flip = parse(Float64, get(ENV, "FLIP", "0.9"))       # PIC/FLIP blend (0=APIC; →1=FLIP, less dissipative)
model = MPMModel(grid, pts, mat; dt = dt, fbar = true, damping = damping, mass_scale = 1.0, flip = flip)
# symmetry rollers on the two cut planes (lateral constraint; grips drive AXIALLY only).
fix!(model, p -> p[2] < 1e-9, :y)
fix!(model, p -> p[3] < 1e-9, :z)
# SYMMETRIC axial drive: pull both grip bands apart at ±vhalf, unbounded into the padding
# so they stay with the grip material as it translates.
lo_grip = p -> p[1] < grip + 1e-9
hi_grip = p -> p[1] > L - grip - 1e-9
prescribe!(model, lo_grip, :x, t -> -vhalf(t))
prescribe!(model, hi_grip, :x, t ->  vhalf(t))

nsteps = Int(round((T_ramp + T_hold) / dt))
@printf("necking_titanium (MPM): %d particles, %d steps, dt=%.3e, elongation=%.2f (%.0f%%)\n",
        np, nsteps, dt, elong, 100elong / L)

# --- run, averaging deformed state over the final ~10% of steps ---
nbin = 20
edges = range(0, L; length = nbin + 1)
navg = max(1, Int(round(0.1 * nsteps)))
accum_r = zeros(np)          # deformed radius  √(y²+z²)
accum_alpha = zeros(np)
nacc = 0
for s in 1:nsteps
    global nacc
    try
        step!(model)
    catch e
        tn = string(nameof(typeof(e)))
        if tn in ("ParticleInversionError", "ParticleOutOfBoundsError", "FBarCellOutOfBoundsError")
            println("stopped at step $s (reached a numerical limit): ", e)
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
            y, z = model.particles.x[p][2], model.particles.x[p][3]
            accum_r[p] += sqrt(y^2 + z^2)
            accum_alpha[p] += model.particles.ᾱ[p]
        end
        nacc += 1
    end
end

# --- neck metrics (match the FEM example: min cross-section radius, peak ᾱ) ---
r_avg = accum_r ./ nacc
alpha_avg = accum_alpha ./ nacc
Rbin = fill(0.0, nbin)                    # outer (max) deformed radius per axial bin
αbin = fill(0.0, nbin)
for p in 1:np
    b = clamp(searchsortedlast(edges, x0[p][1]), 1, nbin)
    Rbin[b] = max(Rbin[b], r_avg[p])      # outer surface = max radius in the bin
    αbin[b] = max(αbin[b], alpha_avg[p])
end
neck_R, imin = findmin(Rbin)
αmax, imax = findmax(αbin)

println("\n=== necking of a notched titanium round bar (MPM, finite strain, F̄) ===")
@printf("material             : Ti-6Al-4V  E=113.8 GPa, ν=0.342, σy0=880 MPa, Hiso=1000\n")
@printf("final KE/IE          : %.3e  (quasi-static gate = 1e-2)\n",
        model.IE > 0 ? kinetic_energy(model) / model.IE : NaN)
@printf("max eq. plastic strn : %.4f  at x ≈ %.2f (notch at L/2 = %.1f)\n",
        αmax, 0.5 * (edges[imax] + edges[imax+1]), L / 2)
@printf("neck radius (min)    : %.4f at x ≈ %.2f   vs R0 = %.1f  ⇒ %.1f%% reduction\n",
        neck_R, 0.5 * (edges[imin] + edges[imin+1]), R0, 100 * (1 - neck_R / R0))

@printf("\nbin |  x   | radius | ᾱ\n")
for b in 1:nbin
    @printf("  %2d | %5.2f | %.4f | %.4f\n", b, 0.5 * (edges[b] + edges[b+1]), Rbin[b], αbin[b])
end

out = write_particles_vtu(joinpath(@__DIR__, "necking_titanium_bar"), model.particles)
println("\nwrote ", out)
