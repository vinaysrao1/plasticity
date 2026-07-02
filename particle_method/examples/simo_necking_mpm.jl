# Simo (1988) round-bar necking with SATURATION (Voce) HARDENING, via MPM —
# the "V3b" benchmark deferred from the original design (DESIGN §15, PROPOSAL
# §8): the published Simo curve needs nonlinear (Voce) isotropic hardening,
# which the shared `lagrangian` kernel now provides (Materials.jl saturation
# hardening). Because explicit MPM reuses that kernel stress-only, this MPM run
# gets saturation hardening "for free" — no MPM-side change beyond passing the
# saturation material.
#
# Companion to `lagrangian/examples/simo_necking_bar.jl` (the same material, the
# full bar, solved by the FEM). Cross-check: both localize a neck at the notch
# with a comparable radius reduction.
#
# SYMMETRY HALF-MODEL (why this differs from the FEM example's full bar). Simo's
# benchmark is symmetric about mid-length, and the faithful way to model it is a
# HALF bar with a symmetry plane at the notch. That is also what makes it work in
# MPM: an earlier full-bar version localized the neck near the DRIVEN grip instead
# of at the notch — at this coarse resolution (h=2, ~6 cells across the diameter)
# MPM's numerical diffusion makes a 1.8% (even a 5%) geometric notch too weak to
# fix the neck site against a driven-side strain bias (the boundary-stencil
# artifact documented in G6 / necking_extreme). A *symmetry plane* is an exact
# kinematic constraint, not a weak geometric perturbation, so putting the notch on
# the symmetry plane pins the neck there — both faithful and robust. The modeled
# domain is x ∈ [0, L/2]: x=0 is the mid-length symmetry plane (and the notch);
# x=L/2 is the pulled grip.
#
# Material (Simo 1988 / Simo & Armero 1992): E=206.9 GPa, ν=0.29, σy0=450 MPa,
#   σy(ᾱ) = 450 + (715−450)(1−e^{−16.93·ᾱ}) + 129.24·ᾱ  [MPa].
# Geometry: solid circular bar R0=6.413 mm, full length L0=53.334 mm (half modeled),
# ~1.8% radius imperfection (Gaussian) centered on the symmetry plane.
#
# Run:  julia --project=. examples/simo_necking_mpm.jl
# Output: simo_necking_mpm.vtu — color by EqPlasticStrain (neck), J (isochoric).

using ParticlePlasticity
using StaticArrays
using Printf
using Statistics

L0, R0 = 53.334, 6.413
Lm = L0 / 2                                   # modeled half-length (symmetry at x=0)
amp, xwid = 0.018, 2.0                        # 1.8% radius imperfection at the symmetry plane
mat = J2Material(E=206.9e3, ν=0.29, σy0=450.0, σsat=715.0, δ=16.93, Hiso=129.24)
ρ = 7.85e-9

Rx(x) = R0 * (1 - amp * exp(-(x / xwid)^2))    # notch minimum at the x=0 symmetry plane
incyl(p) = (p[2]^2 + p[3]^2) <= Rx(p[1])^2     # inside the tapered cylinder

h = 2.0                                        # ~6-7 cells across the diameter
grip = 3.0                                     # rigid grip-band width at the driven end
elong = 0.10 * Lm                              # 10% nominal elongation (matches the FEM full-bar 10%)

pad_r = 2.0
padx_lo, padx_hi = 2.0, elong + 3.0
nx = Int(round((Lm + padx_lo + padx_hi) / h)) + 1
nyz = Int(round((2R0 + 2pad_r) / h)) + 1
grid = Grid(SVector(-padx_lo, -(R0 + pad_r), -(R0 + pad_r)), h, (nx, nyz, nyz))

pts = sample_region(incyl, SVector(0.0, -R0, -R0), SVector(Lm, R0, R0), h; ppc=2, ρ=ρ)
np = length(pts.x)
x0 = copy(pts.x)

K, G = mat.K, mat.G
c_p = sqrt((K + 4G/3) / ρ)
dt = 0.2 * h / c_p
Tnat = Lm / c_p
N_periods = 200
T_ramp = N_periods * Tnat
T_hold = N_periods * Tnat
vfun = t -> t <= T_ramp ? elong * (pi / (2T_ramp)) * sin(pi * t / T_ramp) : 0.0

model = MPMModel(grid, pts, mat; dt=dt, fbar=true, damping=0.02, mass_scale=1.0)
fix!(model, x -> x[1] < 1e-9, :x)                          # symmetry plane at x=0 (notch): no axial motion,
                                                            # lateral FREE so the cross-section can neck
prescribe!(model, x -> x[1] > Lm - grip - 1e-9, :x, vfun)  # pull the driven grip (translating band)
fix!(model, x -> x[1] > Lm - grip - 1e-9, :y)              # lateral grip at the driven end (rigid-body pin)
fix!(model, x -> x[1] > Lm - grip - 1e-9, :z)

nsteps = Int(round((T_ramp + T_hold) / dt))
@printf("simo_necking_mpm (half-model): %d particles, %d steps, dt=%.3e, elong=%.2f mm (%.0f%% nominal)\n",
        np, nsteps, dt, elong, 100elong/Lm)
@printf("yield σy: ᾱ=0 → 450,  ᾱ=0.1 → %.1f,  ᾱ=0.3 → %.1f MPa (saturation)\n",
        450 + (715-450)*(1-exp(-16.93*0.1)) + 129.24*0.1,
        450 + (715-450)*(1-exp(-16.93*0.3)) + 129.24*0.3)

nbin = 10
edges = range(0, Lm; length=nbin+1)
navg = max(1, Int(round(0.1 * nsteps)))
accum_r = zeros(np)
accum_alpha = zeros(np)
nacc = Ref(0)
let nsteps=nsteps
    for s in 1:nsteps
        step!(model)
        if s % (nsteps ÷ 10) == 0 || s == nsteps
            ke = kinetic_energy(model)
            @printf("  step %d  t=%.6f  KE/IE=%.3e\n", s, model.t, model.IE > 0 ? ke/model.IE : NaN)
        end
        if s > nsteps - navg
            for p in 1:np
                accum_r[p] += sqrt(model.particles.x[p][2]^2 + model.particles.x[p][3]^2)
                accum_alpha[p] += model.particles.ᾱ[p]
            end
            nacc[] += 1
        end
    end
end
r_avg = accum_r ./ nacc[]
alpha_avg = accum_alpha ./ nacc[]

Rbin = fill(0.0, nbin); αbin = fill(0.0, nbin)
for p in 1:np
    b = clamp(searchsortedlast(edges, x0[p][1]), 1, nbin)
    Rbin[b] = max(Rbin[b], r_avg[p])
    αbin[b] = max(αbin[b], alpha_avg[p])
end
Rmin, imin = findmin(Rbin)
αmax, imax = findmax(αbin)
Rends = maximum(Rbin)

@printf("\nfinal KE/IE = %.3e\n", model.IE > 0 ? kinetic_energy(model)/model.IE : NaN)
@printf("neck at bin %d (x≈%.1f, symmetry plane/notch at x=0), min radius %.4f vs R0 %.3f ⇒ %.1f%% reduction\n",
        imin, 0.5*(edges[imin]+edges[imin+1]), Rmin, R0, 100*(1 - Rmin/Rends))
@printf("peak ᾱ = %.4f at bin %d\n", αmax, imax)
@printf("\nbin |  x   | outer R | ᾱ      (x=0 is the mid-length symmetry plane/notch)\n")
for b in 1:nbin
    @printf("  %2d | %4.1f | %.4f | %.4f\n", b, 0.5*(edges[b]+edges[b+1]), Rbin[b], αbin[b])
end

out = write_particles_vtu("simo_necking_mpm", model.particles)
println("\nwrote ", out)
