# G1b — F-update volumetric drift (DESIGN §12).
#
# Isolates §4.1 from G1: drive one particle through many steps of a prescribed
# *trace-free* velocity gradient L (a fixed combined shear/rotation/isochoric
# stretch — trace(L)=0 exactly, so the TRUE det F stays exactly 1 for all
# time, independent of the material response: d(ln J)/dt = tr(L) = 0) through
# the *actual* MPM forward-Euler F-update loop
# (Constitutive.particle_stress_update!, i.e. Fₚ ← (I+ΔtLₚ)Fₚ, NOT a
# prescribed F(t)). Any deviation of det Fₚ from 1 is exactly the first-order
# forward-Euler integration error DESIGN §4.1 flags — a MEASUREMENT, not a
# guess, that decides whether the exp/mid-point F-update is ever needed.
#
# This is a measurement to record (DESIGN §12 G1b), not a bare pass/fail gate;
# it also carries a sanity bound (no blow-up / NaN) and a step-halving
# convergence check confirming the drift is O(Δt) (first-order), as expected
# of the forward form.

using ParticlePlasticity
using PlasticityFEM.Materials: J2Material
using StaticArrays
using LinearAlgebra
using Test
using Printf

const _Constitutive = ParticlePlasticity.Constitutive
const _ParticlesMod = ParticlePlasticity.ParticlesMod

# A fixed, generic trace-free velocity gradient (combined shear + isochoric
# stretch + a rotational component) — NOT a pure nilpotent single-shear
# matrix (those integrate exactly with forward Euler; det(I+ΔtL)≡1 for a
# strictly-triangular L, which would trivially hide the effect).
const L_FIXED = SMatrix{3,3,Float64,9}(
     0.30, -0.20,  0.10,
     0.50, -0.30,  0.00,
     0.00,  0.40,  0.00,
)
@assert abs(tr(L_FIXED)) < 1e-14

function run_drift(nsteps::Int, dt::Float64; mat=J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0))
    pts = _ParticlesMod.Particles([SVector(0.0, 0.0, 0.0)], 1.0, 1.0)
    Jhist = zeros(nsteps + 1)
    Jhist[1] = pts.J[1]
    for s in 1:nsteps
        pts.C[1] = L_FIXED   # prescribed Lₚ = Cₚ each step (DESIGN §4)
        _Constitutive.particle_stress_update!(pts, 1, mat, dt)
        Jhist[s+1] = pts.J[1]
    end
    return Jhist
end

@testset "G1b: F-update volumetric drift (measurement)" begin
    total_time = 5.0     # elapsed "physical" time, comparable to a necking run
    for nsteps in (500, 1000, 2000, 4000, 8000)
        dt = total_time / nsteps
        Jhist = run_drift(nsteps, dt)
        @test all(isfinite, Jhist)
        drift = maximum(abs.(Jhist .- 1.0))
        @printf("  G1b  nsteps=%6d  dt=%.3e  max|J-1| = %.6e\n", nsteps, dt, drift)
        @test drift < 0.5   # sanity: bounded, no blow-up
    end

    # step-halving convergence: drift should roughly halve when dt halves
    # (first-order global error of forward Euler), reported not asserted-tight
    # (the local cubic term also contributes at this magnitude of Δt).
    d1 = maximum(abs.(run_drift(1000, total_time / 1000) .- 1.0))
    d2 = maximum(abs.(run_drift(2000, total_time / 2000) .- 1.0))
    d4 = maximum(abs.(run_drift(4000, total_time / 4000) .- 1.0))
    @printf("  G1b  convergence: drift(dt)=%.4e  drift(dt/2)=%.4e  drift(dt/4)=%.4e  ratios=%.3f, %.3f\n",
            d1, d2, d4, d1 / d2, d2 / d4)
    @test d2 < d1
    @test d4 < d2
end
