# G2 — vibrating/stretched elastic bar & APIC angular-momentum conservation
# (DESIGN §12).
#
# G2a: elastic-only 1-D axial vibration of a clamped-free bar: standing-wave
# frequency vs the analytic rod formula T1 = 4L/c_bar, c_bar = sqrt(E/ρ).
# Confirms P2G/G2P, APIC low dissipation, CFL stability.
#
# G2b: a spinning free (no BCs) elastic cube — APIC's affine transfer should
# conserve linear AND angular momentum essentially exactly (Jiang et al. 2015)
# since a rigid rotation has zero rate-of-deformation everywhere, so the
# internal force stays (near) zero and the body just keeps rotating. This is
# *verified*, not assumed (DESIGN §12 G2's explicit instruction).

using ParticlePlasticity
using StaticArrays
using LinearAlgebra
using Test
using Printf

@testset "G2a: clamped-free elastic bar, axial vibration frequency" begin
    E, ν, ρ = 210e3, 0.3, 7.85e-9
    mat = J2Material(E=E, ν=ν, σy0=1e9)   # stays elastic

    L, w = 20.0, 2.0
    h = 1.0
    grid = Grid(SVector(-2.0, -2.0, -2.0), h, (28, 8, 8))
    pts = sample_box(SVector(0.0, 0.0, 0.0), SVector(L, w, w), h; ppc=2, ρ=ρ)
    np = length(pts.x)

    c_bar = sqrt(E / ρ)
    K, G = mat.K, mat.G
    c_p = sqrt((K + 4G / 3) / ρ)
    dt = 0.2 * h / c_p
    T1_analytic = 4L / c_bar

    v0 = 0.05
    for p in 1:np
        x = pts.x[p][1]
        pts.v[p] = SVector(v0 * sin(pi * x / (2L)), 0.0, 0.0)
    end

    model = MPMModel(grid, pts, mat; dt=dt, fbar=false, damping=0.0)
    fix!(model, x -> x[1] < 1e-6, :all)

    tip_mask = [pts.x[p][1] > L - h for p in 1:np]
    nsteps = 4000
    tipvel = Vector{Float64}(undef, nsteps)
    times = Vector{Float64}(undef, nsteps)
    for s in 1:nsteps
        step!(model)
        tipvel[s] = sum(model.particles.v[p][1] for p in 1:np if tip_mask[p]) / count(tip_mask)
        times[s] = model.t
    end

    idx = findfirst(i -> tipvel[i] > 0 && tipvel[i+1] <= 0, 1:nsteps-1)
    @test idx !== nothing
    t1, t2 = times[idx], times[idx+1]
    v1, v2 = tipvel[idx], tipvel[idx+1]
    tcross = t1 + (t2 - t1) * v1 / (v1 - v2)
    T1_num = 4 * tcross
    relerr = abs(T1_num - T1_analytic) / T1_analytic
    @printf("  G2a  T1 analytic=%.6e  T1 MPM=%.6e  rel err=%.4f%%\n",
            T1_analytic, T1_num, 100relerr)
    @test relerr < 0.05   # within 5% of the analytic rod frequency
end

@testset "G2b: spinning free body — APIC angular-momentum conservation" begin
    E, ν, ρ = 210e3, 0.3, 7.85e-9
    mat = J2Material(E=E, ν=ν, σy0=1e9)

    s, h, pad = 4.0, 0.5, 3.0
    n1 = Int(round((s + 2pad) / h)) + 1
    grid = Grid(SVector(-s / 2 - pad, -s / 2 - pad, -s / 2 - pad), h, (n1, n1, n1))
    pts = sample_box(SVector(-s / 2, -s / 2, -s / 2), SVector(s / 2, s / 2, s / 2), h; ppc=2, ρ=ρ)
    np = length(pts.x)

    xc = SVector(0.0, 0.0, 0.0)
    ω = SVector(0.0, 0.0, 5.0e3)
    # A rigid rotation is exactly the affine field v(x) = Ω(x-xc); initialize
    # BOTH the particle velocity and the APIC affine matrix C consistently
    # (Lₚ = Cₚ, DESIGN §4) — this is the correct IC for a known-affine field,
    # not merely a convenience: leaving C=0 at t=0 would itself inject a
    # one-step transient inconsistent with the prescribed rigid motion.
    Ω = SMatrix{3,3,Float64,9}(0, ω[3], -ω[2], -ω[3], 0, ω[1], ω[2], -ω[1], 0)
    for p in 1:np
        r = pts.x[p] - xc
        pts.v[p] = cross(ω, r)
        pts.C[p] = Ω
    end

    K, G = mat.K, mat.G
    c_p = sqrt((K + 4G / 3) / ρ)
    dt = 0.2 * h / c_p
    model = MPMModel(grid, pts, mat; dt=dt, fbar=false, damping=0.0)   # no BCs: free body

    function total_L(model)
        pts = model.particles
        L = zero(SVector{3,Float64})
        for p in eachindex(pts.x)
            L += pts.m[p] * cross(pts.x[p] - xc, pts.v[p])
        end
        return L
    end
    function total_p(model)
        pts = model.particles
        P = zero(SVector{3,Float64})
        for p in eachindex(pts.x)
            P += pts.m[p] * pts.v[p]
        end
        return P
    end

    L0 = total_L(model)
    P0 = total_p(model)

    nsteps = 2000
    for _ in 1:nsteps
        step!(model)
    end

    L1 = total_L(model)
    P1 = total_p(model)
    rel_L_drift = norm(L1 - L0) / norm(L0)
    abs_P_drift = norm(P1 - P0)
    @printf("  G2b  rel angular-momentum drift over %d steps = %.4e\n", nsteps, rel_L_drift)
    @printf("  G2b  abs linear-momentum drift  = %.4e (no external force ⇒ should be ~0)\n", abs_P_drift)

    @test rel_L_drift < 1e-4     # APIC: essentially exact (observed ~1e-7)
    @test abs_P_drift < 1e-8 * norm(pts.m) * norm(ω) * s   # ~machine-precision-scale
end
