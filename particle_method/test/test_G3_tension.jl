# G3 — homogeneous tension/compression block vs FEM, TIGHT (DESIGN §12).
#
# A 1×1×1 cube pulled uniformly (roller/symmetry BCs on the three min faces +
# prescribed x-velocity on the max face), quasi-static via the §6 machinery
# (smooth sin² displacement ramp + a settle/hold phase + light scalar
# damping). Because the deformation is homogeneous, every particle sees the
# same nominal F as the FEM Gauss point, so stress must match FEM closely once
# `KE/IE ≲ 1%`.
#
# Geometry/material mirror `lagrangian/examples/tension_cube.jl` exactly
# (E=210e3, ν=0.3, σy0=250, Hiso=1000 MPa; ε_target=0.01 nominal strain), and
# the FEM reference is obtained by actually running `PlasticityFEM` (the
# reuse-contract dependency) on the identical problem — not just the closed
# form — so this is a genuine cross-solver check.
#
# ROOT-CAUSE NOTE (not papered over — see the final report for the full
# story). The first working version of this test (ppc=2, 512 particles) gave
# only ~2e-3 relative error, not quite DESIGN's "~1e-3". This was diagnosed,
# not assumed: (1) `Constitutive`/`Transfer` were verified bug-free via an
# independent APIC-exactness unit test (affine velocity field reproduced to
# machine precision, see G2); (2) the residual error is a *static* spatial
# pattern (per-particle stress standard deviation ~100 MPa on a ~257 MPa mean,
# concentrated at the cube's free-surface corners — a known consequence of
# quadratic-B-spline MPM's partial stencils at a domain boundary/corner on a
# *single-cube*, coarse-cell mesh, with no GIMP/CPDI correction — DESIGN §1's
# stated non-goal); (3) time-averaging the last 30% of the hold phase does NOT
# change the mean (ruling out residual high-frequency ringing — DESIGN §6's
# suggested first suspect); (4) the mean DOES converge slowly toward FEM as
# the settle duration (in units of the body's own P-wave crossing time,
# mesh-independent) grows — 40 periods: 0.7%, 80: 0.29%, 160: 0.20%; (5)
# increasing particles-per-cell (ppc=2→3, more independent samples of the same
# noisy boundary field, averaging it down) closed the rest of the gap: at
# ppc=3, 160 periods, relative error is 4.8e-4 — inside the ~1e-3 target.
# ppc=3 (1728 particles) is used below.

using ParticlePlasticity
using StaticArrays
using LinearAlgebra
using Test
using Printf
using Statistics

import PlasticityFEM
const FEM = PlasticityFEM

@testset "G3: homogeneous tension block vs FEM (tight)" begin
    E, ν, σy0, Hiso = 210e3, 0.3, 250.0, 1000.0
    ε_target = 0.01

    # --- FEM reference (identical to tension_cube.jl) ---
    mesh = FEM.box_mesh(1.0, 1.0, 1.0, 1, 1, 1)
    steel = FEM.J2Material(E=E, ν=ν, σy0=σy0, Hiso=Hiso)
    femmodel = FEM.Model(mesh, steel)
    FEM.fix!(femmodel, FEM.on_face(mesh, :xmin), :x)
    FEM.fix!(femmodel, FEM.on_face(mesh, :ymin), :y)
    FEM.fix!(femmodel, FEM.on_face(mesh, :zmin), :z)
    FEM.prescribe!(femmodel, FEM.on_face(mesh, :xmax), :x, ε_target)
    fres = FEM.solve!(femmodel; nsteps=20, tol=1e-8, maxiter=25)
    @test fres.converged
    σxx_fem = FEM.gauss_stress(femmodel)[1, 1]

    # --- MPM ---
    mat = J2Material(E=E, ν=ν, σy0=σy0, Hiso=Hiso)
    ρ = 7.85e-9          # steel, mm-N-tonne-s consistent units; mass_scale = 1 (none)
    h = 0.25
    pad = 2h
    n1 = Int(round((1 + 2pad) / h)) + 1
    grid = Grid(SVector(-pad, -pad, -pad), h, (n1, n1, n1))
    pts = sample_box(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0), h; ppc=3, ρ=ρ)
    np = length(pts.x)

    K, G = mat.K, mat.G
    c_p = sqrt((K + 4G / 3) / ρ)                # elastic P-wave speed (§4.3)
    dt = 0.2 * h / c_p
    Tnat = 1.0 / c_p                            # mesh-independent crossing time (L=1)
    N_periods = 160
    T_ramp = N_periods * Tnat
    T_hold = N_periods * Tnat

    # smooth sin² ease-in-ease-out DISPLACEMENT ramp (zero velocity at both
    # ends of the ramp, DESIGN §6 "slow, smooth loading"), holding at the
    # target displacement (zero velocity) for the settle phase.
    vfun = t -> t <= T_ramp ? ε_target * (pi / (2T_ramp)) * sin(pi * t / T_ramp) : 0.0

    model = MPMModel(grid, pts, mat; dt=dt, fbar=false, damping=0.02, mass_scale=1.0)
    fix!(model, x -> x[1] < 1e-9, :x)
    fix!(model, x -> x[2] < 1e-9, :y)
    fix!(model, x -> x[3] < 1e-9, :z)
    prescribe!(model, x -> x[1] > 1.0 - 1e-9, :x, vfun)

    nsteps = Int(round((T_ramp + T_hold) / dt))
    navg = max(1, Int(round(0.3 * nsteps)))     # average the last 30% of the hold phase
    accum = zeros(np)
    nacc = 0
    for s in 1:nsteps
        step!(model)
        if s > nsteps - navg
            for p in 1:np
                accum[p] += particle_cauchy(model, p)[1]
            end
            nacc += 1
        end
    end
    σxx_particles = accum ./ nacc
    σxx_mpm = mean(σxx_particles)

    KE = kinetic_energy(model)
    IE = model.IE
    ke_ie = KE / IE
    relerr = abs(σxx_mpm - σxx_fem) / abs(σxx_fem)

    @printf("  G3  σxx FEM = %.4f MPa   σxx MPM = %.4f MPa   rel. err = %.4e\n",
            σxx_fem, σxx_mpm, relerr)
    @printf("  G3  KE/IE = %.3e (quasi-static gate: ≲ 1%%)\n", ke_ie)

    @test ke_ie < 0.01          # DESIGN §6 quasi-static gate
    @test relerr < 1.5e-3       # ~1e-3 target (observed ~4.8e-4, see root-cause note)
end
