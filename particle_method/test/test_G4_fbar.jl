# G4 — F̄ anti-locking gate (DESIGN §12).
#
# Fully-plastic compression of the same 1×1×1 block as G3 (5% nominal
# compressive strain, well past yield ε_y≈0.12%, so plastic — isochoric — flow
# dominates): with `fbar=false` expect checkerboard pressure (large
# particle-to-particle pressure variance); with `fbar=true` the pressure field
# is smooth (variance collapses) and the mean pressure matches FEM's F-bar
# element (`element=:finite_fbar`) result. Must pass before any necking result
# (G6) is trusted (DESIGN §15).

using ParticlePlasticity
using StaticArrays
using LinearAlgebra
using Test
using Printf
using Statistics

import PlasticityFEM
const FEM = PlasticityFEM

@testset "G4: F-bar anti-locking (fully-plastic compression)" begin
    E, ν, σy0, Hiso = 210e3, 0.3, 250.0, 1000.0
    ε_target = -0.05    # 5% compression, fully plastic

    # --- FEM reference (F-bar element, finer mesh so it's a genuine multi-
    # element homogeneous-field check, not a single-element triviality) ---
    mesh = FEM.box_mesh(1.0, 1.0, 1.0, 4, 4, 4)
    steel = FEM.J2Material(E=E, ν=ν, σy0=σy0, Hiso=Hiso)
    femmodel = FEM.Model(mesh, steel; element=:finite_fbar)
    FEM.fix!(femmodel, FEM.on_face(mesh, :xmin), :x)
    FEM.fix!(femmodel, FEM.on_face(mesh, :ymin), :y)
    FEM.fix!(femmodel, FEM.on_face(mesh, :zmin), :z)
    FEM.prescribe!(femmodel, FEM.on_face(mesh, :xmax), :x, ε_target)
    fres = FEM.solve!(femmodel; nsteps=40, tol=1e-7, maxiter=40)
    @test fres.converged
    σfem = FEM.gauss_stress(femmodel)
    pfem = [(σfem[1, g] + σfem[2, g] + σfem[3, g]) / 3 for g in 1:size(σfem, 2)]
    p_fem_mean = mean(pfem)

    function run_mpm(; fbar::Bool)
        mat = J2Material(E=E, ν=ν, σy0=σy0, Hiso=Hiso)
        ρ = 7.85e-9
        h = 0.25
        pad = 2h
        n1 = Int(round((1 + 2pad) / h)) + 1
        grid = Grid(SVector(-pad, -pad, -pad), h, (n1, n1, n1))
        pts = sample_box(SVector(0.0, 0.0, 0.0), SVector(1.0, 1.0, 1.0), h; ppc=2, ρ=ρ)
        np = length(pts.x)
        K, G = mat.K, mat.G
        c_p = sqrt((K + 4G / 3) / ρ)
        dt = 0.2 * h / c_p
        Tnat = 1.0 / c_p
        N_periods = 160
        T_ramp = N_periods * Tnat
        T_hold = N_periods * Tnat
        vfun = t -> t <= T_ramp ? ε_target * (pi / (2T_ramp)) * sin(pi * t / T_ramp) : 0.0
        model = MPMModel(grid, pts, mat; dt=dt, fbar=fbar, damping=0.02, mass_scale=1.0)
        fix!(model, x -> x[1] < 1e-9, :x)
        fix!(model, x -> x[2] < 1e-9, :y)
        fix!(model, x -> x[3] < 1e-9, :z)
        prescribe!(model, x -> x[1] > 1.0 - 1e-9, :x, vfun)
        nsteps = Int(round((T_ramp + T_hold) / dt))
        for _ in 1:nsteps
            step!(model)
        end
        press = [sum(particle_cauchy(model, p)[1:3]) / 3 for p in 1:np]
        return (mean=mean(press), std=std(press), KEIE=kinetic_energy(model) / model.IE)
    end

    r_false = run_mpm(fbar=false)
    r_true = run_mpm(fbar=true)

    @printf("  G4  FEM (fbar element)   mean pressure = %.3f MPa\n", p_fem_mean)
    @printf("  G4  MPM fbar=false       mean = %.3f  std = %.3f  (CoV=%.1f%%)  KE/IE=%.2e\n",
            r_false.mean, r_false.std, 100r_false.std / abs(r_false.mean), r_false.KEIE)
    @printf("  G4  MPM fbar=true        mean = %.3f  std = %.3f  (CoV=%.1f%%)  KE/IE=%.2e\n",
            r_true.mean, r_true.std, 100r_true.std / abs(r_true.mean), r_true.KEIE)
    @printf("  G4  std reduction fbar=true vs fbar=false: %.2fx\n", r_false.std / r_true.std)

    @test r_false.KEIE < 0.01
    @test r_true.KEIE < 0.01

    # checkerboard: fbar=false has a large pressure variance relative to fbar=true
    @test r_false.std > 3 * r_true.std      # observed ~7x

    # fbar=true's smooth pressure matches FEM's F-bar mean pressure
    relerr_true = abs(r_true.mean - p_fem_mean) / abs(p_fem_mean)
    @printf("  G4  fbar=true rel. err vs FEM = %.4e\n", relerr_true)
    @test relerr_true < 0.05
end
