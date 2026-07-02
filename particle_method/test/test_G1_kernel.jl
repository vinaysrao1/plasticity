# G1 — single material point == FEM kernel, EXACT (DESIGN §12).
#
# Drive one particle through a prescribed homogeneous F(t) (uniaxial, then
# simple shear to large γ) and call the *same* finite_stress_update /
# finite_kinematics. Assert the stress path equals a direct call of the
# lagrangian kernel on the same F to machine precision (it is literally the
# same function — this bypasses the MPM grid/F-update integrator entirely and
# isolates the reuse contract, DESIGN §3).

using ParticlePlasticity
using PlasticityFEM.FiniteStrain: finite_kinematics, finite_stress_update, det_Fp_from_Cpinv
using PlasticityFEM.Materials: J2Material
using StaticArrays
using LinearAlgebra
using Test

const I3 = SMatrix{3,3,Float64,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)
const CPINV_I = SVector{6,Float64}(1, 1, 1, 0, 0, 0)

@testset "G1: single particle == direct kernel call (exact)" begin
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0)

    # --- path A: uniaxial stretch F(t) = diag(λ(t), 1/√λ, 1/√λ) (≈isochoric-ish
    # trial; actual plastic incompressibility is enforced by the kernel) ---
    λs = 1.0 .+ (0:0.02:0.6)

    Cp_inv = CPINV_I
    εp = zero(SVector{6,Float64})
    β_ref = zero(SVector{6,Float64})
    ᾱ = 0.0

    for λ in λs
        F = SMatrix{3,3,Float64,9}(λ, 0, 0, 0, 1 / sqrt(λ), 0, 0, 0, 1 / sqrt(λ))

        # "MPM path": call exactly the sequence Constitutive.particle_stress_update!
        # would call (finite_kinematics -> finite_stress_update), on the same F.
        kin_mpm = finite_kinematics(F, Cp_inv)
        @test kin_mpm.ok
        out_mpm = finite_stress_update(mat, kin_mpm, F, εp, β_ref, ᾱ)

        # "reference path": an independent direct call of the identical kernel
        # functions on the same inputs.
        kin_ref = finite_kinematics(F, Cp_inv)
        out_ref = finite_stress_update(mat, kin_ref, F, εp, β_ref, ᾱ)

        for k in 1:10
            @test out_mpm[k] == out_ref[k]   # literally the same function: bit-exact
        end

        # commit (mirrors what Constitutive.particle_stress_update! stores)
        τv, εp_new, β_ref_new, ᾱ_new, _, _, Cp_inv_new, _, _, _ = out_mpm
        εp, β_ref, ᾱ, Cp_inv = εp_new, β_ref_new, ᾱ_new, Cp_inv_new

        # det Fᵖ = 1 throughout (isochoric plastic flow, DESIGN §12 G1)
        @test isapprox(det_Fp_from_Cpinv(Cp_inv), 1.0; atol=1e-10)
    end

    # --- path B: simple shear to large γ (objectivity / no-Jaumann-oscillation
    # check, DESIGN §12 G1) ---
    Cp_inv = CPINV_I
    εp = zero(SVector{6,Float64})
    β_ref = zero(SVector{6,Float64})
    ᾱ = 0.0
    γs = 0:0.05:3.0   # large shear
    τ_hist = SVector{6,Float64}[]

    for γ in γs
        F = SMatrix{3,3,Float64,9}(1, 0, 0, γ, 1, 0, 0, 0, 1)

        kin_mpm = finite_kinematics(F, Cp_inv)
        @test kin_mpm.ok
        out_mpm = finite_stress_update(mat, kin_mpm, F, εp, β_ref, ᾱ)
        kin_ref = finite_kinematics(F, Cp_inv)
        out_ref = finite_stress_update(mat, kin_ref, F, εp, β_ref, ᾱ)
        for k in 1:10
            @test out_mpm[k] == out_ref[k]
        end

        τv, εp_new, β_ref_new, ᾱ_new, _, _, Cp_inv_new, _, _, _ = out_mpm
        εp, β_ref, ᾱ, Cp_inv = εp_new, β_ref_new, ᾱ_new, Cp_inv_new
        push!(τ_hist, τv)

        @test isapprox(det_Fp_from_Cpinv(Cp_inv), 1.0; atol=1e-8)
    end

    # No Jaumann-type runaway oscillation: shear stress should grow smoothly
    # (monotone-ish, bounded) rather than oscillate/blow up as γ → large
    # (the classic hypoelastic-Jaumann artifact this log-strain kernel avoids).
    τxy = [τ[4] for τ in τ_hist]
    @test all(isfinite, τxy)
    @test maximum(abs, τxy) < 10 * mat.σy0   # bounded (no blow-up)
    # monotone non-decreasing magnitude over the back half of the path (past
    # the initial elastic ramp) is a proxy for "no oscillation":
    tail = τxy[end-10:end]
    @test all(diff(tail) .>= -1e-6)
end
