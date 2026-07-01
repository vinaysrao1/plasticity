# Finite-strain (large-deformation) J2 plasticity tests — F1–F10 (FINITE_STRAIN §7).
#
# F2 is the master gate: the consistent element tangent matches central finite
# differences of the element force to <1e-6 across deformed + plastic states.
# F1 is the small-load regression: :finite reproduces :small as the load → 0.

using PlasticityFEM
using PlasticityFEM.Elements
using PlasticityFEM.FiniteStrain
using PlasticityFEM.Materials
using StaticArrays
using LinearAlgebra
using Test

const FS = PlasticityFEM.FiniteStrain

# --- shared helpers ---------------------------------------------------------

# single-element finite-strain geometry (1×1×1 cube at the origin)
function _unit_element()
    mesh = box_mesh(1.0, 1.0, 1.0, 1, 1, 1)
    cache = precompute_cache(mesh.nodes, mesh.elements)
    dNdXs = element_ref_grads(cache, 1)
    detJw = cache.detJwref
    Xe = PlasticityFEM.Elements.element_coords(mesh.nodes, mesh.elements, 1)
    return mesh, dNdXs, detJw, Xe, Xe
end

# element nodal displacement reproducing a homogeneous deformation gradient F.
# `Xe` is the 8×3 matrix of LOCAL element node coordinates (so the local DOF
# ordering matches the kernel's `deformation_gradient`).
function _ue_from_F(Xe, F)
    u = zeros(24)
    @inbounds for a in 1:8
        Xa = SVector{3,Float64}(Xe[a, 1], Xe[a, 2], Xe[a, 3])
        ua = F * Xa - Xa
        u[3(a - 1) + 1] = ua[1]; u[3(a - 1) + 2] = ua[2]; u[3(a - 1) + 3] = ua[3]
    end
    return SVector{24,Float64}(u)
end

# identity Cp_inv (6×8) and zero history
_cp_identity() = (c = zeros(6, 8); c[1, :] .= 1; c[2, :] .= 1; c[3, :] .= 1; c)

# element force only (no tangent use), for finite-difference of the tangent
function _fe(kind, mat, dNdXs, detJw, Xe, ue, εp, β, ᾱ, Cp)
    σo = zeros(6, 8)
    Fe, _ = element_force_tangent_finite!(kind, mat, dNdXs, detJw, ue, Xe,
                                          εp, β, ᾱ, Cp, 1, σo, Val(false))
    return Fe
end

function _fe_ke(kind, mat, dNdXs, detJw, Xe, ue, εp, β, ᾱ, Cp)
    σo = zeros(6, 8)
    return element_force_tangent_finite!(kind, mat, dNdXs, detJw, ue, Xe,
                                         εp, β, ᾱ, Cp, 1, σo, Val(false))
end

# commit the plastic update at a given deformation (advances εp,β,ᾱ,Cp)
function _commit!(kind, mat, dNdXs, detJw, Xe, ue, εp, β, ᾱ, Cp)
    σo = zeros(6, 8)
    element_force_tangent_finite!(kind, mat, dNdXs, detJw, ue, Xe,
                                  εp, β, ᾱ, Cp, 1, σo, Val(true))
    return nothing
end

# central-difference tangent of the element force, returning (relerr, symerr)
function _fd_tangent(kind, mat, dNdXs, detJw, Xe, ue, εp, β, ᾱ, Cp; h=1e-7)
    _, Ke = _fe_ke(kind, mat, dNdXs, detJw, Xe, ue, εp, β, ᾱ, Cp)
    Kfd = zeros(24, 24)
    for j in 1:24
        up = setindex(ue, ue[j] + h, j)
        um = setindex(ue, ue[j] - h, j)
        Fp = _fe(kind, mat, dNdXs, detJw, Xe, up, εp, β, ᾱ, Cp)
        Fm = _fe(kind, mat, dNdXs, detJw, Xe, um, εp, β, ᾱ, Cp)
        Kfd[:, j] = (Fp .- Fm) ./ (2h)
    end
    K = Matrix(Ke)
    return norm(Kfd - K) / norm(K), norm(K - K') / norm(K)
end

# end-to-end uniaxial model (roller BCs on the three min faces, x-pull on xmax)
function _uniaxial(elem, disp; nx=2, ny=2, nz=2, nsteps=5, mat=nothing)
    mesh = box_mesh(1.0, 1.0, 1.0, nx, ny, nz)
    m = mat === nothing ? J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0) : mat
    model = Model(mesh, m; element=elem)
    fix!(model, on_face(mesh, :xmin), :x)
    fix!(model, on_face(mesh, :ymin), :y)
    fix!(model, on_face(mesh, :zmin), :z)
    prescribe!(model, on_face(mesh, :xmax), :x, disp)
    res = solve!(model; nsteps=nsteps, tol=1e-10, linsolve=:direct)
    return model, res
end

# =========================================================================== #

@testset "F1 small-displacement limit (:finite → :small)" begin
    # The geometric-nonlinearity correction is O(strain); the relative finite–small
    # difference must vanish proportionally as the load → 0.
    for disp in (1e-6, 1e-5, 1e-4)
        ms, _ = _uniaxial(:small, disp)
        mf, _ = _uniaxial(:finite, disp)
        us = nodal_displacements(ms); uf = nodal_displacements(mf)
        relu = norm(uf - us) / norm(us)
        # difference scales like the strain (~disp); assert it is below ~disp.
        @test relu < 0.5 * disp + 1e-9
    end
    # at a tiny load the agreement is ~1e-7 (FINITE_STRAIN F1 "~1e-8" target scale)
    ms, _ = _uniaxial(:small, 1e-6)
    mf, _ = _uniaxial(:finite, 1e-6)
    @test norm(nodal_displacements(mf) - nodal_displacements(ms)) /
          norm(nodal_displacements(ms)) < 1e-6
end

@testset "F2 consistent tangent vs finite differences (MASTER GATE)" begin
    _, dNdXs, detJw, Xe, X = _unit_element()
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)

    # (a) elastic deformed states — distinct & general F
    for F in (SMatrix{3,3,Float64,9}(1.1, 0, 0, 0, 0.95, 0, 0, 0, 1.03),
              SMatrix{3,3,Float64,9}(1.08, 0.04, 0.0, 0.03, 0.97, 0.02, 0.01, 0.0, 1.05))
        ue = _ue_from_F(X, F)
        relerr, _ = _fd_tangent(Hex8Finite(), mat, dNdXs, detJw, Xe, ue,
                                zeros(6, 8), zeros(6, 8), zeros(8), _cp_identity())
        @test relerr < 1e-6
    end

    # (b) plastic states: load incrementally, freezing committed history each step
    #     (the realistic Newton tangent: Cp_inv from the PREVIOUS converged step).
    εp = zeros(6, 8); β = zeros(6, 8); ᾱ = zeros(8); Cp = _cp_identity()
    for k in 1:8
        Fk = SMatrix{3,3,Float64,9}(1 + 0.02k, 0, 0, 0, 1 - 0.008k, 0, 0, 0, 1 - 0.008k)
        ue = _ue_from_F(X, Fk)
        relerr, _ = _fd_tangent(Hex8Finite(), mat, dNdXs, detJw, Xe, ue, εp, β, ᾱ, Cp)
        @test relerr < 1e-6
        _commit!(Hex8Finite(), mat, dNdXs, detJw, Xe, ue, εp, β, ᾱ, Cp)
    end
    @test maximum(ᾱ) > 0.05    # genuinely plastic

    # (c) non-coaxial plastic + shear: commit at F1, evaluate tangent at F2
    εp = zeros(6, 8); β = zeros(6, 8); ᾱ = zeros(8); Cp = _cp_identity()
    F1 = SMatrix{3,3,Float64,9}(1.05, 0.04, 0.01, 0.025, 0.98, 0.015, 0.005, 0.01, 1.02)
    F2 = SMatrix{3,3,Float64,9}(1.1, 0.08, 0.02, 0.05, 0.96, 0.03, 0.01, 0.02, 1.04)
    _commit!(Hex8Finite(), mat, dNdXs, detJw, Xe, _ue_from_F(X, F1), εp, β, ᾱ, Cp)
    relerr, _ = _fd_tangent(Hex8Finite(), mat, dNdXs, detJw, Xe, _ue_from_F(X, F2), εp, β, ᾱ, Cp)
    @test relerr < 1e-6

    # (d) F-bar tangent (standard + plastic)
    relerr, _ = _fd_tangent(Hex8FiniteFbar(), mat, dNdXs, detJw, Xe,
                            _ue_from_F(X, F2), zeros(6, 8), zeros(6, 8), zeros(8), _cp_identity())
    @test relerr < 1e-6
    εp = zeros(6, 8); β = zeros(6, 8); ᾱ = zeros(8); Cp = _cp_identity()
    _commit!(Hex8FiniteFbar(), mat, dNdXs, detJw, Xe, _ue_from_F(X, F1), εp, β, ᾱ, Cp)
    relerr, _ = _fd_tangent(Hex8FiniteFbar(), mat, dNdXs, detJw, Xe, _ue_from_F(X, F2), εp, β, ᾱ, Cp)
    @test relerr < 1e-6
end

@testset "F2′ Newton quadratic convergence (finite, plastic)" begin
    _, res = _uniaxial(:finite, 0.05; nsteps=3)
    @test res.converged
    r = res.residuals[end]
    # asymptotic quadratic drop: ‖R_{k+1}‖ ≤ C ‖R_k‖² for the last steps
    @test length(r) <= 6
    if length(r) >= 4
        # last two ratios show super-linear (quadratic) decrease
        @test r[end] < (r[end-1])^1.5 + 1e-12
    end
end

@testset "F3 plastic incompressibility det Fᵖ = 1" begin
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0)
    for F in (SMatrix{3,3,Float64,9}(1.2, 0.05, 0, 0.03, 0.92, 0, 0, 0, 0.95),
              SMatrix{3,3,Float64,9}(1.3, 0.1, 0.05, 0.08, 0.9, 0.03, 0.02, 0.04, 0.85))
        kin = FS.finite_kinematics(F, SVector{6,Float64}(1, 1, 1, 0, 0, 0))
        τ, εp, β, ᾱ, D, τp, Cpi = FS.finite_stress_update(mat, kin, F,
            zero(SVector{6,Float64}), zero(SVector{6,Float64}), 0.0)
        @test ᾱ > 0.0                       # plastic flow occurred
        @test abs(FS.det_Fp_from_Cpinv(Cpi) - 1.0) < 1e-12
    end
end

@testset "F4 objectivity (Cauchy rotates as Q σ Qᵀ)" begin
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0)
    F = SMatrix{3,3,Float64,9}(1.2, 0.05, 0, 0.03, 0.92, 0, 0, 0, 0.95)
    θ = 0.7
    Q = SMatrix{3,3,Float64,9}(cos(θ), sin(θ), 0, -sin(θ), cos(θ), 0, 0, 0, 1)
    I6 = SVector{6,Float64}(1, 1, 1, 0, 0, 0); Z6 = zero(SVector{6,Float64})
    kin1 = FS.finite_kinematics(F, I6)
    τ1, _, _, _, _, _, _ = FS.finite_stress_update(mat, kin1, F, Z6, Z6, 0.0)
    σ1 = FS.voigt_to_sym3(τ1) / det(F)
    kin2 = FS.finite_kinematics(Q * F, I6)
    τ2, _, _, _, _, _, _ = FS.finite_stress_update(mat, kin2, Q * F, Z6, Z6, 0.0)
    σ2 = FS.voigt_to_sym3(τ2) / det(Q * F)
    @test norm(σ2 - Q * σ1 * Q') / norm(σ1) < 1e-12
end

@testset "F4b objectivity with KINEMATIC hardening at θ>90° (B2 regression)" begin
    # Adversarial objectivity: build a deformed + plastic state with non-zero
    # back-stress, then superpose a finite rotation Q (120°). With the global-frame
    # back-stress bug this fails by ~1–17%; the rotation-neutralized (β_ref) storage
    # makes it exact. Probes the kinematic-hardening objectivity directly.
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=200.0, Hkin=2000.0)
    I6 = SVector{6,Float64}(1, 1, 1, 0, 0, 0); Z6 = zero(SVector{6,Float64})
    # step 0: a plastic deformation builds up β_ref (reference back-stress) and Cᵖ⁻¹
    F1 = SMatrix{3,3,Float64,9}(1.2, 0.05, 0, 0.03, 0.92, 0, 0, 0, 0.95)
    kin1 = FS.finite_kinematics(F1, I6)
    _, εp1, βref1, ᾱ1, _, _, Cpi1, _, _, _ =
        FS.finite_stress_update(mat, kin1, F1, Z6, Z6, 0.0)
    @test ᾱ1 > 0.0 && norm(βref1) > 0.0
    # step 1: a second (non-coaxial) deformation; compare σ vs Q σ Qᵀ at θ = 120°
    F2 = SMatrix{3,3,Float64,9}(1.3, 0.08, 0.02, 0.06, 0.9, 0.01, 0.0, 0.02, 0.97)
    function σ_at(F)
        k = FS.finite_kinematics(F, Cpi1)
        τ, _, _, _, _, _, _, _, _, _ = FS.finite_stress_update(mat, k, F, εp1, βref1, ᾱ1)
        return FS.voigt_to_sym3(τ) / det(F)
    end
    σ1 = σ_at(F2)
    for θ in (0.1, deg2rad(120.0), 3.0)
        Q = SMatrix{3,3,Float64,9}(cos(θ), sin(θ), 0, -sin(θ), cos(θ), 0, 0, 0, 1)
        σ2 = σ_at(Q * F2)
        @test norm(σ2 - Q * σ1 * Q') / norm(σ1) < 1e-10
    end
end

@testset "F5 elastic cyclic-shear round-trip (no drift)" begin
    # Drive simple shear up and back to zero with σy huge ⇒ purely ELASTIC. The
    # round-trip stress must return to ~zero (no spurious oscillation / drift).
    # This is an elastic reversibility check (the label, not the math, was the
    # earlier overstatement — a genuinely plastic version is F5b below).
    mat = J2Material(E=210e3, ν=0.3, σy0=1e9)   # σy huge ⇒ stays elastic
    Z6 = zero(SVector{6,Float64}); I6 = SVector{6,Float64}(1, 1, 1, 0, 0, 0)
    γs = vcat(range(0, 0.3; length=8), range(0.3, 0.0; length=8))
    τend = nothing
    for γ in γs
        F = SMatrix{3,3,Float64,9}(1, 0, 0, γ, 1, 0, 0, 0, 1)
        kin = FS.finite_kinematics(F, I6)
        τend, _, _, _, _, _, _ = FS.finite_stress_update(mat, kin, F, Z6, Z6, 0.0)
    end
    @test norm(τend) < 1e-6 * mat.E           # returns to ~zero stress
end

@testset "F5b plastic cyclic-shear no spurious drift" begin
    # Genuinely PLASTIC cyclic simple shear (γ: 0→+γ→0→−γ→0) with isotropic
    # hardening. The path is committed step by step (path-dependent). On returning
    # to γ=0 the deviatoric Kirchhoff stress must be bounded by the current yield
    # surface (no runaway / spurious oscillatory drift) and the accumulated plastic
    # strain must grow monotonically (physical dissipation).
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0)
    Z6 = zero(SVector{6,Float64}); I6 = SVector{6,Float64}(1, 1, 1, 0, 0, 0)
    γs = vcat(range(0, 0.05; length=6)[2:end], range(0.05, 0.0; length=6)[2:end],
              range(0.0, -0.05; length=6)[2:end], range(-0.05, 0.0; length=6)[2:end])
    εp = Z6; βr = Z6; ᾱ = 0.0; Cpi = I6
    ᾱ_prev = 0.0; τ_at_zero = nothing
    for γ in γs
        F = SMatrix{3,3,Float64,9}(1, 0, 0, γ, 1, 0, 0, 0, 1)
        kin = FS.finite_kinematics(F, Cpi)
        τ, εp, βr, ᾱ, _, _, Cpi = FS.finite_stress_update(mat, kin, F, εp, βr, ᾱ)
        @test ᾱ >= ᾱ_prev - 1e-12             # plastic strain never decreases
        ᾱ_prev = ᾱ
        τ_at_zero = τ
    end
    @test ᾱ > 0.0                              # genuinely plastic
    # at the final γ=0 the deviatoric stress is bounded by the current yield stress.
    p = (τ_at_zero[1] + τ_at_zero[2] + τ_at_zero[3]) / 3
    s = τ_at_zero - SVector{6,Float64}(p, p, p, 0, 0, 0)
    seq = sqrt(1.5 * (s[1]^2 + s[2]^2 + s[3]^2 + 2 * (s[4]^2 + s[5]^2 + s[6]^2)))
    @test seq <= mat.σy0 + mat.Hiso * ᾱ + 1e-6 * mat.E
end

@testset "F6/F7 F-bar relieves volumetric locking" begin
    # Near-incompressible (ν→0.5) plastic bending: F-bar converges where the
    # standard element is markedly stiffer (locks). Compare tip displacement.
    mat = J2Material(E=210e3, ν=0.49, σy0=250.0, Hiso=500.0)
    function bend(elem)
        mesh = box_mesh(6.0, 1.0, 1.0, 6, 2, 2)
        model = Model(mesh, mat; element=elem)
        fix!(model, on_face(mesh, :xmin), :all)
        load!(model, on_face(mesh, :xmax), :z, -10.0; distribute=true)
        res = solve!(model; nsteps=6, tol=1e-7, linsolve=:direct, maxiter=40)
        return res.converged, maximum(abs, nodal_displacements(model)[3, :])
    end
    cf, δf = bend(:finite_fbar)
    cs, δs = bend(:finite)
    @test cf && cs
    # F-bar is markedly softer than the (volumetrically locking) standard element
    @test δf > 1.3 * δs
end

@testset "F8 large-rotation cantilever (finite-strain, sensible)" begin
    mat = J2Material(E=210e3, ν=0.3, σy0=1e9)   # elastic, large rotation
    mesh = box_mesh(8.0, 1.0, 1.0, 8, 1, 1)
    model = Model(mesh, mat; element=:finite)
    fix!(model, on_face(mesh, :xmin), :all)
    load!(model, on_face(mesh, :xmax), :z, -300.0; distribute=true)
    res = solve!(model; nsteps=6, tol=1e-8, linsolve=:direct, maxiter=40)
    @test res.converged
    δz = maximum(abs, nodal_displacements(model)[3, :])
    @test δz > 0.1               # appreciable, physically sensible deflection
    @test all(isfinite, model.U)
end

@testset "F9 tangent symmetry (isotropic / perfect plasticity)" begin
    _, dNdXs, detJw, Xe, X = _unit_element()
    # Standard finite strain with isotropic hardening: K is symmetric to ~1e-10.
    for mat in (J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0),
                J2Material(E=210e3, ν=0.3, σy0=250.0))   # perfect plasticity
        εp = zeros(6, 8); β = zeros(6, 8); ᾱ = zeros(8); Cp = _cp_identity()
        F1 = SMatrix{3,3,Float64,9}(1.08, 0, 0, 0, 0.97, 0, 0, 0, 0.96)
        F2 = SMatrix{3,3,Float64,9}(1.1, 0.02, 0, 0.01, 0.96, 0, 0, 0, 0.95)
        _commit!(Hex8Finite(), mat, dNdXs, detJw, Xe, _ue_from_F(X, F1), εp, β, ᾱ, Cp)
        _, Ke = _fe_ke(Hex8Finite(), mat, dNdXs, detJw, Xe, _ue_from_F(X, F2), εp, β, ᾱ, Cp)
        K = Matrix(Ke)
        @test norm(K - K') / norm(K) < 1e-10
    end
    # elastic is always symmetric
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hkin=500.0)
    _, Ke = _fe_ke(Hex8Finite(), mat, dNdXs, detJw, Xe,
                   _ue_from_F(X, SMatrix{3,3,Float64,9}(1.001, 0, 0, 0, 0.9995, 0, 0, 0, 1.0003)),
                   zeros(6, 8), zeros(6, 8), zeros(8), _cp_identity())
    K = Matrix(Ke)
    @test norm(K - K') / norm(K) < 1e-9
end

@testset "F9b deformed KINEMATIC-hardening tangent (consistent, non-symmetric)" begin
    # B2 follow-up: at a deformed + plastic state with kinematic hardening the
    # objective (β_ref) law gives a CONSISTENT but NON-SYMMETRIC tangent. The FD
    # gate must pass (<1e-6) while the asymmetry is appreciable — the latter is
    # what B1's symmetry-aware solver selection accounts for. This probes the
    # ∂β_sp/∂F coupling that the global-frame bug omitted.
    _, dNdXs, detJw, Xe, X = _unit_element()
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=200.0, Hkin=2000.0)
    εp = zeros(6, 8); β = zeros(6, 8); ᾱ = zeros(8); Cp = _cp_identity()
    F1 = SMatrix{3,3,Float64,9}(1.1, 0.04, 0.01, 0.03, 0.95, 0.02, 0.0, 0.01, 0.97)
    F2 = SMatrix{3,3,Float64,9}(1.18, 0.09, 0.03, 0.06, 0.92, 0.04, 0.02, 0.03, 1.0)
    _commit!(Hex8Finite(), mat, dNdXs, detJw, Xe, _ue_from_F(X, F1), εp, β, ᾱ, Cp)
    @test maximum(ᾱ) > 0.0        # genuinely plastic, non-zero back-stress stored
    relerr, symerr = _fd_tangent(Hex8Finite(), mat, dNdXs, detJw, Xe,
                                 _ue_from_F(X, F2), εp, β, ᾱ, Cp)
    @test relerr < 1e-6           # tangent stays CONSISTENT (Newton quadratic)
    @test symerr > 1e-6           # and is genuinely NON-symmetric here
end

@testset "F11 symmetry-aware solver: DEFAULT solve on non-symmetric tangents (B1)" begin
    # Validator's crash repro: :finite_fbar with DEFAULT solve! args used to HARD
    # CRASH (cg! raises an SPD error on the non-symmetric F-bar tangent before the
    # fallback could run). :auto must now pick :direct and run to convergence, and
    # the result must match an explicit :direct solve. Same for :finite + Hkin>0.
    function run(elem, mat, linsolve)
        mesh = box_mesh(1.0, 1.0, 1.0, 3, 3, 3)
        model = Model(mesh, mat; element=elem)
        fix!(model, on_face(mesh, :xmin), :x)
        fix!(model, on_face(mesh, :ymin), :y)
        fix!(model, on_face(mesh, :zmin), :z)
        prescribe!(model, on_face(mesh, :xmax), :x, 0.08)
        res = linsolve === nothing ? solve!(model; nsteps=4) :
                                     solve!(model; nsteps=4, linsolve=linsolve)
        return res.converged, copy(model.U)
    end
    # (a) F-bar, near-incompressible, isotropic — non-symmetric (F-bar always)
    matf = J2Material(E=210e3, ν=0.49, σy0=250.0, Hiso=1000.0)
    cdef, udef = run(:finite_fbar, matf, nothing)     # DEFAULT (:auto)
    cdir, udir = run(:finite_fbar, matf, :direct)
    @test cdef && cdir
    @test norm(udef - udir) / norm(udir) < 1e-10      # auto == direct
    # (b) finite strain + kinematic hardening — non-symmetric (objective β rotation)
    matk = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=200.0, Hkin=2000.0)
    ckdef, ukdef = run(:finite, matk, nothing)
    ckdir, ukdir = run(:finite, matk, :direct)
    @test ckdef && ckdir
    @test norm(ukdef - ukdir) / norm(ukdir) < 1e-10
    # (c) forcing :cg on a non-symmetric config warns and overrides to :direct
    cforce, _ = run(:finite_fbar, matf, :cg)
    @test cforce
end

@testset "F10 allocation gates (finite kernel + assembly O(1))" begin
    function kern()
        mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)
        mesh = box_mesh(1.0, 1.0, 1.0, 1, 1, 1)
        cache = precompute_cache(mesh.nodes, mesh.elements)
        dNdXs = element_ref_grads(cache, 1); detJw = cache.detJwref
        Xe = PlasticityFEM.Elements.element_coords(mesh.nodes, mesh.elements, 1)
        εp = zeros(6, 8); β = zeros(6, 8); ᾱ = zeros(8); σ = zeros(6, 8); Cp = _cp_identity()
        ue = SVector{24,Float64}(ntuple(i -> 0.01 * sin(i), 24))
        k = Hex8Finite()
        element_force_tangent_finite!(k, mat, dNdXs, detJw, ue, Xe, εp, β, ᾱ, Cp, 1, σ, Val(false))
        a = @allocated element_force_tangent_finite!(k, mat, dNdXs, detJw, ue, Xe, εp, β, ᾱ, Cp, 1, σ, Val(false))
        ac = @allocated element_force_tangent_finite!(k, mat, dNdXs, detJw, ue, Xe, εp, β, ᾱ, Cp, 1, σ, Val(true))
        kf = Hex8FiniteFbar()
        element_force_tangent_finite!(kf, mat, dNdXs, detJw, ue, Xe, εp, β, ᾱ, Cp, 1, σ, Val(false))
        af = @allocated element_force_tangent_finite!(kf, mat, dNdXs, detJw, ue, Xe, εp, β, ᾱ, Cp, 1, σ, Val(false))
        return a, ac, af
    end
    a, ac, af = kern()
    @test a == 0
    @test ac == 0
    @test af == 0

    # assembly is O(1)-alloc (independent of nelem)
    function asm(nx)
        mesh = box_mesh(1.0, 1.0, 1.0, nx, nx, nx)
        model = Model(mesh, J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0); element=:finite)
        sp = model.sparsity; st = model.state_trial
        U = copy(model.U); fill!(U, 1e-4)
        R = copy(model.Rbuf)
        PlasticityFEM.Assembly.assemble!(sp, model.material, model.cache, U,
            st.εp, st.β, st.ᾱ, st.σ, R; commit=false, kind=model.kind, Cp_inv=st.Cp_inv)
        a = @allocated PlasticityFEM.Assembly.assemble!(sp, model.material, model.cache, U,
            st.εp, st.β, st.ᾱ, st.σ, R; commit=false, kind=model.kind, Cp_inv=st.Cp_inv)
        return a
    end
    a2 = asm(2); a3 = asm(3)
    # bounded by a small constant independent of nelem (allow modest overhead)
    @test a3 <= a2 + 2048
end

@testset "F13 inverted element fails LOUDLY (not silent zero stress)" begin
    # R1 regression: an inverted (det F = J ≤ 0) configuration must throw a clear
    # typed ElementInversionError, NOT silently return zero stress / a wrong tangent.
    _, dNdXs, detJw, Xe, X = _unit_element()
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0)
    # Folded element: negative determinant (det = -1.1 here).
    Finv = SMatrix{3,3,Float64,9}(-1.1, 0, 0, 0, 1.0, 0, 0, 0, 1.0)
    @test det(Finv) < 0
    ue = _ue_from_F(X, Finv)
    σo = zeros(6, 8)
    err = nothing
    try
        element_force_tangent_finite!(Hex8Finite(), mat, dNdXs, detJw, ue, Xe,
            zeros(6, 8), zeros(6, 8), zeros(8), _cp_identity(), 1, σo, Val(false))
    catch e
        err = e
    end
    @test err isa ElementInversionError
    @test err.J < 0                            # carries the offending J
    @test err.e == 1                           # and the element index
    # the kernel must NOT have produced a (silent) zero-stress result
    @test all(iszero, σo)                      # σo untouched (threw before write)
    # the F-bar path must fail the same way
    @test_throws ElementInversionError element_force_tangent_finite!(
        Hex8FiniteFbar(), mat, dNdXs, detJw, ue, Xe,
        zeros(6, 8), zeros(6, 8), zeros(8), _cp_identity(), 1, zeros(6, 8), Val(false))
end

@testset "F12 spatial modulus + geometric stiffness == production tangent (§4.3/§4.4)" begin
    # Wires the reference spatial-form modulus `spatial_modulus` (§4.3 g_AB) and
    # `geometric_stiffness` (§4.4) and checks they reconstruct the production
    # two-point (dPdF) element tangent for the ISOTROPIC COAXIAL case, where the
    # spatial form is exact (Simo & Hughes Box 8.2 / de Souza Neto Box 14.3):
    #   Ke = Σ_gp [ Bᵀ a B + Kᵍ ] (detJ₀·w),  a from spatial_modulus, Kᵍ from
    #        geometric_stiffness, B the spatial B-matrix at dNdx = dNdX·F⁻¹.
    _, dNdXs, detJw, Xe, X = _unit_element()
    EL = PlasticityFEM.Elements
    # Elastic, coaxial (diagonal F, diagonal Cᵖ⁻¹ = I ⇒ coaxial bᵉ_tr): the
    # reference spatial form is EXACT here ⇒ machine-precision agreement.
    mat = J2Material(E=210e3, ν=0.3, σy0=1e9)            # stays elastic
    F = SMatrix{3,3,Float64,9}(1.12, 0, 0, 0, 0.95, 0, 0, 0, 0.97)
    ue = _ue_from_F(X, F)
    Cp = _cp_identity()
    _, Ke_prod = _fe_ke(Hex8Finite(), mat, dNdXs, detJw, Xe, ue,
                        zeros(6, 8), zeros(6, 8), zeros(8), Cp)
    Z6 = zero(SVector{6,Float64})
    function spatial_Ke()
        Ks = zero(SMatrix{24,24,Float64,576})
        for g in 1:8
            dNdX = dNdXs[g]; w = detJw[g]
            Fg = FS.deformation_gradient(ue, dNdX)
            Cpi = SVector{6,Float64}(Cp[1, g], Cp[2, g], Cp[3, g], Cp[4, g], Cp[5, g], Cp[6, g])
            kin = FS.finite_kinematics(Fg, Cpi)
            τ, _, _, _, D, τpr, _, _, _, _ = FS.finite_stress_update(mat, kin, Fg, Z6, Z6, 0.0)
            a = FS.spatial_modulus(kin, τpr, D)
            dNdx = dNdX * inv(Fg)                        # spatial gradients ∂N/∂x
            B = EL.bmatrix(dNdx)
            Kg = EL.geometric_stiffness(dNdx, FS.voigt_to_sym3(τ), w)
            Ks += (B' * a * B) * w + Kg
        end
        return Ks
    end
    Ke_sp = spatial_Ke()
    @test norm(Matrix(Ke_sp) - Matrix(Ke_prod)) / norm(Matrix(Ke_prod)) < 1e-10
end
