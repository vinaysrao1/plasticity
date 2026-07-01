# =============================================================================
# hard_validation.jl — independent V&V + performance suite for PlasticityFEM
# =============================================================================
#
# This file is BOTH includable from runtests.jl AND runnable on its own:
#     jl --project=. test/hard_validation.jl
#
# It adds HARD, INDEPENDENT tests beyond the existing T1–T20 plan. It validates
# the ACTUAL (corrected) physics: the implementer uses the associative normal
# N = √(3/2)·n̂ with ᾱ the true equivalent plastic strain, and a correspondingly
# corrected consistent tangent (Materials.jl). Tests are built around that, not
# the design's pre-correction formulas.
#
# This suite originally surfaced a transposed-isoparametric-Jacobian bug (visible
# only on sheared/distorted elements, since axis-aligned box meshes have diagonal
# Jacobians). That bug has since been fixed in Elements.jacobian; section B7b now
# verifies the corrected element directly with a distorted-element patch test.
# =============================================================================

using PlasticityFEM
using PlasticityFEM.Materials
using PlasticityFEM.Elements
using PlasticityFEM.Assembly
using StaticArrays
using LinearAlgebra
using SparseArrays
using Random
using Test

const HV_Z6 = zero(SVector{6,Float64})

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

# Central-difference Jacobian dσ/dε of the return map at a given committed state.
function fd_tangent(mat, ε::SVector{6,Float64}, εp, β, ᾱ; h=1e-7)
    D = zeros(6, 6)
    @inbounds for j in 1:6
        ej = SVector{6,Float64}(ntuple(k -> k == j ? h : 0.0, 6))
        σp = return_map(mat, ε + ej, εp, β, ᾱ)[1]
        σm = return_map(mat, ε - ej, εp, β, ᾱ)[1]
        D[:, j] = (σp - σm) / (2h)
    end
    return D
end

# Drive a single Gauss point to a uniaxial-stress state at axial strain `exx`,
# advancing the committed history. Returns (σ_axial, εp, β, ᾱ) at the END state.
# Lateral strains are Newton-solved so σ_yy = σ_zz = 0 (like the T19 helper).
function uniaxial_advance(mat, exx, εp, β, ᾱ; inner=80)
    εlat = 0.0
    for _ in 1:inner
        ε = SVector{6,Float64}(exx, εlat, εlat, 0, 0, 0)
        σ, _, _, _, D = return_map(mat, ε, εp, β, ᾱ)
        d = D[2, 2] + D[2, 3]
        d == 0 && break
        εlat -= σ[2] / d
        abs(σ[2]) < 1e-12 && break
    end
    ε = SVector{6,Float64}(exx, εlat, εlat, 0, 0, 0)
    σ, εp2, β2, ᾱ2, _ = return_map(mat, ε, εp, β, ᾱ)
    return σ[1], εp2, β2, ᾱ2
end

# 8×3 element node coordinates from a mesh, as the code's SMatrix convention.
function hv_elem_coords(mesh, e)
    SMatrix{8,3,Float64,24}(ntuple(24) do k
        a = (k - 1) % 8 + 1
        j = (k - 1) ÷ 8 + 1
        mesh.nodes[j, mesh.elements[a, e]]
    end)
end

# Voigt permutation matrix for the cyclic coordinate relabel x→y→z→x.
# Normal stresses xx→yy→zz→xx; shears xy→yz→zx→xy.
const HV_PERM = let P = zeros(6, 6)
    P[2, 1] = 1; P[3, 2] = 1; P[1, 3] = 1   # normal
    P[5, 4] = 1; P[6, 5] = 1; P[4, 6] = 1   # shear
    SMatrix{6,6,Float64,36}(P)
end

@testset "hard_validation" begin

# ===========================================================================
# A. CONSTITUTIVE DEEP-VALIDATION
# ===========================================================================

# --- A1: consistent tangent vs central FD at MANY states ------------------
# Single most important correctness test: guards quadratic Newton convergence.
@testset "A1 consistent tangent vs FD (many states)" begin
    mats = (
        J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1500.0, Hkin=0.0),   # pure iso
        J2Material(E=70e3,  ν=0.33, σy0=120.0, Hiso=0.0,    Hkin=900.0),  # pure kin
        J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=800.0),  # combined
        J2Material(E=200e3, ν=0.49, σy0=300.0, Hiso=300.0, Hkin=300.0),  # near-incompr.
    )
    # representative deterministic states: uniaxial, biaxial, pure-shear, 3D
    base_states = (
        SVector{6,Float64}(0.01, 0, 0, 0, 0, 0),                 # uniaxial, large Δγ
        SVector{6,Float64}(0.0013, 0, 0, 0, 0, 0),               # just past yield, small Δγ
        SVector{6,Float64}(0.006, 0.006, 0, 0, 0, 0),            # biaxial
        SVector{6,Float64}(0.0, 0.0, 0.0, 0.006, 0, 0),          # pure shear
        SVector{6,Float64}(0.004, -0.002, 0.001, 0.003, 0.002, 0.0015), # full 3D
    )
    worst = 0.0
    Random.seed!(20240607)
    for mat in mats
        # deterministic states
        for ε in base_states
            σ, _, _, _, D = return_map(mat, ε, HV_Z6, HV_Z6, 0.0)
            # only meaningful where plastic (elastic returns ℂ exactly)
            Dfd = fd_tangent(mat, ε, HV_Z6, HV_Z6, 0.0)
            rel = norm(Matrix(D) - Dfd) / norm(Matrix(D))
            worst = max(worst, rel)
            @test rel <= 1e-5
        end
        # random full-3D plastic increments, including non-trivial committed history
        for _ in 1:25
            ε1 = SVector{6,Float64}(0.004 .* (rand(6) .- 0.5) .* 2)
            # build a committed plastic history by one return
            _, εp1, β1, ᾱ1, _ = return_map(mat, ε1, HV_Z6, HV_Z6, 0.0)
            ε2 = ε1 + SVector{6,Float64}(0.003 .* (rand(6) .- 0.5) .* 2)
            σ2, _, _, _, D2 = return_map(mat, ε2, εp1, β1, ᾱ1)
            Dfd2 = fd_tangent(mat, ε2, εp1, β1, ᾱ1)
            rel = norm(Matrix(D2) - Dfd2) / norm(Matrix(D2))
            worst = max(worst, rel)
            @test rel <= 1e-5
        end
    end
    @info "A1 worst FD-tangent relative error" worst
end

# --- A2: Drucker stability & tangent symmetry ------------------------------
# dσ:dεᵖ ≥ 0 (associated J2 is stable) and D_alg symmetric.
@testset "A2 Drucker stability + tangent symmetry" begin
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=800.0)
    Random.seed!(11)
    for _ in 1:40
        ε = SVector{6,Float64}(0.006 .* (rand(6) .- 0.5) .* 2)
        σ, εp, β, ᾱ, D = return_map(mat, ε, HV_Z6, HV_Z6, 0.0)
        # symmetry to 1e-10 (relative)
        @test norm(Matrix(D) - Matrix(D)') <= 1e-10 * norm(Matrix(D))
        if ᾱ > 0   # plastic step: check Drucker
            # Δεᵖ (tensor) work conjugate to σ. Plastic work increment over the
            # step from the (zero) committed state: dσ·Δεᵖ_tensor ≥ 0. Use the
            # tensor double contraction (engineering shear halves on Δεp shear).
            Δσ = σ                              # stress change from unstressed start
            Δεp = εp                            # plastic strain change (engineering Voigt)
            # tensor contraction σ:εᵖ = Σ σ_i εp_i with shear weight 1/2·2 = 1 for
            # engineering shear paired with physical stress → Σ_{1:3} + Σ_{4:6}.
            work = Δσ[1]*Δεp[1] + Δσ[2]*Δεp[2] + Δσ[3]*Δεp[3] +
                   Δσ[4]*Δεp[4] + Δσ[5]*Δεp[5] + Δσ[6]*Δεp[6]
            @test work >= -1e-12
        end
    end
end

# --- A3: closed-form uniaxial curve, COMBINED iso+kin hardening ------------
# Post-yield uniaxial tangent for combined linear hardening is
#   H_t = E·(Hiso+Hkin) / (E + Hiso + Hkin)
# (kinematic and isotropic moduli enter the 1-D plastic modulus identically).
# Independent of the code's derivation; matches to machine precision.
@testset "A3 combined-hardening uniaxial closed form" begin
    E = 210e3; ν = 0.3; σy0 = 250.0; Hiso = 1000.0; Hkin = 800.0
    mat = J2Material(E=E, ν=ν, σy0=σy0, Hiso=Hiso, Hkin=Hkin)
    εy = σy0 / E
    Ht = E * (Hiso + Hkin) / (E + Hiso + Hkin)
    for εtarget in (0.003, 0.006, 0.01)
        εp = HV_Z6; β = HV_Z6; ᾱ = 0.0
        nsteps = 80
        local σax = 0.0
        for n in 1:nsteps
            σax, εp, β, ᾱ = uniaxial_advance(mat, εtarget * n / nsteps, εp, β, ᾱ)
        end
        σ_an = σy0 + Ht * (εtarget - εy)
        @test σax ≈ σ_an rtol=1e-6
    end
end

# --- A4: Bauschinger / cyclic, no spurious ratcheting ----------------------
@testset "A4 Bauschinger reverse yield + closed cycle" begin
    σy0 = 250.0; Hkin = 2000.0
    mat = J2Material(E=210e3, ν=0.3, σy0=σy0, Hiso=0.0, Hkin=Hkin)
    # forward to +εmax, reverse to −εmax with fine steps; detect reverse yield by
    # ᾱ restarting, and confirm elastic span = 2σy0.
    εmax = 0.01
    εp = HV_Z6; β = HV_Z6; ᾱ = 0.0
    σf = Float64[]; αf = Float64[]
    fwd = collect(range(0, εmax, length=400))
    rev = collect(range(εmax, -εmax, length=800))
    path = vcat(fwd, rev)
    for exx in path
        σax, εp, β, ᾱ = uniaxial_advance(mat, exx, εp, β, ᾱ)
        push!(σf, σax); push!(αf, ᾱ)
    end
    σpeak = maximum(σf)
    nf = length(fwd)
    rstart = 0
    for k in (nf+2):length(σf)
        if αf[k] > αf[k-1] + 1e-12
            rstart = k; break
        end
    end
    @test rstart > 0
    σ_revyield = σf[rstart-1]
    @test (σpeak - σ_revyield) ≈ 2σy0 rtol=2e-2

    # Closed symmetric strain cycle returns to a consistent stress: run a full
    # cycle 0→+εmax→−εmax→0 then again to +εmax and compare the second peak to
    # the first reversal-corrected peak. Pure kinematic hardening: a *symmetric*
    # cycle has no isotropic growth, so successive +peaks must be EQUAL (no
    # ratcheting / no drift beyond what kinematic hardening dictates).
    εp = HV_Z6; β = HV_Z6; ᾱ = 0.0
    cyc = vcat(collect(range(0, εmax, length=200)),
               collect(range(εmax, -εmax, length=400)),
               collect(range(-εmax, εmax, length=400)))
    peaks = Float64[]
    prev = -Inf
    σprev = 0.0
    for (i, exx) in enumerate(cyc)
        σax, εp, β, ᾱ = uniaxial_advance(mat, exx, εp, β, ᾱ)
        # record local maxima near the +εmax turning points
        if i > 1 && abs(exx - εmax) < 1e-9
            push!(peaks, σax)
        end
        σprev = σax
    end
    @test length(peaks) >= 2
    # symmetric cyclic with linear kinematic hardening: equal positive peaks
    @test peaks[end] ≈ peaks[1] rtol=1e-3
end

# --- A5: plastic incompressibility + isotropy (coordinate permutation) -----
@testset "A5 incompressibility + permutation isotropy" begin
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)
    Random.seed!(5)
    for _ in 1:30
        ε = SVector{6,Float64}(0.006 .* (rand(6) .- 0.5) .* 2)
        σ, εp, β, ᾱ, _ = return_map(mat, ε, HV_Z6, HV_Z6, 0.0)
        # tr(Δεᵖ) = 0
        @test εp[1] + εp[2] + εp[3] ≈ 0.0 atol=1e-12
        # mean stress equals trial mean stress (plastic correction is deviatoric)
        σtr = mat.Cmat * ε
        @test (σ[1]+σ[2]+σ[3])/3 ≈ (σtr[1]+σtr[2]+σtr[3])/3 atol=1e-9
        # isotropy: permuting strain axes permutes the stress response.
        # Permutation acts on engineering-shear strain Voigt the SAME way as on
        # physical-shear stress Voigt (the cyclic relabel maps shear→shear with
        # no factor), so σ(Pε) must equal P·σ(ε). Catches Voigt-index / shear-
        # factor bugs in the kernel.
        εperm = HV_PERM * ε
        σperm, εpperm, _, _, _ = return_map(mat, εperm, HV_Z6, HV_Z6, 0.0)
        @test Vector(σperm) ≈ HV_PERM * Vector(σ) atol=1e-9 * (norm(Vector(σ)) + 1)
        @test Vector(εpperm) ≈ HV_PERM * Vector(εp) atol=1e-12
    end
end

# --- A6: pathological inputs -----------------------------------------------
@testset "A6 pathological inputs (no NaN, caps)" begin
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)
    # neutral loading / near-zero deviator (‖ξ‖→0): pure volumetric strain.
    εvol = SVector{6,Float64}(1e-3, 1e-3, 1e-3, 0, 0, 0)
    σ, εp, β, ᾱ, D = return_map(mat, εvol, HV_Z6, HV_Z6, 0.0)
    @test all(isfinite, σ); @test all(isfinite, Matrix(D))
    @test ᾱ == 0.0                       # purely elastic (no deviatoric stress)
    # vanishingly small deviator on top of volumetric — still elastic, finite
    εtiny = SVector{6,Float64}(1e-3, 1e-3, 1e-3, 1e-14, 0, 0)
    σ2, _, _, _, D2 = return_map(mat, εtiny, HV_Z6, HV_Z6, 0.0)
    @test all(isfinite, σ2); @test all(isfinite, Matrix(D2))

    # exactly-at-yield: construct ε so that q_trial == σy0 (f_trial == 0). Pure
    # shear γ gives q = √3·G·γ. Solve γ so q = σy0.
    G = mat.G
    γ = mat.σy0 / (sqrt(3.0) * G)
    εat = SVector{6,Float64}(0, 0, 0, γ, 0, 0)
    σ3, εp3, _, ᾱ3, D3 = return_map(mat, εat, HV_Z6, HV_Z6, 0.0)
    @test all(isfinite, σ3); @test all(isfinite, Matrix(D3))
    # f_trial == 0 is the elastic boundary (code uses f_tr ≤ 0 → elastic): no flow
    @test ᾱ3 == 0.0
    @test Matrix(D3) ≈ Matrix(mat.Cmat) rtol=1e-12

    # perfect plasticity caps at σy0 over many uniaxial steps.
    matpp = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=0.0, Hkin=0.0)
    εp = HV_Z6; β = HV_Z6; ᾱ = 0.0
    σax = 0.0
    for n in 1:300
        σax, εp, β, ᾱ = uniaxial_advance(matpp, 0.02 * n / 300, εp, β, ᾱ)
        @test isfinite(σax)
        @test σax <= matpp.σy0 * (1 + 1e-9)
    end
    @test σax ≈ matpp.σy0 rtol=1e-3
end

# ===========================================================================
# B. ELEMENT / FE MATH
# ===========================================================================

# --- B7: patch tests --------------------------------------------------------
#
# Axis-aligned multi-element patch test (B7) and a distorted single-element
# patch test (B7b) — both must recover a constant strain/stress field exactly.
@testset "B7 patch test (axis-aligned multi-element)" begin
    # 3×2×2 axis-aligned mesh, impose a general LINEAR displacement field
    # u = A·x on ALL boundary nodes, leave interior nodes free, solve, and
    # require every Gauss point of every element to recover the EXACT constant
    # stress ℂ:ε. Axis-aligned ⇒ constant diagonal Jacobian ⇒ must be exact.
    lx, ly, lz = 1.0, 1.0, 1.0
    nx, ny, nz = 3, 2, 2
    mesh = box_mesh(lx, ly, lz, nx, ny, nz)
    mat = J2Material(E=210e3, ν=0.3, σy0=1e9)   # stay elastic
    model = Model(mesh, mat)
    A = @SMatrix [1.0e-3 2.0e-4 1.0e-4;
                  2.0e-4 -5.0e-4 1.5e-4;
                  1.0e-4 1.5e-4 3.0e-4]   # symmetric ⇒ no rigid rotation in field
    tol = 1e-8
    onbnd(x, y, z) = (abs(x) <= tol || abs(x - lx) <= tol ||
                      abs(y) <= tol || abs(y - ly) <= tol ||
                      abs(z) <= tol || abs(z - lz) <= tol)
    for n in 1:mesh.nnodes
        x = mesh.nodes[1, n]; y = mesh.nodes[2, n]; z = mesh.nodes[3, n]
        if onbnd(x, y, z)
            u = A * SVector{3,Float64}(x, y, z)
            prescribe!(model, [n], :x, u[1]; ramp=false)
            prescribe!(model, [n], :y, u[2]; ramp=false)
            prescribe!(model, [n], :z, u[3]; ramp=false)
        end
    end
    solve!(model; nsteps=1, tol=1e-12, maxiter=10)
    σ = gauss_stress(model)
    εv = SVector{6,Float64}(A[1,1], A[2,2], A[3,3],
                            A[1,2]+A[2,1], A[2,3]+A[3,2], A[3,1]+A[1,3])
    σ_exact = mat.Cmat * εv
    maxerr = 0.0
    for g in 1:size(σ, 2)
        maxerr = max(maxerr, norm(σ[:, g] - Vector(σ_exact)))
    end
    @test maxerr <= 1e-9 * (norm(Vector(σ_exact)) + 1)
    # interior nodes must sit on the linear field (equilibrium of a correct
    # element places them there).
    u = nodal_displacements(model)
    interr = 0.0
    for n in 1:mesh.nnodes
        x = mesh.nodes[1, n]; y = mesh.nodes[2, n]; z = mesh.nodes[3, n]
        if !onbnd(x, y, z)
            uexp = A * SVector{3,Float64}(x, y, z)
            interr = max(interr, norm(u[:, n] - Vector(uexp)))
        end
    end
    @test interr <= 1e-10
end

@testset "B7b single-element patch test (sheared / distorted element)" begin
    # ---------------------------------------------------------------------
    # The isoparametric Jacobian is J_ij = ∂x_i/∂ξ_j = Σ_a x_{a,i} ∂N_a/∂ξ_j =
    # (Xe' * dN)_ij, with spatial gradients dN/dx = dN · J⁻¹. For an axis-aligned
    # box J is diagonal, so a transpose error would be invisible there; this test
    # uses a NON-symmetric (sheared) Jacobian, where any transpose error in J or
    # the B-matrix breaks the constant-strain patch test. Two checks:
    #   (1) completeness  Σ_a x_a ⊗ ∇N_a = I  (must be machine-zero), and
    #   (2) constant-strain recovery: impose a linear field u = A·x on the element
    #       nodes; B·u_e at every Gauss point must equal Voigt(sym(A)) exactly.
    # ---------------------------------------------------------------------
    NN = PlasticityFEM.Elements.NODE_NAT
    S = @SMatrix [1.0 0.15 0.05; 0.0 1.0 0.1; 0.0 0.0 1.0]   # non-symmetric shear
    Xe = SMatrix{8,3,Float64,24}(ntuple(24) do k
        a = (k - 1) % 8 + 1
        j = (k - 1) ÷ 8 + 1
        (S * SVector{3,Float64}(NN[a,1], NN[a,2], NN[a,3]))[j]
    end)

    # (1) completeness on this constant-Jacobian element is machine-zero for a
    #     correct element (would be ≈0.26 with a transposed Jacobian).
    maxcomp = 0.0
    for ξ in PlasticityFEM.Elements.GAUSS_PTS
        dN = hex8_dshape(ξ)
        J = PlasticityFEM.Elements.jacobian(Xe, dN)
        dNdx = dN * inv(J)
        C = @SMatrix [sum(Xe[a, i] * dNdx[a, j] for a in 1:8) for i in 1:3, j in 1:3]
        maxcomp = max(maxcomp, norm(Matrix(C) - I))
    end
    @test maxcomp <= 1e-12

    # (2) constant-strain patch test on the distorted element. Linear field
    #     u(x) = A·x ⇒ exact strain ε = sym(A) everywhere. Build nodal dofs and
    #     check B·u_e == Voigt(sym(A)) (engineering shear) at every Gauss point.
    A = @SMatrix [0.003 0.001 -0.002; 0.0005 -0.001 0.0015; 0.002 0.0 0.0007]
    ue = SVector{24,Float64}(ntuple(24) do c
        a = (c - 1) ÷ 3 + 1
        i = (c - 1) % 3 + 1
        (A * SVector{3,Float64}(Xe[a,1], Xe[a,2], Xe[a,3]))[i]
    end)
    εxx = A[1,1]; εyy = A[2,2]; εzz = A[3,3]
    γxy = A[1,2] + A[2,1]; γyz = A[2,3] + A[3,2]; γzx = A[3,1] + A[1,3]
    εexact = SVector{6,Float64}(εxx, εyy, εzz, γxy, γyz, γzx)
    maxstrainerr = 0.0
    for ξ in PlasticityFEM.Elements.GAUSS_PTS
        dN = hex8_dshape(ξ)
        J = PlasticityFEM.Elements.jacobian(Xe, dN)
        B = PlasticityFEM.Elements.bmatrix(dN * inv(J))
        maxstrainerr = max(maxstrainerr, norm(B * ue - εexact))
    end
    @test maxstrainerr <= 1e-12
end

# --- B8: rigid body modes of the free-free assembled K ---------------------
@testset "B8 free-free K has exactly 6 zero modes" begin
    mesh = box_mesh(1.0, 1.0, 1.0, 2, 2, 2)
    mat = J2Material(E=210e3, ν=0.3, σy0=1e9)   # elastic
    model = Model(mesh, mat)
    U = zeros(3 * mesh.nnodes)
    st = model.state_trial
    K, _ = assemble!(model.sparsity, mat, model.cache, U, st.εp, st.β, st.ᾱ, st.σ, model.Rbuf)
    Kd = Symmetric(Matrix(K))
    ev = eigvals(Kd)
    scale = maximum(abs, ev)
    nzero = count(<(1e-8 * scale), abs.(ev))
    @test nzero == 6                              # 3 translations + 3 rotations
    # all other eigenvalues strictly positive (semi-definite, no negatives)
    @test minimum(ev) >= -1e-8 * scale
    @test count(>(1e-8 * scale), ev) == length(ev) - 6
end

# --- B9: assembly equals an independent triplet assembly -------------------
@testset "B9 assembly == independent sparse(I,J,V) + symmetry" begin
    mesh = box_mesh(1.0, 1.0, 1.0, 2, 2, 2)
    mat = J2Material(E=210e3, ν=0.3, σy0=1e9)   # elastic ⇒ K independent of U
    model = Model(mesh, mat)
    ndof = 3 * mesh.nnodes
    U = zeros(ndof)
    st = model.state_trial
    K, _ = assemble!(model.sparsity, mat, model.cache, U, st.εp, st.β, st.ᾱ, st.σ, model.Rbuf)

    # independent from-scratch COO triplet assembly using the SAME element kernel
    Ivec = Int[]; Jvec = Int[]; Vvec = Float64[]
    εp = zeros(6, 8); β = zeros(6, 8); ᾱ = zeros(8); σo = zeros(6, 8)
    for e in 1:mesh.nelem
        edofs = ntuple(24) do c
            a = (c - 1) ÷ 3 + 1; comp = (c - 1) % 3 + 1
            3 * (mesh.elements[a, e] - 1) + comp
        end
        ue = SVector{24,Float64}(ntuple(i -> U[edofs[i]], 24))
        Bs_e, Jw_e = PlasticityFEM.Elements.element_geometry(model.cache, e)
        _, Ke = element_force_tangent!(mat, Bs_e, Jw_e,
                                       ue, εp, β, ᾱ, 1, σo, Val(false))
        for c in 1:24, r in 1:24
            push!(Ivec, edofs[r]); push!(Jvec, edofs[c]); push!(Vvec, Ke[r, c])
        end
    end
    Kindep = sparse(Ivec, Jvec, Vvec, ndof, ndof)
    @test norm(Matrix(K) - Matrix(Kindep)) <= 1e-12 * norm(Matrix(K))
    # global symmetry
    Kd = Matrix(K)
    @test norm(Kd - Kd') <= 1e-10 * norm(Kd)
end

# --- B10: reaction / energy balance at a converged elastic step ------------
@testset "B10 reaction + energy balance (force & displacement control)" begin
    # FORCE control: ∑reactions + ∑applied = 0, and external work = internal energy.
    mesh = box_mesh(4.0, 1.0, 1.0, 4, 2, 2)
    mat = J2Material(E=210e3, ν=0.3, σy0=1e9)   # elastic
    model = Model(mesh, mat)
    fix!(model, on_face(mesh, :xmin))
    Ptot = -40.0
    loadnodes = on_face(mesh, :xmax)
    load!(model, loadnodes, :z, Ptot; distribute=true)
    solve!(model; nsteps=2, tol=1e-12)

    st = model.state_trial
    K, Fint = assemble!(model.sparsity, mat, model.cache, model.U,
                        st.εp, st.β, st.ᾱ, st.σ, copy(model.Rbuf))
    # reactions = F_int at fixed (xmin) dofs; balance the applied z load.
    fixnodes = on_face(mesh, :xmin)
    Rz = sum(Fint[3*(n-1)+3] for n in fixnodes)
    @test Rz + Ptot ≈ 0.0 atol=1e-7 * abs(Ptot) + 1e-7
    # full reaction vector balance in all 3 components
    Rx = sum(Fint[3*(n-1)+1] for n in fixnodes)
    Ry = sum(Fint[3*(n-1)+2] for n in fixnodes)
    @test Rx ≈ 0.0 atol=1e-6
    @test Ry ≈ 0.0 atol=1e-6
    # energy: external work W_ext = F_applied · u (on loaded dofs) equals
    # internal strain energy ½ Uᵀ K U for a linear-elastic converged state.
    Wext = 0.0
    for n in loadnodes
        Wext += (Ptot / length(loadnodes)) * model.U[3*(n-1)+3]
    end
    Uint = 0.5 * dot(model.U, Matrix(K) * model.U)
    @test Wext ≈ 2 * Uint rtol=1e-6   # W_ext = UᵀKU = 2·(½UᵀKU); both = work done

    # DISPLACEMENT control: prescribe a stretch, reactions on the pulled face
    # must balance reactions on the held face.
    mesh2 = box_mesh(1.0, 1.0, 1.0, 2, 2, 2)
    model2 = Model(mesh2, mat)
    fix!(model2, on_face(mesh2, :xmin), :x)
    fix!(model2, on_face(mesh2, :ymin), :y)
    fix!(model2, on_face(mesh2, :zmin), :z)
    prescribe!(model2, on_face(mesh2, :xmax), :x, 1e-3)
    solve!(model2; nsteps=1, tol=1e-12)
    st2 = model2.state_trial
    _, Fint2 = assemble!(model2.sparsity, mat, model2.cache, model2.U,
                         st2.εp, st2.β, st2.ᾱ, st2.σ, copy(model2.Rbuf))
    # reaction on xmin (held, x) + reaction on xmax (pulled, x) = 0
    Rx_min = sum(Fint2[3*(n-1)+1] for n in on_face(mesh2, :xmin))
    Rx_max = sum(Fint2[3*(n-1)+1] for n in on_face(mesh2, :xmax))
    @test Rx_min + Rx_max ≈ 0.0 atol=1e-8 * (abs(Rx_max) + 1)
end

# ===========================================================================
# C. PERFORMANCE & RESOURCE VALIDATION
# ===========================================================================

# --- C11: zero-allocation hot kernels --------------------------------------
@testset "C11 zero-allocation hot kernels" begin
    # return_map: plastic + elastic, measured inside a function (no global boxing).
    function rm_alloc()
        mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)
        Z = zero(SVector{6,Float64})
        εpl = SVector{6,Float64}(0.01, 0, 0, 0, 0, 0)
        εel = SVector{6,Float64}(1e-5, 0, 0, 0, 0, 0)
        return_map(mat, εpl, Z, Z, 0.0); return_map(mat, εel, Z, Z, 0.0)  # warmup
        ap = @allocated return_map(mat, εpl, Z, Z, 0.0)
        ae = @allocated return_map(mat, εel, Z, Z, 0.0)
        return ap, ae
    end
    ap, ae = rm_alloc()
    @test ap == 0
    @test ae == 0

    # element_force_tangent!: elastic & plastic, commit & non-commit.
    function elt_alloc()
        mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)
        mesh = box_mesh(1.0, 1.0, 1.0, 1, 1, 1)
        cache = precompute_cache(mesh.nodes, mesh.elements)
        εp = zeros(6, 8); β = zeros(6, 8); ᾱ = zeros(8); σ = zeros(6, 8)
        B1, Jw1 = element_geometry(cache, 1)
        ue_el = SVector{24,Float64}(ntuple(i -> 1e-6 * i, 24))   # elastic
        ue_pl = SVector{24,Float64}(ntuple(i -> 1e-3 * i, 24))   # plastic
        # warmups (all four branches)
        element_force_tangent!(mat, B1, Jw1, ue_el, εp, β, ᾱ, 1, σ, Val(false))
        element_force_tangent!(mat, B1, Jw1, ue_pl, εp, β, ᾱ, 1, σ, Val(false))
        element_force_tangent!(mat, B1, Jw1, ue_pl, εp, β, ᾱ, 1, σ, Val(true))
        a_el  = @allocated element_force_tangent!(mat, B1, Jw1, ue_el, εp, β, ᾱ, 1, σ, Val(false))
        a_pl  = @allocated element_force_tangent!(mat, B1, Jw1, ue_pl, εp, β, ᾱ, 1, σ, Val(false))
        a_com = @allocated element_force_tangent!(mat, B1, Jw1, ue_pl, εp, β, ᾱ, 1, σ, Val(true))
        return a_el, a_pl, a_com
    end
    a_el, a_pl, a_com = elt_alloc()
    @test a_el == 0
    @test a_pl == 0
    @test a_com == 0
end

# --- C12: O(1) assembly allocation across mesh sizes -----------------------
@testset "C12 assemble! allocation is O(1) in nelem" begin
    function asm_alloc(nx)
        mesh = box_mesh(1.0, 1.0, 1.0, nx, nx, nx)
        mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0)
        model = Model(mesh, mat)
        U = zeros(3 * mesh.nnodes); st = model.state_trial
        assemble!(model.sparsity, mat, model.cache, U, st.εp, st.β, st.ᾱ, st.σ, model.Rbuf)  # warmup
        return @allocated assemble!(model.sparsity, mat, model.cache, U,
                                    st.εp, st.β, st.ᾱ, st.σ, model.Rbuf)
    end
    a8   = asm_alloc(2)    # 8 elements
    a64  = asm_alloc(4)    # 64 elements
    a512 = asm_alloc(8)    # 512 elements
    @info "C12 assemble! allocations (bytes)" a8 a64 a512
    # small constant, independent of nelem (64× span here)
    @test a8 < 4096
    @test a64 < 4096
    @test a512 < 4096
    # near-constant across sizes (tight band; allow tiny machinery wobble)
    @test abs(a512 - a8) <= 256
end

# --- C13: memory footprint scales ~linearly with nelem ---------------------
@testset "C13 model memory + nnz scale ~linearly" begin
    nelems = Int[]; nnzs = Int[]; states = Int[]
    for nx in (2, 4, 6, 8)
        mesh = box_mesh(1.0, 1.0, 1.0, nx, nx, nx)
        mat = J2Material(E=210e3, ν=0.3, σy0=250.0)
        model = Model(mesh, mat)
        push!(nelems, mesh.nelem)
        push!(nnzs, nnz(model.sparsity.K))
        # state storage bytes (committed + trial SoA arrays)
        sc = model.state_committed; stt = model.state_trial
        nb = sum(sizeof, (sc.εp, sc.β, sc.ᾱ, sc.σ, stt.εp, stt.β, stt.ᾱ, stt.σ))
        push!(states, nb)
    end
    # per-element ratios must stay bounded (no super-linear blowup).
    rnnz = nnzs ./ nelems
    rstate = states ./ nelems
    @info "C13 scaling" nelems nnzs rnnz_ratio=(maximum(rnnz)/minimum(rnnz)) rstate_per_elem=rstate
    @test maximum(rnnz) / minimum(rnnz) < 2.0       # nnz ~ O(nelem)
    @test all(≈(rstate[1]), rstate)                 # state is exactly linear (SoA)
end

# --- C14: assemble! compute time scales ~linearly --------------------------
@testset "C14 assemble! time ~linear in nelem (sanity gate)" begin
    # SANITY GATE with JIT/noise slack: per-element assemble time should be
    # roughly constant. We assert the max/min per-element time ratio is bounded.
    function asm_time(nx; reps=30)
        mesh = box_mesh(1.0, 1.0, 1.0, nx, nx, nx)
        mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0)
        model = Model(mesh, mat)
        U = zeros(3 * mesh.nnodes); st = model.state_trial
        assemble!(model.sparsity, mat, model.cache, U, st.εp, st.β, st.ᾱ, st.σ, model.Rbuf)  # warmup
        best = Inf
        for _ in 1:3   # take the best of a few timing loops to suppress noise
            t0 = time_ns()
            for _ in 1:reps
                assemble!(model.sparsity, mat, model.cache, U, st.εp, st.β, st.ᾱ, st.σ, model.Rbuf)
            end
            best = min(best, (time_ns() - t0) / reps)
        end
        return mesh.nelem, best / mesh.nelem      # ns per element
    end
    res = [asm_time(nx) for nx in (4, 6, 8, 10)]
    perelem = [r[2] for r in res]
    ratio = maximum(perelem) / minimum(perelem)
    @info "C14 ns/element across sweep" nelems=[r[1] for r in res] perelem ratio
    @test ratio < 3.0    # generous slack: linear scaling ⇒ flat per-element time
end

# --- C15: type stability of the hot path -----------------------------------
@testset "C15 type stability (@inferred hot kernels)" begin
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)
    ε = SVector{6,Float64}(0.01, 0, 0, 0, 0, 0)
    # return_map fully inferred (no Union/Any). Throws if inference fails.
    rt = @inferred return_map(mat, ε, HV_Z6, HV_Z6, 0.0)
    @test rt[1] isa SVector{6,Float64}
    @test rt[5] isa SMatrix{6,6,Float64,36}

    mesh = box_mesh(1.0, 1.0, 1.0, 1, 1, 1)
    cache = precompute_cache(mesh.nodes, mesh.elements)
    εp = zeros(6, 8); β = zeros(6, 8); ᾱ = zeros(8); σ = zeros(6, 8)
    ue = SVector{24,Float64}(ntuple(i -> 1e-3 * i, 24))
    Bs1, Jw1 = element_geometry(cache, 1)
    ft = @inferred element_force_tangent!(mat, Bs1, Jw1, ue,
                                          εp, β, ᾱ, 1, σ, Val(false))
    @test ft[1] isa SVector{24,Float64}
    @test ft[2] isa SMatrix{24,24,Float64,576}
end

# --- C16: Newton iteration counts on a moderately large plastic problem ----
@testset "C16 Newton scaling (mesh-independent quadratic convergence)" begin
    # 12×12×12 plastic problem; consistent tangent ⇒ small, ~mesh-independent
    # iteration count per load step.
    function newton_iters(n; nsteps=8)
        mesh = box_mesh(1.0, 1.0, 1.0, n, n, n)
        mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)
        model = Model(mesh, mat)
        fix!(model, on_face(mesh, :xmin), :x)
        fix!(model, on_face(mesh, :ymin), :y)
        fix!(model, on_face(mesh, :zmin), :z)
        prescribe!(model, on_face(mesh, :xmax), :x, 0.01)
        res = solve!(model; nsteps=nsteps, tol=1e-8, maxiter=20)
        return res
    end
    res_small = newton_iters(6)
    res_large = newton_iters(12)
    @test res_small.converged
    @test res_large.converged
    @info "C16 iters" small=res_small.iters large=res_large.iters
    @test maximum(res_small.iters) <= 8
    @test maximum(res_large.iters) <= 8
    # mesh-independence: iteration counts should not blow up with refinement.
    @test maximum(res_large.iters) <= maximum(res_small.iters) + 2
    # asymptotic quadratic drop on a clearly plastic step.
    plastic = findfirst(h -> length(h) >= 4, res_large.residuals)
    if plastic !== nothing
        h = res_large.residuals[plastic]
        ok = false
        for k in 2:(length(h)-1)
            if h[k] < 1.0 && h[k+1] <= 50.0 * h[k]^2 + 1e-12
                ok = true
            end
        end
        @test ok
    end
end

end # @testset hard_validation
