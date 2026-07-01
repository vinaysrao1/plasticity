"""
    FiniteStrain

Finite-strain (large-deformation) J2 plasticity kernels: the geometric
pre-/post-processor around the unchanged small-strain `return_map`
(see docs/FINITE_STRAIN.md). Pure, allocation-free, StaticArrays-only.

The central idea (FINITE_STRAIN §0): with the multiplicative split F = Fᵉ·Fᵖ,
Hencky (logarithmic) elasticity and an exponential-map plastic update, the
stress-update problem expressed in the **elastic logarithmic strain** is
algebraically identical to the small-strain radial return. So `return_map` is
reused verbatim; finite strain is a geometric wrapper:

    F → εᵉ_tr (log strain, 6-Voigt) → [return_map] → τ, D
      → exp-map plastic update (Cp_inv), spatial tangent (material + geometric).

Voigt ordering `[xx,yy,zz,xy,yz,zx]`, engineering shear for strain, physical
shear for stress — identical to v1, so `return_map` consumes/produces the same
layout. Symmetric tensors are stored 6-Voigt; F, Fᵖ are full 3×3 SMatrix.
"""
module FiniteStrain

using StaticArrays
using LinearAlgebra
using ..Materials: J2Material, return_map

export ElementKind, Hex8Small, Hex8Finite, Hex8FiniteFbar
export deformation_gradient, finite_kinematics, finite_stress_update,
       spatial_modulus, sym3_to_voigt, voigt_to_sym3, det_Fp_from_Cpinv,
       dPdF, first_piola, polar_RU, dR_polar, dtau_dbeta, ElementInversionError

"""
    ElementInversionError(e, g, J) <: Exception

Thrown when an element's deformation gradient has det F = J ≤ 0 at element index
`e`, Gauss point `g` (a folded / inverted configuration). J ≤ 0 makes the
spectral log-strain decomposition (½ ln bᵉ_tr) ill-defined, so continuing would
silently produce garbage (zero stress, wrong tangent). The solver has no
step-cutting machinery, so this is raised as a clear, typed failure rather than
swallowed. Carries the offending element/GP indices and J for diagnosis.
"""
struct ElementInversionError <: Exception
    e::Int
    g::Int
    J::Float64
end

function Base.showerror(io::IO, err::ElementInversionError)
    print(io, "ElementInversionError: element $(err.e), Gauss point $(err.g) ",
          "is inverted (det F = J = $(err.J) ≤ 0). The log-strain stress update ",
          "requires J > 0; reduce the load step or fix the mesh/BCs.")
end

# --- element-kind dispatch seam (FINITE_STRAIN §6.1) ---

"""
    ElementKind

Type-level selector for the assembly hot loop so it dispatches statically (no
runtime branch in the kernel, mirroring `Val{COMMIT}`/`Val{UNIFORM}`).
"""
abstract type ElementKind end
struct Hex8Small      <: ElementKind end   # v1 path (default), small strain
struct Hex8Finite     <: ElementKind end   # finite strain, standard F
struct Hex8FiniteFbar <: ElementKind end   # finite strain, F-bar (§5)

const I3 = SMatrix{3,3,Float64,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)
# Tolerance for treating two trial eigenvalues as degenerate (repeated stretch).
const EIG_TOL = 1e-9
# Engineering-shear metric W = diag(1,1,1,2,2,2): ⟨a,b⟩ = aᵀ W b reproduces the
# physical-shear tensor inner product for symmetric 6-Voigt tensors.
const W6 = SMatrix{6,6,Float64,36}(Diagonal(SVector{6,Float64}(1, 1, 1, 2, 2, 2)))

# --- Voigt <-> 3×3 symmetric tensor helpers ---

"""
    voigt_to_sym3(v) -> SMatrix{3,3}

Symmetric 3×3 tensor from a 6-Voigt `[xx,yy,zz,xy,yz,zx]` vector with *physical*
(tensor) shear components (no engineering factor). Used for Cp_inv (stored as a
tensor) and the Kirchhoff stress.
"""
@inline function voigt_to_sym3(v::SVector{6,Float64})
    return SMatrix{3,3,Float64,9}(v[1], v[4], v[6],
                                  v[4], v[2], v[5],
                                  v[6], v[5], v[3])
end

"""
    sym3_to_voigt(A) -> SVector{6}

6-Voigt `[xx,yy,zz,xy,yz,zx]` (physical shear) from a symmetric 3×3 tensor.
"""
@inline function sym3_to_voigt(A::SMatrix{3,3,Float64,9})
    return SVector{6,Float64}(A[1, 1], A[2, 2], A[3, 3], A[1, 2], A[2, 3], A[1, 3])
end

# --- §2: kinematics ---

"""
    deformation_gradient(ue, dNdX) -> SMatrix{3,3}

F = I + Σₐ uₐ ⊗ (∂Nₐ/∂X) (FINITE_STRAIN §2.1). `ue::SVector{24}` is the element
nodal displacement (node-major), `dNdX::SMatrix{8,3}` the *reference* shape
gradients ∂Nₐ/∂X.
"""
@inline function deformation_gradient(ue::SVector{24,Float64},
                                      dNdX::SMatrix{8,3,Float64,24})
    # F_ij = δ_ij + Σₐ u_{a,i} ∂N_a/∂X_j
    H = zero(SMatrix{3,3,Float64,9})
    @inbounds for a in 1:8
        ux = ue[3(a - 1) + 1]; uy = ue[3(a - 1) + 2]; uz = ue[3(a - 1) + 3]
        gx = dNdX[a, 1]; gy = dNdX[a, 2]; gz = dNdX[a, 3]
        H += SMatrix{3,3,Float64,9}(ux * gx, uy * gx, uz * gx,
                                    ux * gy, uy * gy, uz * gy,
                                    ux * gz, uy * gz, uz * gz)
    end
    return I3 + H
end

"""
    FiniteKin

Per-Gauss-point finite-strain kinematic data (the geometric pre-processor
output, FINITE_STRAIN §2): the trial elastic log-strain in 6-Voigt
engineering-shear form `εe_tr` (input to `return_map`), the trial-elastic
principal stretches-squared `b` (eigenvalues of bᵉ_tr) and their orthonormal
spatial directions `n` (columns), `Finv = F⁻¹`, `J = det F`, and an `ok` flag
(false if J ≤ 0). Allocation-free value type.
"""
struct FiniteKin
    εe_tr::SVector{6,Float64}   # trial elastic Hencky strain, engineering-shear Voigt
    b::SVector{3,Float64}       # trial elastic eigenvalues (λᵉ_tr_A)²
    n::SMatrix{3,3,Float64,9}   # eigenvectors n_A as columns
    Finv::SMatrix{3,3,Float64,9}
    J::Float64
    ok::Bool
end

"""
    finite_kinematics(F, Cp_inv_n) -> FiniteKin

Geometric pre-processor (FINITE_STRAIN §2.2–2.3): from the deformation gradient
`F` and the committed plastic state `Cp_inv_n` (6-Voigt, physical shear),
form bᵉ_tr = F·Cp_inv_n·Fᵀ, spectrally decompose it, and assemble the trial
Hencky strain εᵉ_tr = ½ ln bᵉ_tr in engineering-shear 6-Voigt. Allocation-free.
"""
@inline function finite_kinematics(F::SMatrix{3,3,Float64,9},
                                   Cp_inv_n::SVector{6,Float64})
    J = det(F)
    if J <= 0
        return FiniteKin(zero(SVector{6,Float64}), SVector{3,Float64}(1, 1, 1),
                         I3, I3, J, false)
    end
    Cpi = voigt_to_sym3(Cp_inv_n)
    be_tr = F * Cpi * F'
    # symmetric eigendecomposition (closed-form via StaticArrays, alloc-free)
    E = eigen(Symmetric(be_tr))
    b = E.values
    n = E.vectors
    # εᵉ_tr = Σ_A ln(λ_A) n_A⊗n_A,  with (λᵉ_A)² = b_A ⇒ ln λ_A = ½ ln b_A
    lnλ = SVector{3,Float64}(0.5 * log(b[1]), 0.5 * log(b[2]), 0.5 * log(b[3]))
    εmat = lnλ[1] * (n[:, 1] * n[:, 1]') +
           lnλ[2] * (n[:, 2] * n[:, 2]') +
           lnλ[3] * (n[:, 3] * n[:, 3]')
    # engineering-shear Voigt (off-diagonals ×2) for return_map's strain input
    εe_tr = SVector{6,Float64}(εmat[1, 1], εmat[2, 2], εmat[3, 3],
                               2εmat[1, 2], 2εmat[2, 3], 2εmat[1, 3])
    return FiniteKin(εe_tr, b, n, inv(F), J, true)
end

# --- polar decomposition (rotation-neutralized back-stress, FINITE_STRAIN §3.4) ---
#
# Kinematic hardening needs the back-stress to be OBJECTIVE: under a superposed
# spatial rotation Q (F → QF) the Kirchhoff stress must rotate as Q τ Qᵀ. The
# trial Hencky strain already rotates correctly (eigenvalues of bᵉ_tr = F Cᵖ⁻¹ Fᵀ
# are Q-invariant, eigenvectors co-rotate), but a back-stress stored in the global
# spatial frame does NOT co-rotate, breaking frame-indifference. The remedy
# (de Souza Neto §14, Simo & Hughes): store β in the rotation-neutralized
# (reference/intermediate) configuration `β_ref` (Q-invariant, like Cᵖ⁻¹) and
# push it forward to the spatial frame with the polar rotation R of F (F = R U):
# β_sp = R β_ref Rᵀ. Under F → QF, R → QR ⇒ β_sp → Q β_sp Qᵀ — objective.

"""
    polar_RU(F) -> (R, U)

Right polar decomposition F = R·U with R orthogonal (det R = +1 for det F > 0)
and U symmetric positive-definite, via U = (FᵀF)^½ and R = F·U⁻¹. Allocation-free.
"""
@inline function polar_RU(F::SMatrix{3,3,Float64,9})
    C = Symmetric(F' * F)
    E = eigen(C)
    # FᵀF is SPD for nonsingular F, but a transient/degenerate Newton iterate can
    # yield a tiny negative eigenvalue numerically; floor at 0 before sqrt.
    sq = SVector{3,Float64}(sqrt(max(E.values[1], 0.0)),
                            sqrt(max(E.values[2], 0.0)),
                            sqrt(max(E.values[3], 0.0)))
    V = E.vectors
    U = V * SMatrix{3,3,Float64,9}(Diagonal(sq)) * V'
    inv_sq = SVector{3,Float64}(sq[1] > 0 ? 1 / sq[1] : 0.0,
                                sq[2] > 0 ? 1 / sq[2] : 0.0,
                                sq[3] > 0 ? 1 / sq[3] : 0.0)
    Uinv = V * SMatrix{3,3,Float64,9}(Diagonal(inv_sq)) * V'
    return F * Uinv, U
end

"""
    dR_polar(R, U, dF) -> SMatrix{3,3}

Directional derivative of the polar rotation R for a perturbation `dF` of F = R·U.
With Ω = RᵀṘ skew, the symmetric/skew split of RᵀdF gives the Sylvester equation
Ω U + U Ω = RᵀdF − dFᵀR; solving it in U's eigenbasis (componentwise division by
λ_i + λ_j) yields Ṙ = R Ω. Allocation-free.
"""
@inline function dR_polar(R::SMatrix{3,3,Float64,9}, U::SMatrix{3,3,Float64,9},
                          dF::SMatrix{3,3,Float64,9})
    A = R' * dF
    rhs = A - A'                         # = Ω U + U Ω
    E = eigen(Symmetric(U))
    V = E.vectors; λ = E.values
    rb = V' * rhs * V
    Ωb = SMatrix{3,3,Float64,9}(ntuple(Val(9)) do k
        i = (k - 1) % 3 + 1; j = (k - 1) ÷ 3 + 1
        rb[i, j] / (λ[i] + λ[j])
    end)
    return R * (V * Ωb * V')
end

"""
    dtau_dbeta(mat, εe_tr, β_sp, ᾱ_n) -> SMatrix{6,6}

Sensitivity ∂τ/∂β_sp of the Kirchhoff stress to the (spatial) back-stress at
fixed trial strain (FINITE_STRAIN §4.6). Needed for the consistent tangent once
β_sp depends on F through the polar rotation (objective kinematic hardening). In
the plastic branch the deviatoric corrector is c(ξ) = a·ξ with ξ = s_tr − β_sp,
a = 2G√(3/2)Δγ/‖ξ‖; since σ = σ_tr − c and ∂ξ/∂β = −I, ∂τ/∂β = ∂c/∂ξ:

    ∂τ/∂β = a·I₆ + ξ ⊗ (∂a/∂β),   ∂a/∂β_j = [2G√(3/2)/(3G+H)]·(Y/‖ξ‖²)·(Wξ)_j/‖ξ‖

with Y = σy0 + Hiso·ᾱ_n, H = Hiso + Hkin, W the engineering-shear metric. Zero in
the elastic branch. FD-verified to ~1e-10. Allocation-free.
"""
@inline function dtau_dbeta(mat::J2Material, εe_tr::SVector{6,Float64},
                            β_sp::SVector{6,Float64}, ᾱ_n::Float64)
    G = mat.G; H = mat.Hiso + mat.Hkin; s32 = sqrt(1.5)
    σ_tr = mat.Cmat * εe_tr
    p = (σ_tr[1] + σ_tr[2] + σ_tr[3]) / 3
    s = σ_tr - SVector{6,Float64}(p, p, p, 0, 0, 0)
    ξ = s - β_sp
    nrm = sqrt(ξ[1]^2 + ξ[2]^2 + ξ[3]^2 + 2 * (ξ[4]^2 + ξ[5]^2 + ξ[6]^2))
    Y = mat.σy0 + mat.Hiso * ᾱ_n
    f = s32 * nrm - Y
    f <= 0.0 && return zero(SMatrix{6,6,Float64,36})
    Δγ = f / (3G + H)
    a = 2G * s32 * Δγ / nrm
    da_dnrm = 2G * s32 / (3G + H) * (Y / nrm^2)
    g = (da_dnrm / nrm) * (W6 * ξ)        # ∂a/∂β (6-vector)
    diag6 = SMatrix{6,6,Float64,36}(Diagonal(SVector{6,Float64}(1, 1, 1, 1, 1, 1)))
    return a * diag6 + ξ * g'
end

# --- §3: constitutive update + exponential-map plastic update ---

"""
    finite_stress_update(mat, kin, F, εp_n, β_ref_n, ᾱ_n)
        -> (τ_voigt, εp_new, β_ref_new, ᾱ_new, D, τ_princ, Cp_inv_new, β_sp, R, U)

Feed the trial Hencky strain through the unchanged `return_map`, then perform the
exponential-map plastic update (FINITE_STRAIN §3). Returns:
- `τ_voigt`  — Kirchhoff stress, 6-Voigt physical shear (work-conjugate to εᵉ);
- `εp_new`, `β_ref_new`, `ᾱ_new` — updated log-space history; `β_ref` is the
  back-stress stored in the **reference (rotation-neutralized) configuration**;
- `D`        — the 6×6 algorithmic modulus ∂τ/∂εᵉ_tr from `return_map`;
- `τ_princ`  — the three principal Kirchhoff stresses τ_A (in the trial frame);
- `Cp_inv_new` — updated plastic state Cᵖ⁻¹_{n+1} = F⁻¹·bᵉ_{n+1}·F⁻ᵀ (6-Voigt);
- `β_sp`     — the spatial back-stress used in the return map (for the tangent);
- `R`        — the polar rotation of `F` (for the tangent's β-rotation term).

**Objectivity (FINITE_STRAIN §3.4).** With kinematic hardening the back-stress
must co-rotate with the material. We store β in the reference configuration
(`β_ref`, Q-invariant like Cᵖ⁻¹), push it forward with the polar rotation
β_sp = R·β_ref·Rᵀ, run the return map in the spatial trial frame, then pull the
updated back-stress back to reference. Under a superposed rotation Q (F → QF) the
trial strain rotates and R → QR, so β_sp → Q β_sp Qᵀ and τ → Q τ Qᵀ — exact frame
indifference (objectivity error ~1e-15 even at 120°). For isotropic/perfect
plasticity (Hkin = 0) β_ref ≡ 0 and this reduces to the symmetric path.

The plastic flow is coaxial with bᵉ_tr (associative J2), so the converged elastic
log strain shares the trial principal directions; bᵉ_{n+1} is reconstructed from
the corrected principal log strains εᵉ_A = εᵉ_tr_A − Δεᵖ_A. Allocation-free.
"""
@inline function finite_stress_update(mat::J2Material, kin::FiniteKin,
                                      F::SMatrix{3,3,Float64,9},
                                      εp_n::SVector{6,Float64},
                                      β_ref_n::SVector{6,Float64},
                                      ᾱ_n::Float64)
    n = kin.n   # eigenbasis (columns)

    # In the log-strain framework the plastic configuration is carried entirely by
    # Cᵖ⁻¹_n (via bᵉ_tr = F Cᵖ⁻¹_n Fᵀ), so εᵉ_tr = ½ ln bᵉ_tr is ALREADY the trial
    # *elastic* strain. Call `return_map` with a ZERO additive plastic strain
    # (passing εp_n would double-count plasticity). The back stress is supplied in
    # the SPATIAL frame β_sp = R β_ref Rᵀ (objectivity, §3.4); for Hkin = 0 this is
    # zero and the call is unchanged. The J2 kernel is rotation-covariant, so it
    # returns the spatial Kirchhoff stress τ and the algorithmic modulus
    # D = ∂τ/∂εᵉ_tr; the two-point tangent (§4.5/§4.6) consumes D plus the
    # ∂β_sp/∂F coupling.
    # The back-stress co-rotation is only needed for kinematic hardening; for
    # Hkin = 0 (isotropic / perfect) β_ref ≡ 0 ⇒ β_sp = 0 for any R, so skip the
    # polar decomposition entirely (cheaper, and avoids it on a degenerate trial F).
    Z6 = zero(SVector{6,Float64})
    if mat.Hkin > 0
        R, U = polar_RU(F)
        β_sp = sym3_to_voigt(R * voigt_to_sym3(β_ref_n) * R')
    else
        R = I3; U = I3; β_sp = Z6
    end
    τ_voigt, εp_new_inc, β_sp_new, ᾱ_new, D = return_map(mat, kin.εe_tr, Z6, β_sp, ᾱ_n)
    # pull the updated back-stress back to the reference configuration
    β_ref_new = mat.Hkin > 0 ? sym3_to_voigt(R' * voigt_to_sym3(β_sp_new) * R) : β_sp_new

    # principal Kirchhoff stresses (for diagnostics / the spatial-form fallback)
    τmat = voigt_to_sym3(τ_voigt)
    τ_princ = SVector{3,Float64}(dot(n[:, 1], τmat * n[:, 1]),
                                 dot(n[:, 2], τmat * n[:, 2]),
                                 dot(n[:, 3], τmat * n[:, 3]))

    # accumulated plastic strain (diagnostic); authoritative state is Cᵖ⁻¹.
    εp_new = εp_n + εp_new_inc

    # Corrected elastic log-strain tensor εᵉ_{n+1} = εᵉ_tr − Δεᵖ (global frame).
    # Δεᵖ (engineering-shear Voigt) = εp_new_inc (return map started from zero).
    # exp-map to bᵉ_{n+1} = exp(2 εᵉ_{n+1}); det Fᵖ preserved since tr Δεᵖ = 0.
    εe_v = kin.εe_tr - εp_new_inc
    εe_tens = SMatrix{3,3,Float64,9}(εe_v[1], 0.5εe_v[4], 0.5εe_v[6],
                                     0.5εe_v[4], εe_v[2], 0.5εe_v[5],
                                     0.5εe_v[6], 0.5εe_v[5], εe_v[3])
    EE = eigen(Symmetric(2εe_tens))
    be_new = exp(EE.values[1]) * (EE.vectors[:, 1] * EE.vectors[:, 1]') +
             exp(EE.values[2]) * (EE.vectors[:, 2] * EE.vectors[:, 2]') +
             exp(EE.values[3]) * (EE.vectors[:, 3] * EE.vectors[:, 3]')
    Cp_inv_mat = kin.Finv * be_new * kin.Finv'
    Cp_inv_new = sym3_to_voigt(Cp_inv_mat)

    return τ_voigt, εp_new, β_ref_new, ᾱ_new, D, τ_princ, Cp_inv_new, β_sp, R, U
end

# --- §4.3: spatial material modulus a (principal-axis form) ---

"""
    spatial_modulus(kin, τ_princ, D) -> SMatrix{6,6}

Spatial algorithmic elasticity tensor `a` (6×6 Voigt, engineering-shear strain /
physical-shear stress convention so `Bᵀ a B` is correct) from the principal
Kirchhoff stresses `τ_princ`, the trial eigenvalues `kin.b`, and the return-map
modulus `D` (FINITE_STRAIN §4.3, Simo & Hughes Box 8.2 / de Souza Neto Box 14.3):

    a = Σ_A Σ_B (D_AB − 2 τ_A δ_AB)(m_A⊗m_B)
      + Σ_A Σ_{B≠A} g_AB (m_AB⊗m_AB + m_AB⊗m_BA)

with g_AB = (τ_A b_B − τ_B b_A)/(b_A − b_B), degenerate limit
g_AB → ½(D_BB − D_AB) − τ_A. Here `D` must be the **principal-frame** algorithmic
modulus (i.e. evaluated with a diagonal trial log strain), whose upper-left 3×3
is the principal block D_AB = ∂τ_A/∂εᵉ_tr_B. Allocation-free.

This is the reference spatial-form material modulus (Box 8.2 / Box 14.3). It is
exact for an isotropic (coaxial) response and, combined with the §4.4 geometric
stiffness, reproduces the element tangent in that case. The production element
kernel instead uses the equivalent two-point P–F form (`dPdF`, §4.5), which is
also valid for the non-coaxial corrector of kinematic hardening; that form is the
one FD-verified by the master gate F2.
"""
@inline function spatial_modulus(kin::FiniteKin, τ_princ::SVector{3,Float64},
                                 D::SMatrix{6,6,Float64,36})
    n = kin.n
    b = kin.b
    # principal block D_AB = ∂τ_A/∂εᵉ_tr_B (upper-left 3×3 of the v1 modulus)
    Dp = SMatrix{3,3,Float64,9}(D[1, 1], D[2, 1], D[3, 1],
                                D[1, 2], D[2, 2], D[3, 2],
                                D[1, 3], D[2, 3], D[3, 3])
    n1 = n[:, 1]; n2 = n[:, 2]; n3 = n[:, 3]
    nA = (n1, n2, n3)

    # m_A = n_A⊗n_A as symmetric 3×3 tensors (the coaxial dyads).
    m1 = n1 * n1'; m2 = n2 * n2'; m3 = n3 * n3'
    ms = (m1, m2, m3)

    # coaxial coefficient c_AB = D_AB − 2 τ_A δ_AB
    c = SMatrix{3,3,Float64,9}(ntuple(9) do k
        A = (k - 1) % 3 + 1; Bidx = (k - 1) ÷ 3 + 1
        Dp[A, Bidx] - (A == Bidx ? 2τ_princ[A] : 0.0)
    end)

    # coupling coefficients g_AB (off-diagonal). g_AB and g_BA are evaluated
    # independently (each ordered pair contributes its own term).
    g12 = _gAB(τ_princ[1], τ_princ[2], b[1], b[2], Dp[1, 1], Dp[1, 2])
    g21 = _gAB(τ_princ[2], τ_princ[1], b[2], b[1], Dp[2, 2], Dp[2, 1])
    g13 = _gAB(τ_princ[1], τ_princ[3], b[1], b[3], Dp[1, 1], Dp[1, 3])
    g31 = _gAB(τ_princ[3], τ_princ[1], b[3], b[1], Dp[3, 3], Dp[3, 1])
    g23 = _gAB(τ_princ[2], τ_princ[3], b[2], b[3], Dp[2, 2], Dp[2, 3])
    g32 = _gAB(τ_princ[3], τ_princ[2], b[3], b[2], Dp[3, 3], Dp[3, 2])
    # non-symmetric mixed dyads m_AB = n_A⊗n_B
    m12 = n1 * n2'; m21 = n2 * n1'
    m13 = n1 * n3'; m31 = n3 * n1'
    m23 = n2 * n3'; m32 = n3 * n2'

    # Assemble the full 4th-order spatial tensor a_ijkl, then read off the Voigt
    # image (FINITE_STRAIN §4.3). Each Voigt row I and col J maps to tensor pairs
    # (i,j),(k,l); the engineering-shear convention is automatic because with the
    # v1 bmatrix the shear strain entered as γ=2ε and the symmetric a_ijkl already
    # contracts a_ijkl ε_kl = a_ij12 ε_12 + a_ij21 ε_21 = 2 a_ij12 ε_12, so the
    # Voigt entry equals a_ijkl directly (FD-verified to <1e-8):
    #   a_ijkl = Σ_A Σ_B c_AB (m_A)_ij (m_B)_kl
    #          + Σ_A Σ_{B≠A} g_AB [ (m_AB)_ij (m_AB)_kl + (m_AB)_ij (m_BA)_kl ]
    a66 = SMatrix{6,6,Float64,36}(ntuple(36) do kidx
        I = (kidx - 1) % 6 + 1
        Jdx = (kidx - 1) ÷ 6 + 1
        i, j = _VPAIR[I]
        k, l = _VPAIR[Jdx]
        v = 0.0
        for A in 1:3, Bidx in 1:3
            v += c[A, Bidx] * ms[A][i, j] * ms[Bidx][k, l]
        end
        # ordered-pair coupling: each (A,B) with B≠A contributes
        # g_AB(m_AB⊗m_AB + m_AB⊗m_BA). Sum all six ordered pairs.
        v += g12 * (m12[i, j] * m12[k, l] + m12[i, j] * m21[k, l])
        v += g21 * (m21[i, j] * m21[k, l] + m21[i, j] * m12[k, l])
        v += g13 * (m13[i, j] * m13[k, l] + m13[i, j] * m31[k, l])
        v += g31 * (m31[i, j] * m31[k, l] + m31[i, j] * m13[k, l])
        v += g23 * (m23[i, j] * m23[k, l] + m23[i, j] * m32[k, l])
        v += g32 * (m32[i, j] * m32[k, l] + m32[i, j] * m23[k, l])
        v
    end)
    return a66
end

# Voigt index → tensor (i,j) pair (physical, symmetric): [xx,yy,zz,xy,yz,zx].
const _VPAIR = ((1, 1), (2, 2), (3, 3), (1, 2), (2, 3), (1, 3))

# eigenvalue-coupling coefficient with the degenerate limit (FINITE_STRAIN §4.3).
@inline function _gAB(τA::Float64, τB::Float64, bA::Float64, bB::Float64,
                      DBB::Float64, DAB::Float64)
    if abs(bA - bB) < EIG_TOL * (abs(bA) + abs(bB) + 1.0)
        # degenerate limit g_AB → ½(D_BB − D_AB) − τ_A
        return 0.5 * (DBB - DAB) - τA
    else
        return (τA * bB - τB * bA) / (bA - bB)
    end
end

# --- two-point (P–F) consistent tangent (FINITE_STRAIN §4.5) ---
#
# The element tangent uses the first Piola–Kirchhoff form K = ∫ Gᵀ A G dV with
# A = ∂P/∂F (9×9), which automatically contains BOTH the material and geometric
# (initial-stress) contributions — no coaxiality assumption, valid for the
# non-coaxial corrector that combined iso+kin hardening produces. A is FD-verified
# against P(F) and the assembled Kᵉ against fᵉ (master gate F2).

# Contract the 4th-order derivative of Y = ½ ln(be) with a symmetric tensor H:
# (dY/dbe : H). In be's eigenbasis (Q, eigenvalues b_A): [dY:H]_AB = γ_AB H_AB
# with γ_AA = 1/(2 b_A) and γ_AB = (½ln b_A − ½ln b_B)/(b_A − b_B) (A≠B), and the
# degenerate limit γ_AB → 1/(2 b_A). Allocation-free.
@inline function _dhalflog_contract(Q::SMatrix{3,3,Float64,9}, b::SVector{3,Float64},
                                    H::SMatrix{3,3,Float64,9})
    Hp = Q' * H * Q                      # H in eigenbasis
    ly = SVector{3,Float64}(0.5log(b[1]), 0.5log(b[2]), 0.5log(b[3]))
    γ = SMatrix{3,3,Float64,9}(ntuple(9) do k
        A = (k - 1) % 3 + 1; Bi = (k - 1) ÷ 3 + 1
        if A == Bi
            1.0 / (2b[A])
        elseif abs(b[A] - b[Bi]) < EIG_TOL * (abs(b[A]) + abs(b[Bi]) + 1.0)
            1.0 / (b[A] + b[Bi])         # limit of (lyA-lyB)/(bA-bB)
        else
            (ly[A] - ly[Bi]) / (b[A] - b[Bi])
        end
    end)
    Yp = SMatrix{3,3,Float64,9}(ntuple(9) do k
        A = (k - 1) % 3 + 1; Bi = (k - 1) ÷ 3 + 1
        γ[A, Bi] * Hp[A, Bi]
    end)
    return Q * Yp * Q'                   # back to the global frame
end

# strain-Voigt (engineering shear) of a symmetric 3×3 tensor
@inline _eng_voigt(S::SMatrix{3,3,Float64,9}) =
    SVector{6,Float64}(S[1, 1], S[2, 2], S[3, 3], 2S[1, 2], 2S[2, 3], 2S[1, 3])

"""
    dPdF(kin, Cp_inv_n, D, τ_voigt, F; β_ref, R, U, dtdb) -> SMatrix{9,9}

First Piola–Kirchhoff tangent A = ∂P/∂F (FINITE_STRAIN §4.5/§4.6), P = τ·F⁻ᵀ.
Built by analytic differentiation: ∂be/∂F (be = F·Cᵖ⁻¹·Fᵀ), ∂εᵉ/∂F via the
log-derivative contraction, ∂τ/∂F = D : ∂εᵉ/∂F, and the product rule on τ·F⁻ᵀ.
The 9-vector layout is column-major F: index q = (col-1)*3 + row.

When kinematic hardening is active the spatial back-stress β_sp = R·β_ref·Rᵀ also
depends on F through the polar rotation R, contributing the additional term
∂τ/∂β_sp · ∂β_sp/∂F (with ∂β_sp/∂F from the polar-rotation derivative `dR_polar`).
This makes the tangent NON-SYMMETRIC but keeps it CONSISTENT (FD-verified <1e-6).
For Hkin = 0, `dtdb` (∂τ/∂β) is zero and this term vanishes (symmetric path).
Allocation-free.
"""
@inline function dPdF(kin::FiniteKin, Cp_inv_n::SVector{6,Float64},
                      D::SMatrix{6,6,Float64,36}, τ_voigt::SVector{6,Float64},
                      F::SMatrix{3,3,Float64,9};
                      β_ref::SVector{6,Float64}=zero(SVector{6,Float64}),
                      R::SMatrix{3,3,Float64,9}=I3,
                      U::SMatrix{3,3,Float64,9}=I3,
                      dtdb::SMatrix{6,6,Float64,36}=zero(SMatrix{6,6,Float64,36}))
    Cpi = voigt_to_sym3(Cp_inv_n)
    Finv = kin.Finv
    FinvT = Finv'
    τ = voigt_to_sym3(τ_voigt)
    b = kin.b; Q = kin.n
    βref_mat = voigt_to_sym3(β_ref)
    # Whether the back-stress-rotation term is active (kinematic hardening).
    kin_active = !iszero(β_ref) || !iszero(dtdb)

    # For each F-component (p,q): ∂F = e_p⊗e_q. Build column of A (9 stress comps
    # of ∂P) stacked column-major.
    cols = ntuple(Val(9)) do col
        p = (col - 1) % 3 + 1
        q = (col - 1) ÷ 3 + 1
        dF = _unit3(p, q)
        # ∂be = dF·Cpi·Fᵀ + F·Cpi·dFᵀ  (symmetric)
        dbe = dF * Cpi * F' + F * Cpi * dF'
        # ∂εᵉ = (d½ln be):∂be
        dεe = _dhalflog_contract(Q, b, dbe)
        # ∂τ = D : ∂εᵉ   (engineering Voigt)
        dτv = D * _eng_voigt(dεe)
        # back-stress-rotation coupling: ∂τ += ∂τ/∂β_sp · ∂β_sp/∂F (kinematic only)
        if kin_active
            dR = dR_polar(R, U, dF)
            dβsp = dR * βref_mat * R' + R * βref_mat * dR'
            dτv = dτv + dtdb * sym3_to_voigt(dβsp)
        end
        dτ = voigt_to_sym3(dτv)
        # P = τ·F⁻ᵀ ⇒ ∂P = ∂τ·F⁻ᵀ + τ·∂(F⁻ᵀ);  ∂(F⁻ᵀ) = −F⁻ᵀ·∂Fᵀ·F⁻ᵀ
        dFinvT = -FinvT * dF' * FinvT
        dP = dτ * FinvT + τ * dFinvT
        # stack dP column-major (row r, col c) → index (c-1)*3 + r
        SVector{9,Float64}(dP[1, 1], dP[2, 1], dP[3, 1],
                           dP[1, 2], dP[2, 2], dP[3, 2],
                           dP[1, 3], dP[2, 3], dP[3, 3])
    end
    return hcat(cols...)
end

@inline _unit3(p::Int, q::Int) = SMatrix{3,3,Float64,9}(ntuple(Val(9)) do k
    r = (k - 1) % 3 + 1; c = (k - 1) ÷ 3 + 1
    (r == p && c == q) ? 1.0 : 0.0
end)

"""
    first_piola(τ_voigt, Finv) -> SMatrix{3,3}

First Piola–Kirchhoff stress P = τ·F⁻ᵀ from the Kirchhoff stress (6-Voigt) and
F⁻¹ (so F⁻ᵀ = Finv'). Allocation-free.
"""
@inline first_piola(τ_voigt::SVector{6,Float64}, Finv::SMatrix{3,3,Float64,9}) =
    voigt_to_sym3(τ_voigt) * Finv'

"""
    det_Fp_from_Cpinv(Cp_inv) -> Float64

det Fᵖ from the stored Cᵖ⁻¹ = Fᵖ⁻¹Fᵖ⁻ᵀ: det Cᵖ⁻¹ = (det Fᵖ)⁻² ⇒
det Fᵖ = (det Cᵖ⁻¹)^(−1/2). Used by the F3 incompressibility test.
"""
@inline function det_Fp_from_Cpinv(Cp_inv::SVector{6,Float64})
    M = voigt_to_sym3(Cp_inv)
    return 1.0 / sqrt(det(M))
end

end # module
