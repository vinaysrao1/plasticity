"""
    Materials

J2 / von Mises rate-independent plasticity with combined linear isotropic +
linear kinematic hardening. Pure, allocation-free constitutive kernel
(`return_map`) operating on StaticArrays. See DESIGN.md §2.
"""
module Materials

using StaticArrays
using LinearAlgebra

export J2Material, return_map, elastic_matrix

# Voigt ordering everywhere: [xx, yy, zz, xy, yz, zx]  (DESIGN §2.1, §9)
# Strain Voigt uses engineering shear (γ = 2ε); stress Voigt uses physical shear.

"""
    elastic_matrix(λ, G) -> SMatrix{6,6}

Isotropic elasticity matrix ℂ in Voigt form (DESIGN §2.2). Shear-diagonal
entries are `G` (not 2G) because engineering shear carries the factor 2.
"""
function elastic_matrix(λ::Float64, G::Float64)
    a = λ + 2G
    return @SMatrix [a   λ   λ   0.0 0.0 0.0;
                     λ   a   λ   0.0 0.0 0.0;
                     λ   λ   a   0.0 0.0 0.0;
                     0.0 0.0 0.0 G   0.0 0.0;
                     0.0 0.0 0.0 0.0 G   0.0;
                     0.0 0.0 0.0 0.0 0.0 G]
end

"""
    J2Material(; E, ν, σy0, Hiso=0.0, Hkin=0.0)

Linear-elastic / J2-plastic material. Derived moduli G, K, λ and the elastic
Voigt matrix `Cmat` are precomputed for the hot loop (DESIGN §4.1).
"""
struct J2Material
    E::Float64
    ν::Float64
    σy0::Float64
    Hiso::Float64
    Hkin::Float64
    G::Float64
    K::Float64
    λ::Float64
    Cmat::SMatrix{6,6,Float64,36}
end

function J2Material(; E::Real, ν::Real, σy0::Real, Hiso::Real=0.0, Hkin::Real=0.0)
    E = Float64(E); ν = Float64(ν); σy0 = Float64(σy0)
    Hiso = Float64(Hiso); Hkin = Float64(Hkin)
    # Physical-validity guards (DESIGN §4.1): a non-positive σy0 makes the
    # zero-deviator yield normal n̂ = 0/0 = NaN propagate; negative hardening or
    # ν ≥ 0.5 (incompressible) break the moduli.
    E > 0 || throw(ArgumentError("E must be positive (got $E)"))
    -1 < ν < 0.5 || throw(ArgumentError("ν must satisfy −1 < ν < 0.5 (got $ν)"))
    σy0 > 0 || throw(ArgumentError("σy0 must be positive (got $σy0)"))
    Hiso ≥ 0 || throw(ArgumentError("Hiso must be ≥ 0 (got $Hiso)"))
    Hkin ≥ 0 || throw(ArgumentError("Hkin must be ≥ 0 (got $Hkin)"))
    G = E / (2 * (1 + ν))
    K = E / (3 * (1 - 2ν))
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    return J2Material(E, ν, σy0, Hiso, Hkin, G, K, λ, elastic_matrix(λ, G))
end

# Volumetric structure 𝟙 = [1,1,1,0,0,0]ᵀ (DESIGN §2.5)
const ONE6 = SVector{6,Float64}(1, 1, 1, 0, 0, 0)
# 𝟙 ⊗ 𝟙 : upper-left 3×3 block of ones (DESIGN §2.5)
const ONExONE = SMatrix{6,6,Float64,36}(ONE6 * ONE6')
# Symmetric 4th-order identity in Voigt: diag(1,1,1,½,½,½) (DESIGN §2.5)
const ISYM = SMatrix{6,6,Float64,36}(Diagonal(SVector{6,Float64}(1, 1, 1, 0.5, 0.5, 0.5)))
# Deviatoric symmetric identity I_dev = I_sym − (1/3) 𝟙⊗𝟙
const IDEV = ISYM - ONExONE / 3

"""
    devnorm(s) -> Float64

Frobenius norm of a symmetric deviator written in Voigt with *physical* shear
components: ‖s‖ = sqrt(s_xx²+s_yy²+s_zz² + 2(s_xy²+s_yz²+s_zx²))  (DESIGN §2.3).
"""
@inline function devnorm(s::SVector{6,Float64})
    return sqrt(s[1]^2 + s[2]^2 + s[3]^2 + 2 * (s[4]^2 + s[5]^2 + s[6]^2))
end

"""
    return_map(mat, ε, εp_n, β_n, ᾱ_n)
        -> (σ_new, εp_new, β_new, ᾱ_new, D_alg)

Radial-return mapping for J2 plasticity with combined linear hardening
(DESIGN §2.3–2.5). Pure & allocation-free: all quantities are StaticArrays.

Arguments are the current total strain `ε` (engineering-shear Voigt) and the
committed history `{εp_n, β_n, ᾱ_n}` of the previous converged load step.
"""
@inline function return_map(mat::J2Material,
                            ε::SVector{6,Float64},
                            εp_n::SVector{6,Float64},
                            β_n::SVector{6,Float64},
                            ᾱ_n::Float64)
    G = mat.G; K = mat.K
    Hiso = mat.Hiso; Hkin = mat.Hkin

    # --- trial elastic predictor (DESIGN §2.3) ---
    σ_tr = mat.Cmat * (ε - εp_n)
    p_tr = (σ_tr[1] + σ_tr[2] + σ_tr[3]) / 3
    s_tr = σ_tr - p_tr * ONE6                  # deviatoric trial stress
    ξ_tr = s_tr - β_n                          # relative (shifted) stress; β deviatoric
    nrm = devnorm(ξ_tr)
    q_tr = sqrt(1.5) * nrm                     # von Mises effective relative stress

    f_tr = q_tr - (mat.σy0 + Hiso * ᾱ_n)       # yield function (DESIGN §2.4)

    if f_tr <= 0.0
        # elastic step: accept trial, tangent is elastic ℂ
        return (σ_tr, εp_n, β_n, ᾱ_n, mat.Cmat)
    end

    # --- plastic corrector ---
    # CORRECTION to DESIGN §2.4. As written, DESIGN uses Δεᵖ = Δγ·n̂ and labels
    # Δγ the *equivalent* plastic strain increment, but with the unit deviatoric
    # normal n̂ (tensor norm 1) that makes ᾱ = ‖Δεᵖ‖, which is NOT the equivalent
    # plastic strain. That convention over-hardens and fails the T2 analytical
    # target (it gives ᾱ = √(3/2)·εᵖ_axial under uniaxial loading instead of
    # ᾱ = εᵖ_axial). The physically correct, self-consistent J2 flow uses the
    # associative normal N = √(3/2)·n̂ (so ∂q/∂σ, with q = √(3/2)‖ξ‖):
    #
    #     Δεᵖ = Δγ·√(3/2)·n̂ (tensor),  σ = σ_tr − 2G·Δγ·√(3/2)·n̂,
    #     ᾱ_{n+1} = ᾱ_n + Δγ  (now the true equivalent plastic strain),
    #     β̇ = (2/3)H_kin·ε̇ᵖ  ⇒  Δβ = (2/3)H_kin·Δγ·√(3/2)·n̂.
    #
    # The plastic multiplier denominator 3G+H_iso+H_kin is unchanged because
    # q_new = √(3/2)‖ξ_tr − 2GΔγ√(3/2)n̂‖ = q_tr − 3G·Δγ. Verified to reproduce
    # the T2 closed form σ = σy0 + (E·Hiso/(E+Hiso))(ε − σy0/E) exactly.
    s32 = sqrt(1.5)
    n̂ = ξ_tr / nrm                             # unit deviatoric normal (physical shears)
    Δγ = f_tr / (3G + Hiso + Hkin)             # closed-form equivalent-plastic-strain increment

    # stress: deviatoric correction only (mean stress unchanged → incompressibility)
    σ_new = σ_tr - (2G * Δγ * s32) * n̂

    # plastic strain increment Δεᵖ = Δγ·√(3/2)·n̂ (tensor) in engineering-shear
    # Voigt (factor 2 on shears so the *tensorial* plastic strain matches N)
    Δεp = (Δγ * s32) * SVector{6,Float64}(n̂[1], n̂[2], n̂[3], 2n̂[4], 2n̂[5], 2n̂[6])
    εp_new = εp_n + Δεp

    # back-stress update (deviatoric, physical shears)
    β_new = β_n + (2 / 3) * Hkin * (Δγ * s32) * n̂

    ᾱ_new = ᾱ_n + Δγ

    # --- consistent (algorithmic) tangent ---
    # Exact ∂σ/∂ε for the Convention-B return above, derived by carrying the
    # engineering-shear metric W=diag(1,1,1,2,2,2) through ∂n̂/∂ε and using that
    # n̂ is deviatoric (n̂'W·I_dev = n̂'). Verified vs central finite differences
    # to ~1e-11 (DESIGN §2.5 FD check) and symmetric (associative J2).
    #
    #   β0 = 2G·Δγ/q_trial,   γ0 = 2G/(3G+Hiso+Hkin)
    #   D = ℂ − 3G·β0·I_dev + 3G·(β0 − γ0)·(n̂⊗n̂)
    β0 = 2G * Δγ / q_tr
    γ0 = 2G / (3G + Hiso + Hkin)
    D_alg = mat.Cmat - (3G * β0) * IDEV + (3G * (β0 - γ0)) * (n̂ * n̂')

    return (σ_new, εp_new, β_new, ᾱ_new, D_alg)
end

end # module
