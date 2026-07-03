"""
    Materials

J2 / von Mises rate-independent plasticity with combined isotropic + linear
kinematic hardening. The isotropic law may be **linear** (`σy = σy0 + Hiso·ᾱ`,
closed-form return) or **nonlinear saturation (Voce)**
(`σy = σy0 + (σ∞−σy0)(1−e^{−δᾱ}) + Hiso·ᾱ`, local-Newton return) — see
`return_map`. Pure, allocation-free constitutive kernel operating on
StaticArrays. See DESIGN.md §2.
"""
module Materials

using StaticArrays
using LinearAlgebra

export J2Material, return_map, elastic_matrix, yield_stress, yield_slope

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
    J2Material(; E, ν, σy0, Hiso=0.0, Hkin=0.0, σsat=σy0, δ=0.0)

Linear-elastic / J2-plastic material. Derived moduli G, K, λ and the elastic
Voigt matrix `Cmat` are precomputed for the hot loop (DESIGN §4.1).

**Isotropic hardening law** (the yield stress as a function of the equivalent
plastic strain ᾱ):

    σy(ᾱ) = σy0 + (σsat − σy0)(1 − e^{−δ·ᾱ}) + Hiso·ᾱ

- Default (`σsat = σy0` or `δ = 0`): the saturation term vanishes and this is
  the original **linear** law `σy = σy0 + Hiso·ᾱ` (closed-form return, bit-for-bit
  unchanged from before this field existed).
- With `σsat > σy0` and `δ > 0`: **nonlinear saturation (Voce) hardening** — the
  yield rises from `σy0` toward a saturation stress `σsat` at rate `δ`, plus a
  linear tail `Hiso·ᾱ`. This is the classic Simo (1988) round-bar-necking law.
  The return map then requires a scalar local Newton iteration on Δγ
  (`return_map`), since the consistency condition is nonlinear in Δγ.

`σsat` is the saturation stress (σ∞ in the literature), `δ` the saturation
exponent (dimensionless, per unit equivalent plastic strain).

**Limitation:** saturation hardening is currently supported for *isotropic*
hardening only (`Hkin = 0`). Combining it with kinematic hardening (`Hkin > 0`)
is rejected at construction — the *stress* update handles the combination, but
the finite-strain kinematic-hardening consistent-tangent term (`dtau_dbeta`)
still assumes a linear isotropic yield, so a combined saturation+kinematic FEM
solve would use an inconsistent tangent (slower Newton). It is blocked rather
than silently used.
"""
struct J2Material
    E::Float64
    ν::Float64
    σy0::Float64
    Hiso::Float64
    Hkin::Float64
    σsat::Float64
    δ::Float64
    G::Float64
    K::Float64
    λ::Float64
    Cmat::SMatrix{6,6,Float64,36}
end

function J2Material(; E::Real, ν::Real, σy0::Real, Hiso::Real=0.0, Hkin::Real=0.0,
                    σsat::Real=σy0, δ::Real=0.0)
    E = Float64(E); ν = Float64(ν); σy0 = Float64(σy0)
    Hiso = Float64(Hiso); Hkin = Float64(Hkin)
    σsat = Float64(σsat); δ = Float64(δ)
    # Physical-validity guards (DESIGN §4.1): a non-positive σy0 makes the
    # zero-deviator yield normal n̂ = 0/0 = NaN propagate; negative hardening or
    # ν ≥ 0.5 (incompressible) break the moduli.
    E > 0 || throw(ArgumentError("E must be positive (got $E)"))
    -1 < ν < 0.5 || throw(ArgumentError("ν must satisfy −1 < ν < 0.5 (got $ν)"))
    σy0 > 0 || throw(ArgumentError("σy0 must be positive (got $σy0)"))
    Hiso ≥ 0 || throw(ArgumentError("Hiso must be ≥ 0 (got $Hiso)"))
    Hkin ≥ 0 || throw(ArgumentError("Hkin must be ≥ 0 (got $Hkin)"))
    # Saturation-hardening guards: σsat ≥ σy0 (yield rises toward saturation, not
    # below the initial yield) and δ ≥ 0 (a decaying, not growing, exponential).
    σsat ≥ σy0 || throw(ArgumentError("σsat must be ≥ σy0 (got σsat=$σsat, σy0=$σy0)"))
    δ ≥ 0 || throw(ArgumentError("δ must be ≥ 0 (got $δ)"))
    # Saturation + kinematic hardening is not yet supported (see the docstring).
    if δ > 0 && σsat > σy0 && Hkin > 0
        throw(ArgumentError("saturation hardening (σsat>σy0, δ>0) with kinematic " *
                            "hardening (Hkin>0) is not yet supported; use one or the other"))
    end
    G = E / (2 * (1 + ν))
    K = E / (3 * (1 - 2ν))
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    return J2Material(E, ν, σy0, Hiso, Hkin, σsat, δ, G, K, λ, elastic_matrix(λ, G))
end

"""
    yield_stress(mat, ᾱ) -> Float64

Isotropic yield stress σy(ᾱ) = σy0 + (σsat−σy0)(1−e^{−δᾱ}) + Hiso·ᾱ (see
`J2Material`). Reduces to the linear `σy0 + Hiso·ᾱ` when `σsat == σy0` or `δ == 0`
(the saturation term is then exactly 0).
"""
@inline yield_stress(mat::J2Material, ᾱ::Float64) =
    mat.σy0 + (mat.σsat - mat.σy0) * (1 - exp(-mat.δ * ᾱ)) + mat.Hiso * ᾱ

"""
    yield_slope(mat, ᾱ) -> Float64

Isotropic hardening modulus H'(ᾱ) = dσy/dᾱ = δ(σsat−σy0)e^{−δᾱ} + Hiso. Reduces
to the constant `Hiso` in the linear case (used in the consistent tangent).
"""
@inline yield_slope(mat::J2Material, ᾱ::Float64) =
    mat.δ * (mat.σsat - mat.σy0) * exp(-mat.δ * ᾱ) + mat.Hiso

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

Radial-return mapping for J2 plasticity with combined isotropic (linear or
saturation/Voce) + linear kinematic hardening (DESIGN §2.3–2.5). Pure &
allocation-free: all quantities are StaticArrays.

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

    f_tr = q_tr - yield_stress(mat, ᾱ_n)       # yield function (DESIGN §2.4)

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
    s32 = sqrt(1.5)
    n̂ = ξ_tr / nrm                             # unit deviatoric normal (physical shears)

    # Plastic multiplier Δγ from the consistency condition
    #     r(Δγ) = q_tr − (3G + Hkin)·Δγ − σy(ᾱ_n + Δγ) = 0,
    # since q_new = q_tr − 3G·Δγ (elastic relaxation) and the back-stress growth
    # contributes an extra −Hkin·Δγ, leaving the isotropic yield σy(ᾱ_n+Δγ).
    if mat.δ > 0 && mat.σsat > mat.σy0
        # NONLINEAR saturation hardening: r is nonlinear in Δγ (via the Voce
        # exponential) ⇒ scalar local Newton. r is monotone-decreasing and convex
        # (σy is concave), so Newton from the linear-tangent guess converges
        # quadratically and monotonically. Scalars only ⇒ allocation-free.
        Δγ = f_tr / (3G + Hkin + yield_slope(mat, ᾱ_n))     # linearized initial guess
        @inbounds for _ in 1:50
            Hp = yield_slope(mat, ᾱ_n + Δγ)
            r = q_tr - (3G + Hkin) * Δγ - yield_stress(mat, ᾱ_n + Δγ)
            dΔγ = r / (3G + Hkin + Hp)          # −r/r′,  r′ = −(3G+Hkin+Hp)
            Δγ += dΔγ
            abs(dΔγ) <= 1e-13 * (Δγ + 1e-14) && break
        end
    else
        # LINEAR hardening: r is linear in Δγ ⇒ the exact closed form. This branch
        # is bit-for-bit identical to the original linear-only kernel. Verified to
        # reproduce the T2 closed form σ = σy0 + (E·Hiso/(E+Hiso))(ε − σy0/E) exactly.
        Δγ = f_tr / (3G + Hiso + Hkin)
    end

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
    #   β0 = 2G·Δγ/q_trial,   γ0 = 2G/(3G + Hkin + H′),   H′ = dσy/dᾱ at ᾱ_{n+1}
    #   D = ℂ − 3G·β0·I_dev + 3G·(β0 − γ0)·(n̂⊗n̂)
    # For linear hardening H′ ≡ Hiso and this is the original tangent verbatim; for
    # saturation hardening H′ is the (state-dependent) tangent modulus at ᾱ_{n+1}.
    Hprime = yield_slope(mat, ᾱ_new)
    β0 = 2G * Δγ / q_tr
    γ0 = 2G / (3G + Hkin + Hprime)
    D_alg = mat.Cmat - (3G * β0) * IDEV + (3G * (β0 - γ0)) * (n̂ * n̂')

    return (σ_new, εp_new, β_new, ᾱ_new, D_alg)
end

end # module
