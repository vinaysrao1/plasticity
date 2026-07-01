"""
    Elements

Hex8 trilinear hexahedron: shape functions, isoparametric Jacobian, B-matrix,
2×2×2 Gauss quadrature, geometry caching, and the allocation-free element
force/tangent kernel. See DESIGN.md §3.
"""
module Elements

using StaticArrays
using LinearAlgebra
using ..Materials: J2Material, return_map
using ..FiniteStrain: FiniteStrain, ElementKind, Hex8Small, Hex8Finite, Hex8FiniteFbar,
    deformation_gradient, finite_kinematics, finite_stress_update,
    voigt_to_sym3, sym3_to_voigt, dPdF, first_piola, dtau_dbeta, ElementInversionError

export hex8_shape, hex8_dshape, jacobian, bmatrix, precompute_cache,
       element_force_tangent!, element_force_tangent_finite!, geometric_stiffness,
       ElementCache, element_geometry, element_coords,
       element_ref_grads, element_centroid_J0,
       GAUSS_PTS, GAUSS_WTS, NODE_NAT

const I3const = SMatrix{3,3,Float64,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)

# Natural coordinates of the 8 nodes (DESIGN §3.1, VTK_HEXAHEDRON ordering)
const NODE_NAT = SMatrix{8,3,Float64,24}(
    # ξ                          η                          ζ
    -1, 1, 1, -1, -1, 1, 1, -1,
    -1, -1, 1, 1, -1, -1, 1, 1,
    -1, -1, -1, -1, 1, 1, 1, 1)

# 2×2×2 Gauss points & weights (DESIGN §3.5): coords ±1/√3, weight 1 each.
const _g = 1 / sqrt(3.0)
const GAUSS_PTS = SVector{8,SVector{3,Float64}}(
    SVector(-_g, -_g, -_g),
    SVector(_g, -_g, -_g),
    SVector(_g, _g, -_g),
    SVector(-_g, _g, -_g),
    SVector(-_g, -_g, _g),
    SVector(_g, -_g, _g),
    SVector(_g, _g, _g),
    SVector(-_g, _g, _g))
const GAUSS_WTS = SVector{8,Float64}(1, 1, 1, 1, 1, 1, 1, 1)

"""
    hex8_shape(ξ) -> SVector{8}

Trilinear shape functions N_a at natural coordinate ξ=(ξ,η,ζ) (DESIGN §3.2).
"""
@inline function hex8_shape(ξ::SVector{3,Float64})
    return SVector{8,Float64}(ntuple(a -> 0.125 *
        (1 + NODE_NAT[a, 1] * ξ[1]) *
        (1 + NODE_NAT[a, 2] * ξ[2]) *
        (1 + NODE_NAT[a, 3] * ξ[3]), 8))
end

"""
    hex8_dshape(ξ) -> SMatrix{8,3}

Shape-function derivatives ∂N_a/∂ξ_j w.r.t. natural coords (DESIGN §3.2).
"""
@inline function hex8_dshape(ξ::SVector{3,Float64})
    return SMatrix{8,3,Float64,24}(ntuple(8 * 3) do k
        a = (k - 1) % 8 + 1          # row (node)
        j = (k - 1) ÷ 8 + 1          # col (direction)
        ξa = NODE_NAT[a, 1]; ηa = NODE_NAT[a, 2]; ζa = NODE_NAT[a, 3]
        if j == 1
            0.125 * ξa * (1 + ηa * ξ[2]) * (1 + ζa * ξ[3])
        elseif j == 2
            0.125 * ηa * (1 + ξa * ξ[1]) * (1 + ζa * ξ[3])
        else
            0.125 * ζa * (1 + ξa * ξ[1]) * (1 + ηa * ξ[2])
        end
    end)
end

# ∂N/∂ξ at the element centroid (0,0,0) — a compile-time constant (for F-bar).
const DN_CENTROID = hex8_dshape(SVector{3,Float64}(0, 0, 0))

"""
    jacobian(Xe, dN) -> SMatrix{3,3}

Isoparametric Jacobian J_ij = ∂x_i/∂ξ_j = Σ_a X_a,i ∂N_a/∂ξ_j = Xeᵀ dN
(DESIGN §3.3). `Xe` is the 8×3 matrix of element node coordinates. The spatial
gradients are then `dN/dx = dN · J⁻¹`. (`detJ` is invariant to this transpose,
so axis-aligned box meshes — diagonal J — are unaffected; the transpose only
matters for sheared/distorted elements.)
"""
@inline jacobian(Xe::SMatrix{8,3,Float64,24}, dN::SMatrix{8,3,Float64,24}) = Xe' * dN

"""
    bmatrix(dNdx) -> SMatrix{6,24}

Strain-displacement matrix B (Voigt [xx,yy,zz,xy,yz,zx], engineering shear)
from spatial shape-function gradients dN/dx (8×3) (DESIGN §3.4).
"""
@inline function bmatrix(dNdx::SMatrix{8,3,Float64,24})
    return SMatrix{6,24,Float64,144}(ntuple(6 * 24) do k
        r = (k - 1) % 6 + 1          # Voigt row
        c = (k - 1) ÷ 6 + 1          # local DOF column (1..24)
        a = (c - 1) ÷ 3 + 1          # node
        comp = (c - 1) % 3 + 1       # component x/y/z
        Nx = dNdx[a, 1]; Ny = dNdx[a, 2]; Nz = dNdx[a, 3]
        # B_a block (6×3) per DESIGN §3.4
        if r == 1
            comp == 1 ? Nx : 0.0
        elseif r == 2
            comp == 2 ? Ny : 0.0
        elseif r == 3
            comp == 3 ? Nz : 0.0
        elseif r == 4          # xy
            comp == 1 ? Ny : (comp == 2 ? Nx : 0.0)
        elseif r == 5          # yz
            comp == 2 ? Nz : (comp == 3 ? Ny : 0.0)
        else                   # zx
            comp == 1 ? Nz : (comp == 3 ? Nx : 0.0)
        end
    end)
end

"""
    element_geometry(Xe) -> (Bs, detJw)

Compute the per-Gauss-point B-matrices `Bs::SVector{8,SMatrix{6,24}}` and
`detJ·w` weights `detJw::SVector{8}` for an element with node coordinates
`Xe::SMatrix{8,3}`. Allocation-free; this is the v1 geometry math factored out so
it can be reused for both the cached reference element and the on-the-fly fallback
(SCALING.md §2.2).
"""
@inline function element_geometry(Xe::SMatrix{8,3,Float64,24})
    # `Val(8)` makes the tuple length a compile-time constant so `ntuple` fully
    # unrolls and the per-GP `geo` tuple stays on the stack — keeping this kernel
    # allocation-free (a plain `ntuple(_, 8)` here heap-allocates ~87 kB/call,
    # which would blow both the build-time uniformity scan and the non-uniform
    # assembly memory budget at scale).
    geo = ntuple(Val(8)) do g
        dN = hex8_dshape(GAUSS_PTS[g])
        J = jacobian(Xe, dN)
        (bmatrix(dN * inv(J)), det(J) * GAUSS_WTS[g])
    end
    Bs = SVector{8,SMatrix{6,24,Float64,144}}(ntuple(g -> geo[g][1], Val(8)))
    detJw = SVector{8,Float64}(ntuple(g -> geo[g][2], Val(8)))
    return Bs, detJw
end

"""
    element_ref_grads(Xe) -> SVector{8, SMatrix{8,3}}

Reference shape gradients ∂Nₐ/∂X per Gauss point (8×3 each) for an element with
*undeformed* node coordinates `Xe::SMatrix{8,3}` (FINITE_STRAIN §2.1, §6.3).
These equal the v1 spatial gradients of the reference element (`dN·J_ref⁻¹`) and
depend only on the undeformed mesh, so they share the v1 uniform-mesh caching.
Allocation-free.
"""
@inline function element_ref_grads(Xe::SMatrix{8,3,Float64,24})
    return SVector{8,SMatrix{8,3,Float64,24}}(ntuple(Val(8)) do g
        dN = hex8_dshape(GAUSS_PTS[g])
        J = jacobian(Xe, dN)
        dN * inv(J)
    end)
end

"""
    element_centroid_J0(Xe, ue) -> Float64

J₀ = det F₀ at the element centroid (natural coords (0,0,0)) for the F-bar
formulation (FINITE_STRAIN §5). `Xe` are reference node coords, `ue` the element
nodal displacements. Allocation-free.
"""
@inline function element_centroid_J0(Xe::SMatrix{8,3,Float64,24}, ue::SVector{24,Float64})
    dN = DN_CENTROID
    J = jacobian(Xe, dN)
    dNdX0 = dN * inv(J)
    # F₀ = I + Σₐ uₐ ⊗ ∂Nₐ/∂X|₀
    F0 = SMatrix{3,3,Float64,9}(1, 0, 0, 0, 1, 0, 0, 0, 1)
    @inbounds for a in 1:8
        ux = ue[3(a - 1) + 1]; uy = ue[3(a - 1) + 2]; uz = ue[3(a - 1) + 3]
        gx = dNdX0[a, 1]; gy = dNdX0[a, 2]; gz = dNdX0[a, 3]
        F0 += SMatrix{3,3,Float64,9}(ux * gx, uy * gx, uz * gx,
                                     ux * gy, uy * gy, uz * gy,
                                     ux * gz, uy * gz, uz * gz)
    end
    return det(F0)
end

# Element node coordinates as an 8×3 SMatrix (allocation-free).
@inline function element_coords(nodes::Matrix{Float64}, elements::AbstractMatrix{<:Integer}, e::Integer)
    return SMatrix{8,3,Float64,24}(ntuple(Val(24)) do k
        a = (k - 1) % 8 + 1
        j = (k - 1) ÷ 8 + 1
        nodes[j, elements[a, e]]
    end)
end

"""
    ElementCache

Element geometry source for the assembler (SCALING.md §2.2). For a **uniform box
mesh** every element is geometrically identical up to translation, so a *single*
reference set of B-matrices and detJ·w is cached (`uniform=true`) — replacing the
v1 per-element cache (≈31 GB at 10M) with ≈9 kB. For non-uniform meshes
(`uniform=false`) the cache stores only the node coordinates and recomputes the
geometry on the fly per element (zero extra memory; modest compute).

Fields:
- `uniform` — true if every element shares one reference geometry.
- `Bref`, `detJwref` — the single reference set (valid iff `uniform`).
- `nodes`, `elements` — kept for the on-the-fly recompute fallback.
"""
struct ElementCache
    uniform::Bool
    Bref::SVector{8,SMatrix{6,24,Float64,144}}
    detJwref::SVector{8,Float64}
    dNdXref::SVector{8,SMatrix{8,3,Float64,24}}   # reference shape grads ∂N/∂X (§6.3)
    nodes::Matrix{Float64}
    elements::Matrix{Int}
end

"""
    element_geometry(cache, e) -> (Bs, detJw)

Geometry for element `e`: the cached reference set if the mesh is uniform,
otherwise recomputed on the fly from node coordinates. Allocation-free — used in
the hot assembly loop (SCALING.md §2.2).
"""
@inline function element_geometry(cache::ElementCache, e::Integer)
    if cache.uniform
        return cache.Bref, cache.detJwref
    else
        Xe = element_coords(cache.nodes, cache.elements, e)
        return element_geometry(Xe)
    end
end

"""
    element_ref_grads(cache, e) -> SVector{8, SMatrix{8,3}}

Reference shape gradients ∂N/∂X for element `e`: the cached reference set if the
mesh is uniform, otherwise recomputed on the fly. Allocation-free (FINITE_STRAIN
§6.3).
"""
@inline function element_ref_grads(cache::ElementCache, e::Integer)
    if cache.uniform
        return cache.dNdXref
    else
        Xe = element_coords(cache.nodes, cache.elements, e)
        return element_ref_grads(Xe)
    end
end

# Detect whether every element is geometrically identical to element 1 *up to
# translation* — the exact condition under which one reference set of B-matrices
# and detJ·w is valid for all elements (SCALING.md §2.2). We compare each node's
# offset from the element's first node against element 1's offsets; if they all
# match, every element is a rigid translate of element 1 (same shape ⇒ same B,
# same detJ). This is alloc-free scalar work (no per-element B/inv(J)), so it does
# not blow the build-time memory budget at scale, and it is exact — it does not
# weaken to merely matching detJ.
function _is_uniform(nodes::Matrix{Float64}, elements::Matrix{Int})
    nelem = size(elements, 2)
    nelem <= 1 && return true
    @inbounds begin
        n11 = elements[1, 1]
        L = 0.0
        for i in 2:8, d in 1:3
            L = max(L, abs(nodes[d, elements[i, 1]] - nodes[d, n11]))
        end
        tol = 1e-9 * (L + 1.0)
        for e in 2:nelem
            ne1 = elements[1, e]
            for i in 2:8
                nei = elements[i, e]; ni1 = elements[i, 1]
                for d in 1:3
                    off_e = nodes[d, nei] - nodes[d, ne1]
                    off_1 = nodes[d, ni1] - nodes[d, n11]
                    abs(off_e - off_1) > tol && return false
                end
            end
        end
    end
    return true
end

"""
    precompute_cache(nodes, elements) -> ElementCache

Build the element-geometry cache (SCALING.md §2.2). Detects whether the mesh is a
uniform box: if so caches a single reference element; otherwise stores the mesh
for on-the-fly geometry recompute. `nodes` is 3×nnodes, `elements` is 8×nelem.
"""
function precompute_cache(nodes::Matrix{Float64}, elements::Matrix{Int})
    uniform = _is_uniform(nodes, elements)
    if uniform
        Xe = element_coords(nodes, elements, 1)
        Bref, detJwref = element_geometry(Xe)
        dNdXref = element_ref_grads(Xe)
        @assert all(>(0), detJwref) "non-positive detJ in reference element (check node ordering)"
        return ElementCache(true, Bref, detJwref, dNdXref, nodes, elements)
    else
        # placeholder reference set (unused when uniform=false)
        Bz = zero(SMatrix{6,24,Float64,144})
        Bref = SVector{8,SMatrix{6,24,Float64,144}}(ntuple(_ -> Bz, 8))
        detJwref = zero(SVector{8,Float64})
        Gz = zero(SMatrix{8,3,Float64,24})
        dNdXref = SVector{8,SMatrix{8,3,Float64,24}}(ntuple(_ -> Gz, 8))
        return ElementCache(false, Bref, detJwref, dNdXref, nodes, elements)
    end
end

"""
    element_force_tangent!(mat, Bs, detJw, ue, εp, β, ᾱ, e, σout, commit=Val(false))
        -> (Fe, Ke)

Element internal force (SVector{24}) and consistent tangent (SMatrix{24,24})
by looping the 8 Gauss points (DESIGN §3.6). Allocation-free.

`εp`, `β` (6×ngp), `ᾱ` (ngp), `σout` (6×ngp) are the working SoA state arrays;
`e` is the element index (gp global index = (e-1)*8 + g). When `commit=Val(true)`
the updated per-GP plastic state is written back; otherwise only stresses are
recorded (for assembly). `commit` is a `Val` (not a keyword) so the branch is
resolved at compile time and the kernel stays allocation-free.
"""
@inline function element_force_tangent!(mat::J2Material,
                                        Bs::SVector{8,SMatrix{6,24,Float64,144}},
                                        detJw::SVector{8,Float64},
                                        ue::SVector{24,Float64},
                                        εp::Matrix{Float64},
                                        β::Matrix{Float64},
                                        ᾱ::Vector{Float64},
                                        e::Int,
                                        σout::Matrix{Float64},
                                        ::Val{COMMIT}=Val(false)) where {COMMIT}
    Fe = zero(SVector{24,Float64})
    Ke = zero(SMatrix{24,24,Float64,576})
    @inbounds for g in 1:8
        B = Bs[g]
        w = detJw[g]
        idx = (e - 1) * 8 + g
        ε = B * ue
        εp_n = SVector{6,Float64}(εp[1, idx], εp[2, idx], εp[3, idx],
                                  εp[4, idx], εp[5, idx], εp[6, idx])
        β_n = SVector{6,Float64}(β[1, idx], β[2, idx], β[3, idx],
                                 β[4, idx], β[5, idx], β[6, idx])
        ᾱ_n = ᾱ[idx]
        σ, εp_new, β_new, ᾱ_new, D = return_map(mat, ε, εp_n, β_n, ᾱ_n)
        Fe += (B' * σ) * w
        Ke += (B' * (D * B)) * w
        # record stress for output / postprocessing
        σout[1, idx] = σ[1]; σout[2, idx] = σ[2]; σout[3, idx] = σ[3]
        σout[4, idx] = σ[4]; σout[5, idx] = σ[5]; σout[6, idx] = σ[6]
        if COMMIT
            εp[1, idx] = εp_new[1]; εp[2, idx] = εp_new[2]; εp[3, idx] = εp_new[3]
            εp[4, idx] = εp_new[4]; εp[5, idx] = εp_new[5]; εp[6, idx] = εp_new[6]
            β[1, idx] = β_new[1]; β[2, idx] = β_new[2]; β[3, idx] = β_new[3]
            β[4, idx] = β_new[4]; β[5, idx] = β_new[5]; β[6, idx] = β_new[6]
            ᾱ[idx] = ᾱ_new
        end
    end
    return Fe, Ke
end

# --- finite-strain element kernel (FINITE_STRAIN §2–5) ---

"""
    geometric_stiffness(dNdx, τmat, w) -> SMatrix{24,24}

Geometric / initial-stress stiffness (FINITE_STRAIN §4.4):
Kᵍ[(a,i),(b,k)] = δ_ik (∂Nₐ/∂x)ᵀ τ (∂N_b/∂x) · w, with `dNdx::SMatrix{8,3}` the
spatial gradients and `τmat` the 3×3 Kirchhoff stress. Adds the same scalar
`g_aᵀ τ g_b · w` to all three diagonal component-blocks of node-pair (a,b).
Allocation-free.
"""
@inline function geometric_stiffness(dNdx::SMatrix{8,3,Float64,24},
                                     τmat::SMatrix{3,3,Float64,9}, w::Float64)
    return SMatrix{24,24,Float64,576}(ntuple(24 * 24) do k
        r = (k - 1) % 24 + 1
        c = (k - 1) ÷ 24 + 1
        i = (r - 1) % 3 + 1     # component of row dof
        kk = (c - 1) % 3 + 1    # component of col dof
        if i != kk
            0.0
        else
            a = (r - 1) ÷ 3 + 1
            b = (c - 1) ÷ 3 + 1
            ga = SVector{3,Float64}(dNdx[a, 1], dNdx[a, 2], dNdx[a, 3])
            gb = SVector{3,Float64}(dNdx[b, 1], dNdx[b, 2], dNdx[b, 3])
            (dot(ga, τmat * gb)) * w
        end
    end)
end

"""
    element_force_tangent_finite!(kind, mat, dNdXs, detJw, ue, Xe, εp, β, ᾱ, Cp_inv,
                                  e, σout, commit=Val(false)) -> (Fe, Ke)

Finite-strain element internal force (SVector{24}) and consistent tangent
(SMatrix{24,24}) by looping the 8 Gauss points (FINITE_STRAIN §4). `kind` is the
zero-size `ElementKind` (`Hex8Finite` or `Hex8FiniteFbar`) selecting standard F
vs F-bar (§5). `dNdXs::SVector{8,SMatrix{8,3}}` are the *reference* shape
gradients ∂N/∂X, `Xe` the reference node coords (needed for the F-bar centroid
J₀). `Cp_inv` (6×ngp) is the plastic configuration. `σout` records **Kirchhoff**
stress (the Model converts to Cauchy on reporting). On `commit=Val(true)` the
updated plastic history (εp, β, ᾱ, Cp_inv) is written back. Allocation-free.
"""
@inline function element_force_tangent_finite!(kind::ElementKind,
                                               mat::J2Material,
                                               dNdXs::SVector{8,SMatrix{8,3,Float64,24}},
                                               detJw::SVector{8,Float64},
                                               ue::SVector{24,Float64},
                                               Xe::SMatrix{8,3,Float64,24},
                                               εp::Matrix{Float64},
                                               β::Matrix{Float64},
                                               ᾱ::Vector{Float64},
                                               Cp_inv::Matrix{Float64},
                                               e::Int,
                                               σout::Matrix{Float64},
                                               ::Val{COMMIT}=Val(false)) where {COMMIT}
    fbar = kind isa Hex8FiniteFbar
    # F-bar centroid quantities (FINITE_STRAIN §5): F₀ and J₀ at natural coords
    # (0,0,0), plus the centroid G-operator for the ∂J₀/∂uₑ coupling.
    dNdX0 = fbar ? _centroid_ref_grads(Xe) : zero(SMatrix{8,3,Float64,24})
    F0 = fbar ? deformation_gradient(ue, dNdX0) : I3const
    J0 = fbar ? det(F0) : 1.0
    G0 = fbar ? _Gmatrix(dNdX0) : zero(SMatrix{9,24,Float64,216})

    Fe = zero(SVector{24,Float64})
    Ke = zero(SMatrix{24,24,Float64,576})
    @inbounds for g in 1:8
        dNdX = dNdXs[g]
        w = detJw[g]
        idx = (e - 1) * 8 + g

        F = deformation_gradient(ue, dNdX)
        Jg = det(F)
        # F-bar: replace F by F̄ = (J₀/J)^{1/3} F (volumetric part from centroid).
        # `cbrt` handles a (transiently) negative ratio gracefully during Newton.
        scale = fbar ? cbrt(J0 / Jg) : 1.0
        Fbar = fbar ? scale * F : F

        Cpi_n = SVector{6,Float64}(Cp_inv[1, idx], Cp_inv[2, idx], Cp_inv[3, idx],
                                   Cp_inv[4, idx], Cp_inv[5, idx], Cp_inv[6, idx])
        kin = finite_kinematics(Fbar, Cpi_n)
        # Fail LOUDLY on an inverted element (J = det F ≤ 0): the log-strain update
        # is ill-defined there and would otherwise silently return zero stress and a
        # wrong tangent (kin carries placeholder Finv=I, εe_tr=0). The solver has no
        # step-cutting, so throw a clear typed error (FiniteStrain.ElementInversionError).
        kin.ok || throw(ElementInversionError(e, g, kin.J))
        εp_n = SVector{6,Float64}(εp[1, idx], εp[2, idx], εp[3, idx],
                                  εp[4, idx], εp[5, idx], εp[6, idx])
        β_n = SVector{6,Float64}(β[1, idx], β[2, idx], β[3, idx],
                                 β[4, idx], β[5, idx], β[6, idx])
        ᾱ_n = ᾱ[idx]

        τ, εp_new, β_new, ᾱ_new, D, τ_princ, Cpi_new, β_sp, Rpol, Upol =
            finite_stress_update(mat, kin, Fbar, εp_n, β_n, ᾱ_n)

        # Two-point (P–F) form (FINITE_STRAIN §4.5/§4.6): P = τ·F̄⁻ᵀ, A = ∂P/∂F̄
        # (9×9), G (9×24) maps uₑ→F via the reference gradients. fe = ∫ Gᵀ P,
        # Ke = ∫ Gᵀ A G — this automatically contains both material and geometric
        # parts. With kinematic hardening the spatial back-stress depends on F̄
        # through the polar rotation, so `dPdF` adds the ∂τ/∂β·∂β_sp/∂F̄ coupling
        # (objective; makes the tangent non-symmetric but consistent).
        Gmat = _Gmatrix(dNdX)
        P = first_piola(τ, kin.Finv)              # 3×3 (from F̄)
        Pv = _p9(P)
        dtdb = mat.Hkin > 0 ? dtau_dbeta(mat, kin.εe_tr, β_sp, ᾱ_n) :
                              zero(SMatrix{6,6,Float64,36})
        A = dPdF(kin, Cpi_n, D, τ, Fbar; β_ref=β_n, R=Rpol, U=Upol, dtdb=dtdb)  # ∂P/∂F̄

        Fe += (Gmat' * Pv) * w
        if fbar
            # chain through F̄(F): Ã = ∂P/∂F = A·∂F̄/∂uₑ including the centroid J₀
            # coupling (de Souza Neto Box 15.2), as a 9×24 effective G.
            Geff = _fbar_Geff(Gmat, F, F0, G0, scale)
            Ke += (Gmat' * (A * Geff)) * w
        else
            Ke += (Gmat' * (A * Gmat)) * w
        end

        # record Kirchhoff stress (Model reports Cauchy σ = τ/J)
        σout[1, idx] = τ[1]; σout[2, idx] = τ[2]; σout[3, idx] = τ[3]
        σout[4, idx] = τ[4]; σout[5, idx] = τ[5]; σout[6, idx] = τ[6]
        if COMMIT
            εp[1, idx] = εp_new[1]; εp[2, idx] = εp_new[2]; εp[3, idx] = εp_new[3]
            εp[4, idx] = εp_new[4]; εp[5, idx] = εp_new[5]; εp[6, idx] = εp_new[6]
            β[1, idx] = β_new[1]; β[2, idx] = β_new[2]; β[3, idx] = β_new[3]
            β[4, idx] = β_new[4]; β[5, idx] = β_new[5]; β[6, idx] = β_new[6]
            ᾱ[idx] = ᾱ_new
            Cp_inv[1, idx] = Cpi_new[1]; Cp_inv[2, idx] = Cpi_new[2]; Cp_inv[3, idx] = Cpi_new[3]
            Cp_inv[4, idx] = Cpi_new[4]; Cp_inv[5, idx] = Cpi_new[5]; Cp_inv[6, idx] = Cpi_new[6]
        end
    end
    return Fe, Ke
end

# --- two-point (P–F) element helpers (FINITE_STRAIN §4.5) ---

# G (9×24): ∂F/∂uₑ. F = I + Σₐ uₐ⊗∂Nₐ/∂X ⇒ ∂F_ij/∂u_a^k = δ_ik ∂N_a/∂X_j. The F
# 9-vector is column-major: index (j-1)*3 + i. Allocation-free.
@inline function _Gmatrix(dNdX::SMatrix{8,3,Float64,24})
    return SMatrix{9,24,Float64,216}(ntuple(Val(216)) do idx
        r = (idx - 1) % 9 + 1          # F component (column-major (j-1)*3+i)
        c = (idx - 1) ÷ 9 + 1          # local dof
        i = (r - 1) % 3 + 1
        j = (r - 1) ÷ 3 + 1
        a = (c - 1) ÷ 3 + 1
        k = (c - 1) % 3 + 1
        (i == k) ? dNdX[a, j] : 0.0
    end)
end

# 3×3 P stacked column-major into a 9-vector (matches the G/A layout).
@inline _p9(P::SMatrix{3,3,Float64,9}) =
    SVector{9,Float64}(P[1, 1], P[2, 1], P[3, 1], P[1, 2], P[2, 2], P[3, 2], P[1, 3], P[2, 3], P[3, 3])

# centroid reference gradients ∂N/∂X at natural coords (0,0,0) (for F-bar).
@inline function _centroid_ref_grads(Xe::SMatrix{8,3,Float64,24})
    return DN_CENTROID * inv(jacobian(Xe, DN_CENTROID))
end

# Effective G for F-bar: ∂F̄/∂uₑ where F̄ = (J₀/J)^{1/3} F (de Souza Neto Box 15.2).
# ∂F̄/∂uₑ = scale·G + (scale/3)·vec(F)·(g₀ − g_J)ᵀ, with g_J = ∂lnJ/∂uₑ =
# vec(F⁻ᵀ)ᵀ·G and g₀ = ∂lnJ₀/∂uₑ = vec(F₀⁻ᵀ)ᵀ·G₀ (centroid). Allocation-free.
@inline function _fbar_Geff(Gmat::SMatrix{9,24,Float64,216}, F::SMatrix{3,3,Float64,9},
                            F0::SMatrix{3,3,Float64,9},
                            G0::SMatrix{9,24,Float64,216}, scale::Float64)
    gJ = Gmat' * _p9(inv(F)')               # ∂lnJ/∂uₑ  (24-vector)
    g0 = G0' * _p9(inv(F0)')                # ∂lnJ₀/∂uₑ (24-vector)
    vF = _p9(F)
    return scale * Gmat + (scale / 3) * (vF * (g0 - gJ)')
end

end # module
