"""
    MeshMod

Structured box mesh generator and predicate-based node/face selection.
See DESIGN.md §4.2.
"""
module MeshMod

export Mesh, box_mesh, select_nodes, on_face, dof, GaussState, reset_state!

"""
    Mesh

`nodes` is 3×nnodes (column = node coords), `elements` is 8×nelem (column =
Hex8 node ids). Contiguous columns for cache-friendly access (DESIGN §4.2).
"""
struct Mesh
    nodes::Matrix{Float64}
    elements::Matrix{Int}
    nnodes::Int
    nelem::Int
end

"""
    box_mesh(lx, ly, lz, nx, ny, nz) -> Mesh

Structured grid of nx×ny×nz Hex8 elements filling [0,lx]×[0,ly]×[0,lz].
Lexicographic node numbering (x fastest). (DESIGN §4.2)
"""
function box_mesh(lx, ly, lz, nx::Int, ny::Int, nz::Int)
    lx = Float64(lx); ly = Float64(ly); lz = Float64(lz)
    npx = nx + 1; npy = ny + 1; npz = nz + 1
    nnodes = npx * npy * npz
    nodes = Matrix{Float64}(undef, 3, nnodes)
    # x fastest, then y, then z
    nid(i, j, k) = (k * npy + j) * npx + i + 1   # i,j,k are 0-based
    for k in 0:nz, j in 0:ny, i in 0:nx
        n = nid(i, j, k)
        nodes[1, n] = lx * i / nx
        nodes[2, n] = ly * j / ny
        nodes[3, n] = lz * k / nz
    end
    nelem = nx * ny * nz
    elements = Matrix{Int}(undef, 8, nelem)
    e = 0
    for k in 0:nz-1, j in 0:ny-1, i in 0:nx-1
        e += 1
        # Hex8 / VTK ordering (DESIGN §3.1): bottom face CCW then top face CCW
        elements[1, e] = nid(i,     j,     k)
        elements[2, e] = nid(i + 1, j,     k)
        elements[3, e] = nid(i + 1, j + 1, k)
        elements[4, e] = nid(i,     j + 1, k)
        elements[5, e] = nid(i,     j,     k + 1)
        elements[6, e] = nid(i + 1, j,     k + 1)
        elements[7, e] = nid(i + 1, j + 1, k + 1)
        elements[8, e] = nid(i,     j + 1, k + 1)
    end
    return Mesh(nodes, elements, nnodes, nelem)
end

"""
    select_nodes(mesh, pred) -> Vector{Int}

Node ids whose coordinates satisfy `pred(x,y,z)::Bool` (DESIGN §4.2).
"""
function select_nodes(mesh::Mesh, pred)
    ids = Int[]
    @inbounds for n in 1:mesh.nnodes
        if pred(mesh.nodes[1, n], mesh.nodes[2, n], mesh.nodes[3, n])
            push!(ids, n)
        end
    end
    return ids
end

"""
    on_face(mesh, face) -> Vector{Int}

Nodes on a bounding-box face, `face ∈ {:xmin,:xmax,:ymin,:ymax,:zmin,:zmax}`.
Uses a tolerance derived from the bounding box (DESIGN §4.2).
"""
function on_face(mesh::Mesh, face::Symbol)
    xmin = minimum(@view mesh.nodes[1, :]); xmax = maximum(@view mesh.nodes[1, :])
    ymin = minimum(@view mesh.nodes[2, :]); ymax = maximum(@view mesh.nodes[2, :])
    zmin = minimum(@view mesh.nodes[3, :]); zmax = maximum(@view mesh.nodes[3, :])
    L = max(xmax - xmin, ymax - ymin, zmax - zmin)
    tol = 1e-8 * max(L, 1.0)
    pred = if face === :xmin
        (x, y, z) -> abs(x - xmin) <= tol
    elseif face === :xmax
        (x, y, z) -> abs(x - xmax) <= tol
    elseif face === :ymin
        (x, y, z) -> abs(y - ymin) <= tol
    elseif face === :ymax
        (x, y, z) -> abs(y - ymax) <= tol
    elseif face === :zmin
        (x, y, z) -> abs(z - zmin) <= tol
    elseif face === :zmax
        (x, y, z) -> abs(z - zmax) <= tol
    else
        error("unknown face $face (use :xmin,:xmax,:ymin,:ymax,:zmin,:zmax)")
    end
    return select_nodes(mesh, pred)
end

"""
    dof(node, comp) -> Int

Global DOF index of (node, component): 3(node−1)+comp, comp∈{1,2,3} (DESIGN §9).
"""
@inline dof(node::Integer, comp::Integer) = 3 * (node - 1) + comp

"""
    GaussState

Struct-of-arrays per-Gauss-point state, sized over ngp_total = nelem×8
(DESIGN §4.3). `εp` engineering-shear Voigt plastic strain, `β` back-stress
deviator (physical shear), `ᾱ` accumulated plastic strain, `σ` stress (output).

`Cp_inv` (6×ngp, symmetric Voigt, physical shear) is the finite-strain plastic
configuration Cᵖ⁻¹ = Fᵖ⁻¹Fᵖ⁻ᵀ, initialized to the identity `[1,1,1,0,0,0]`
(FINITE_STRAIN §6.2). Small-strain models leave it at identity (unused).
"""
struct GaussState
    εp::Matrix{Float64}   # 6 × ngp
    β::Matrix{Float64}    # 6 × ngp
    ᾱ::Vector{Float64}    # ngp
    σ::Matrix{Float64}    # 6 × ngp
    Cp_inv::Matrix{Float64}   # 6 × ngp  (finite strain; identity for small strain)
end

function GaussState(ngp::Int)
    Cp_inv = zeros(6, ngp)
    @inbounds for g in 1:ngp
        Cp_inv[1, g] = 1.0; Cp_inv[2, g] = 1.0; Cp_inv[3, g] = 1.0
    end
    return GaussState(zeros(6, ngp), zeros(6, ngp), zeros(ngp), zeros(6, ngp), Cp_inv)
end

function Base.copyto!(dst::GaussState, src::GaussState)
    copyto!(dst.εp, src.εp)
    copyto!(dst.β, src.β)
    copyto!(dst.ᾱ, src.ᾱ)
    copyto!(dst.σ, src.σ)
    copyto!(dst.Cp_inv, src.Cp_inv)
    return dst
end

"""
    reset_state!(st)

Reset a `GaussState` to the undeformed, unhardened state: zero `εp,β,ᾱ,σ` and
set `Cp_inv` to the identity `[1,1,1,0,0,0]` per Gauss point (FINITE_STRAIN §6.2).
"""
function reset_state!(st::GaussState)
    fill!(st.εp, 0.0); fill!(st.β, 0.0); fill!(st.ᾱ, 0.0); fill!(st.σ, 0.0)
    fill!(st.Cp_inv, 0.0)
    ngp = length(st.ᾱ)
    @inbounds for g in 1:ngp
        st.Cp_inv[1, g] = 1.0; st.Cp_inv[2, g] = 1.0; st.Cp_inv[3, g] = 1.0
    end
    return st
end

end # module
