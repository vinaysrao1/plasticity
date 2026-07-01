# Torsion of a CYLINDRICAL cantilever — round-vs-square comparison.
#
# Companion to cantilever_torsion_T72.jl: the same length (L=10) and the same
# cross-sectional AREA (1.0, matching the 1×1 square ⇒ a solid circle of radius
# R = 1/√π ≈ 0.564), the same clamp (x=0), and the SAME applied torque (T=72)
# about the long axis. Only the cross-section shape differs. Uses the FINITE-STRAIN
# element (see cantilever_torsion_T72.jl).
#
# Why this is an instructive comparison: the solid circle is the *optimal* torsion
# section. In the small-strain first-yield sense, a square concentrates shear at
# its edge mid-points (yield torque ≈ 0.208·a³·σy0/√3 ≈ 30), while an equal-area
# circle spreads it over a smoother boundary (yield torque ≈ π R³ σy0/(2√3) ≈ 40.7)
# — so the circle is the stiffer, later-yielding section.
#
# But T=72 is ≈1.8–2.4× those yield torques, and with FINITE STRAIN the bars twist
# into a genuinely large-deformation, fully-plastic regime (end twist ≈0.6–0.7 rad,
# ≈35–40°). At this load essentially the whole cross-section yields for BOTH shapes
# — not the small-strain picture of a thin yielded ring. The round shaft, being the
# stiffer section, twists slightly LESS than the square for the same torque, yet
# develops a higher peak von-Mises / plastic strain. Compare the two .vtu files in
# ParaView (Warp By Vector + colour by EqPlasticStrain) to see the difference.
#
# The mesh is an all-hex O-grid: a structured n×n grid on the square [-1,1]² is
# mapped onto the disk by the smooth concentric (elliptical) map, then extruded
# along x. This keeps every element a valid 8-node hex (no degenerate wedge at the
# centre that a naive polar mesh would create). With n=16, nx=160 the grid is
# topologically 16×16×160 — (17²·161)=46,529 nodes = 139,587 DOFs, identical to
# the 160×16×16 box. Solved with CG+AMG. The O-grid is a distorted (non-uniform)
# mesh, so the finite-strain solve recomputes element geometry every iteration and
# is noticeably slower than the uniform box — run with threads:
#   JULIA_NUM_THREADS=8 julia --project=. examples/cylinder_torsion_T72.jl

using PlasticityFEM

# Concentric (elliptical) map of the square [-1,1]² onto the unit disk. Smooth and
# bijective ⇒ all-hex, positively-oriented elements; centre→centre, edge-mid→axis
# points, corners→the 45° points on the circle.
@inline _square_to_disk(u, v) = (u * sqrt(1 - v * v / 2), v * sqrt(1 - u * u / 2))

# Build a solid cylinder (axis = x, radius R, length L) of all-hex elements from a
# structured n×n cross-section grid extruded into nx axial layers (O-grid).
function cylinder_mesh(L, R, n::Int, nx::Int)
    npc = n + 1
    nlayer = npc * npc
    nnodes = nlayer * (nx + 1)
    nodes = Matrix{Float64}(undef, 3, nnodes)
    nid(i, j, k) = k * nlayer + j * npc + i + 1          # 0-based i,j,k
    for k in 0:nx, j in 0:n, i in 0:n
        u = -1.0 + 2.0 * i / n
        v = -1.0 + 2.0 * j / n
        my, mz = _square_to_disk(u, v)
        id = nid(i, j, k)
        nodes[1, id] = L * k / nx        # axial coordinate x
        nodes[2, id] = R * my            # y (disk plane)
        nodes[3, id] = R * mz            # z (disk plane)
    end
    nelem = n * n * nx
    elements = Matrix{Int}(undef, 8, nelem)
    e = 0
    for k in 0:nx-1, j in 0:n-1, i in 0:n-1
        e += 1
        # Hex8 / VTK ordering (Mesh.jl §3.1); (i,j)→(y,z) cross-section, k→x axial.
        # (y,z,x) is a cyclic permutation of (x,y,z) ⇒ right-handed ⇒ detJ > 0.
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

function main()
    L = 10.0
    R = 1 / sqrt(π)                       # area = πR² = 1.0 (same as the 1×1 square)
    mesh = cylinder_mesh(L, R, 16, 160)   # 16×16×160 grid → 46,529 nodes = 139,587 DOFs
    mat  = J2Material(E = 210e3, ν = 0.3, σy0 = 250.0, Hiso = 1000.0)
    # Finite-strain (large-deformation) element. Isotropic hardening only ⇒ the
    # consistent tangent is symmetric, so solve! keeps the CG+AMG scaling path.
    model = Model(mesh, mat; element = :finite)

    # clamp the base disk (x = 0 face)
    fix!(model, on_face(mesh, :xmin))

    # same torque T about x on the x = L face, as tangential nodal forces.
    # Disk is centred at (y,z)=(0,0), so the offset is just (y,z): (Fy,Fz)=k(−z,y),
    # k = T / Σ(y²+z²) (zero net force, net moment T about x). Ramped over steps.
    T_total = 72.0                        # SAME torque as the square example
    endface = on_face(mesh, :xmax)
    S = 0.0
    for nd in endface
        S += mesh.nodes[2, nd]^2 + mesh.nodes[3, nd]^2
    end
    k = T_total / S
    for nd in endface
        y = mesh.nodes[2, nd]; z = mesh.nodes[3, nd]
        load!(model, [nd], :y, -k * z)
        load!(model, [nd], :z,  k * y)
    end

    res = solve!(model; nsteps = 120, tol = 1e-8, maxiter = 60)

    # postprocess
    u   = nodal_displacements(model)
    σ   = gauss_stress(model)
    ᾱ   = equivalent_plastic_strain(model)
    ngp = length(ᾱ)
    σvm_max = maximum(von_mises(@view σ[:, i]) for i in 1:ngp)

    num = 0.0; den = 0.0
    for nd in endface
        y = mesh.nodes[2, nd]; z = mesh.nodes[3, nd]
        num += y * u[3, nd] - z * u[2, nd]
        den += y * y + z * z
    end
    twist = num / den                     # radians of twist of the free end

    println("=== torsion of a CYLINDRICAL cantilever (FINITE STRAIN; same area & torque as the square) ===")
    println("radius R / area      : ", R, " / ", π * R^2)
    println("mesh                 : 16×16×160 O-grid  ($(mesh.nelem) elems, $(3*mesh.nnodes) DOFs)")
    println("applied torque T     : ", T_total, "   (circular yield torque ≈ 40.7; square's ≈ 30)")
    println("converged            : ", res.converged)
    println("end-face twist (rad) : ", twist)
    println("max von Mises stress : ", σvm_max, "   (yield σy0 = 250)")
    println("max eq. plastic strn : ", maximum(ᾱ))
    println("yielded Gauss points : ", count(>(0.0), ᾱ), " / ", ngp,
            "  (finite-strain large twist ⇒ nearly the whole section is plastic)")

    println("\nwrote: ", write_vtu(joinpath(@__DIR__, "cylinder_torsion_T72"), model))
    return res.converged
end

main()
