# Necking of a notched titanium round bar — finite-strain FEM (F-bar), PERFECTLY
# SYMMETRIC setup. Counterpart of particle_method/examples/necking_titanium_bar.jl;
# both solvers run this identical problem so the neck compares apples-to-apples.
#
# Symmetric formulation (see also the MPM file):
#   • round bar with a SMOOTH PARABOLIC profile — center `taper` thinner than the ends
#     (default 12%) — so the mid-length is the global-min section that localizes the neck;
#   • TENSION applied uniformly on BOTH end faces: u_x = ∓δ/2 (each end moves ±δ/2),
#     with the ends left FREE in y,z so the cross-section contracts (no lateral grip
#     ⇒ no grip stress raiser, no grip-corner folding);
#   • in-plane rigid-body modes removed by SYMMETRY-PLANE ROLLERS on the two axis
#     planes: u_y=0 on the y=0 plane, u_z=0 on the z=0 plane. These remove y/z
#     translation and rotation about the bar axis with ZERO over-constraint — an
#     axisymmetric neck keeps those planes planar anyway.
#
# Displacement (not load) control: necking is a softening instability, so a prescribed
# displacement traverses the post-peak branch stably where load control would snap through.
#
# FINE MESH: cross-section O-grid ncross×ncross (ncross EVEN ⇒ node columns lie exactly
# on the y=0 and z=0 planes for the rollers), naxial axial layers geometrically GRADED
# so the smallest elements sit at the notch. F-bar ⇒ direct (UMFPACK) solver.
#   julia -t auto --project=. examples/necking_titanium_symmetric.jl

using PlasticityFEM

# concentric square→disk map ([-1,1]² → unit disk, all-hex, no centre degeneracy)
@inline _square_to_disk(u, v) = (u * sqrt(1 - v * v / 2), v * sqrt(1 - u * u / 2))

const R0     = 5.0                          # END radius (bar is thickest at the ends)
const Lbar   = 50.0
const taper  = parse(Float64, get(ENV, "TAPER", "0.12"))   # center is `taper` THINNER than the ends

# smooth parabolic profile: R(x) = R0·(1 − taper·(1 − s²)), s = (x−L/2)/(L/2) ∈ [−1,1].
# Minimum (thinnest) at mid-length, rising to R0 at both ends — a single smooth curve, no
# junctions, so the center is the unambiguous global-min section that localizes the neck.
Rprofile(x) = (s = (x - Lbar / 2) / (Lbar / 2); R0 * (1 - taper * (1 - s^2)))

"""
    necking_mesh(R0, L, ncross, naxial; grade) -> Mesh

Circular bar (axis = x) of Hex8 O-grid elements whose radius follows the smooth parabolic
`Rprofile(x)` (thinnest at mid-length). Axial layers geometrically graded, finest at the
center. `ncross` must be even so node columns fall on the y=0 and z=0 planes.
"""
function necking_mesh(R0, L, ncross::Int, naxial::Int; grade = 1.02)
    @assert iseven(naxial) "naxial must be even (symmetric grading about the notch)"
    @assert iseven(ncross) "ncross must be even (node columns on the y=0 / z=0 planes)"
    half = naxial ÷ 2
    ssum = sum(grade^j for j in 0:half-1)
    h0 = (L / 2) / ssum
    xs = Vector{Float64}(undef, naxial + 1)
    c = half + 1
    xs[c] = L / 2
    posR = L / 2; posL = L / 2
    for j in 0:half-1
        posR += h0 * grade^j; xs[c + (j + 1)] = posR
        posL -= h0 * grade^j; xs[c - (j + 1)] = posL
    end
    xs[1] = 0.0; xs[end] = L

    npc = ncross + 1
    nlayer = npc * npc
    nnodes = nlayer * (naxial + 1)
    nodes = Matrix{Float64}(undef, 3, nnodes)
    nid(i, j, k) = k * nlayer + j * npc + i + 1
    for k in 0:naxial
        x = xs[k + 1]
        Rx = Rprofile(x)
        for j in 0:ncross, i in 0:ncross
            u = -1.0 + 2.0 * i / ncross
            v = -1.0 + 2.0 * j / ncross
            my, mz = _square_to_disk(u, v)
            id = nid(i, j, k)
            nodes[1, id] = x
            nodes[2, id] = Rx * my
            nodes[3, id] = Rx * mz
        end
    end

    nelem = ncross * ncross * naxial
    elements = Matrix{Int}(undef, 8, nelem)
    e = 0
    for k in 0:naxial-1, j in 0:ncross-1, i in 0:ncross-1
        e += 1
        elements[1, e] = nid(i,     j,     k); elements[2, e] = nid(i + 1, j,     k)
        elements[3, e] = nid(i + 1, j + 1, k); elements[4, e] = nid(i,     j + 1, k)
        elements[5, e] = nid(i,     j,     k + 1); elements[6, e] = nid(i + 1, j,     k + 1)
        elements[7, e] = nid(i + 1, j + 1, k + 1); elements[8, e] = nid(i,     j + 1, k + 1)
    end
    return Mesh(nodes, elements, nnodes, nelem)
end

function main()
    ncross = parse(Int, get(ENV, "NCROSS", "12"))   # FINE: 12×12 = 144 cells per cross-section
    naxial = parse(Int, get(ENV, "NAXIAL", "160"))  # FINE: 160 graded axial layers (finest at notch)
    grade  = parse(Float64, get(ENV, "GRADE", "1.02"))   # 1.0 ⇒ uniform axial spacing
    mesh = necking_mesh(R0, Lbar, ncross, naxial; grade = grade)

    ti = J2Material(E = 113.8e3, ν = 0.342, σy0 = 880.0, Hiso = 1000.0)
    model = Model(mesh, ti; element = :finite_fbar)

    # --- perfectly symmetric BCs ---
    δ = 0.09 * Lbar                                  # 9% nominal elongation (total)
    tol = 1e-6 * R0
    yplane = select_nodes(mesh, (x, y, z) -> abs(y) < tol)   # y=0 symmetry plane
    zplane = select_nodes(mesh, (x, y, z) -> abs(z) < tol)   # z=0 symmetry plane
    fix!(model, yplane, :y)                          # roller: removes y-translation + x-rotation
    fix!(model, zplane, :z)                          # roller: removes z-translation + x-rotation
    # Symmetric pull: both ends ±δ/2. The end faces are ALSO laterally gripped (y,z fixed):
    # the neck is at the tapered center (global-min section) far from the ends, so end grips
    # don't affect it, but they stabilize the coarse tapered corner elements (free ends +
    # taper give a near-singular corner that diverges). FEM is quasi-static ⇒ this one-ended-
    # vs-symmetric / gripped-vs-free end choice does not change the center neck.
    fix!(model, on_face(mesh, :xmin), :y); fix!(model, on_face(mesh, :xmin), :z)
    fix!(model, on_face(mesh, :xmax), :y); fix!(model, on_face(mesh, :xmax), :z)
    prescribe!(model, on_face(mesh, :xmin), :x, -δ / 2)
    prescribe!(model, on_face(mesh, :xmax), :x,  δ / 2)

    nsteps = parse(Int, get(ENV, "NSTEPS", "160"))   # deep 12% neck needs small load steps
    res = solve!(model; nsteps = nsteps, tol = 1e-7, maxiter = 40)

    # --- postprocess: binned neck profile (same 20-bin format as the MPM example) ---
    u = nodal_displacements(model)
    ᾱ = equivalent_plastic_strain(model)             # per Gauss point (8 per element)

    nbin = 20
    edges = range(0, Lbar; length = nbin + 1)
    binof(x) = clamp(searchsortedlast(edges, x), 1, nbin)
    Rbin = fill(0.0, nbin)
    αbin = fill(0.0, nbin)

    npc = ncross + 1; nlayer = npc * npc
    isbnd(i, j) = (i == 0 || i == ncross || j == 0 || j == ncross)
    for k in 0:naxial
        xref = mesh.nodes[1, k * nlayer + 1]
        b = binof(xref)
        for j in 0:ncross, i in 0:ncross
            isbnd(i, j) || continue
            nd = k * nlayer + j * npc + i + 1
            y = mesh.nodes[2, nd] + u[2, nd]
            z = mesh.nodes[3, nd] + u[3, nd]
            Rbin[b] = max(Rbin[b], sqrt(y^2 + z^2))
        end
    end
    for e in 1:mesh.nelem
        k = (e - 1) ÷ (ncross * ncross)
        xc = 0.5 * (mesh.nodes[1, k*nlayer+1] + mesh.nodes[1, (k+1)*nlayer+1])
        b = binof(xc)
        amax = 0.0
        for g in 1:8
            amax = max(amax, ᾱ[(e - 1) * 8 + g])
        end
        αbin[b] = max(αbin[b], amax)
    end

    neck_R, imin = findmin(Rbin)
    αmax, imax = findmax(αbin)

    println("\n=== necking of a notched titanium round bar (FEM, finite strain, F-bar, symmetric) ===")
    println("mesh                 : $(ncross)×$(ncross)×$(naxial)  ($(mesh.nelem) elems, $(3*mesh.nnodes) DOFs)")
    println("material             : Ti-6Al-4V  E=113.8 GPa, ν=0.342, σy0=880 MPa, Hiso=1000")
    println("converged            : ", res.converged, "   (", length(res.iters), " load steps run)")
    println("max eq. plastic strn : ", round(αmax, digits=4), "  at x ≈ ",
            round(0.5 * (edges[imax] + edges[imax+1]), digits=2), " (notch at L/2 = ", Lbar/2, ")")
    println("neck radius (min)    : ", round(neck_R, digits=4), " at x ≈ ",
            round(0.5 * (edges[imin] + edges[imin+1]), digits=2),
            "   vs R0 = ", R0, "  ⇒ ", round(100*(1 - neck_R/R0), digits=1), "% reduction")

    println("\nbin |  x   | radius | ᾱ")
    for b in 1:nbin
        xm = 0.5 * (edges[b] + edges[b+1])
        println("  ", lpad(b,2), " | ", lpad(round(xm,digits=2),5), " | ",
                round(Rbin[b], digits=4), " | ", round(αbin[b], digits=4))
    end

    println("\nwrote: ", write_vtu(joinpath(@__DIR__, "necking_titanium_symmetric"), model))
    return res.converged
end

main()
