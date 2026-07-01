# Tensile necking of a notched titanium round bar (finite strain, F-bar).
#
# The classic large-strain plasticity benchmark: a circular bar pulled in tension
# develops a localized **neck** that runs away to failure. A perfect bar would
# deform uniformly, so we seed a single **tiny notch** (a ~1.5% radius reduction
# at mid-length) to localize the neck at a known place. This is a genuinely 3D,
# large-rotation, near-incompressible plastic problem — exactly what the
# finite-strain F-bar element (`element=:finite_fbar`) is for.
#
# Mesh (as specified):
#   • circular cross-section meshed by an all-hex O-grid (concentric square→disk
#     map): ncross=10 ⇒ 10×10 = **100 cells per cross-section**.
#   • **200 elements along the length**, axially GRADED — fine near the mid-length
#     notch (to resolve the neck) and coarsening toward the gripped ends.
#   ⇒ 100×200 = 20,000 Hex8 elements, (11²·201)=24,321 nodes, ~73k DOFs.
#
# Material: Ti-6Al-4V (representative, linearized hardening).
#
# F-bar ⇒ non-symmetric tangent ⇒ `solve!` auto-selects the direct (UMFPACK)
# solver (CG+AMG is for the structured-box scaling path, not this graded mesh).
# This is a sizeable run; reduce `ncross`/`naxial`/`nsteps` for a quick look.
# Necking is a sharp localization: the F-bar Newton step is step-size sensitive,
# so if a step fails to converge near the neck, increase `nsteps` (smaller load
# increments). Run with threads for the assembly:
#   julia -t auto --project=. examples/necking_titanium_bar.jl

using PlasticityFEM

# concentric (elliptical) square→disk map: [-1,1]² → unit disk, all-hex, no centre
# degeneracy (same map used in cylinder_torsion.jl).
@inline _square_to_disk(u, v) = (u * sqrt(1 - v * v / 2), v * sqrt(1 - u * u / 2))

"""
    necking_mesh(R0, L, ncross, naxial; notch_amp, notch_width, grade) -> Mesh

Solid circular bar (axis = x, radius R0, length L) of Hex8 elements:
- cross-section: `ncross`×`ncross` O-grid cells (concentric square→disk map);
- axial: `naxial` (even) elements, geometrically GRADED so the smallest elements
  sit at the mid-length notch (ratio `grade` between successive element sizes from
  the centre out), coarsening toward the ends;
- notch: the cross-section radius dips to `R0·(1 − notch_amp)` in a narrow Gaussian
  of width `notch_width` centred at x = L/2 — the imperfection that localizes the neck.
"""
function necking_mesh(R0, L, ncross::Int, naxial::Int;
                      notch_amp = 0.015, notch_width = 1.0, grade = 1.02)
    @assert iseven(naxial) "naxial must be even (symmetric grading about the notch)"
    half = naxial ÷ 2

    # axial layer positions: symmetric geometric grading, finest at the centre.
    # element sizes from the centre outward h_j = h0·grade^j, Σ_{0}^{half-1} = L/2.
    ssum = sum(grade^j for j in 0:half-1)
    h0 = (L / 2) / ssum
    xs = Vector{Float64}(undef, naxial + 1)
    c = half + 1                       # 1-based index of the centre layer
    xs[c] = L / 2
    posR = L / 2; posL = L / 2
    for j in 0:half-1
        posR += h0 * grade^j; xs[c + (j + 1)] = posR
        posL -= h0 * grade^j; xs[c - (j + 1)] = posL
    end
    xs[1] = 0.0; xs[end] = L           # pin exact endpoints

    npc = ncross + 1
    nlayer = npc * npc
    nnodes = nlayer * (naxial + 1)
    nodes = Matrix{Float64}(undef, 3, nnodes)
    nid(i, j, k) = k * nlayer + j * npc + i + 1
    for k in 0:naxial
        x = xs[k + 1]
        Rx = R0 * (1 - notch_amp * exp(-((x - L / 2) / notch_width)^2))   # notch
        for j in 0:ncross, i in 0:ncross
            u = -1.0 + 2.0 * i / ncross
            v = -1.0 + 2.0 * j / ncross
            my, mz = _square_to_disk(u, v)
            id = nid(i, j, k)
            nodes[1, id] = x            # axial
            nodes[2, id] = Rx * my      # y
            nodes[3, id] = Rx * mz      # z
        end
    end

    nelem = ncross * ncross * naxial
    elements = Matrix{Int}(undef, 8, nelem)
    e = 0
    for k in 0:naxial-1, j in 0:ncross-1, i in 0:ncross-1
        e += 1
        # Hex8/VTK ordering; (i,j)→(y,z), k→x axial (right-handed ⇒ detJ>0).
        elements[1, e] = nid(i,     j,     k); elements[2, e] = nid(i + 1, j,     k)
        elements[3, e] = nid(i + 1, j + 1, k); elements[4, e] = nid(i,     j + 1, k)
        elements[5, e] = nid(i,     j,     k + 1); elements[6, e] = nid(i + 1, j,     k + 1)
        elements[7, e] = nid(i + 1, j + 1, k + 1); elements[8, e] = nid(i,     j + 1, k + 1)
    end
    return Mesh(nodes, elements, nnodes, nelem)
end

function main()
    R0 = 5.0; L = 50.0                  # 10 mm diameter, 50 mm long
    ncross = 10                         # 10×10 = 100 cells per cross-section
    naxial = 200                        # 200 elements lengthwise (graded)
    mesh = necking_mesh(R0, L, ncross, naxial;
                        notch_amp = 0.015, notch_width = 1.0, grade = 1.02)

    # Ti-6Al-4V (representative; the solver supports linear hardening, so this is a
    # linearized fit of titanium's modest post-yield hardening).
    ti = J2Material(E = 113.8e3, ν = 0.342, σy0 = 880.0, Hiso = 1000.0)

    # F-bar finite-strain element (needed for the near-incompressible plastic neck)
    model = Model(mesh, ti; element = :finite_fbar)

    # gripped tension: clamp x=0; pull x=L axially with its lateral motion gripped
    fix!(model, on_face(mesh, :xmin))
    fix!(model, on_face(mesh, :xmax), :y)
    fix!(model, on_face(mesh, :xmax), :z)
    elong = 0.12 * L                    # 12% nominal elongation (well past neck onset)
    prescribe!(model, on_face(mesh, :xmax), :x, elong)

    res = solve!(model; nsteps = 60, tol = 1e-7, maxiter = 40)

    # --- postprocess: where did it neck? ---
    u  = nodal_displacements(model)
    ᾱ  = equivalent_plastic_strain(model)
    gp_max = argmax(ᾱ)
    e_max  = (gp_max - 1) ÷ 8 + 1
    k_max  = (e_max - 1) ÷ (ncross * ncross)         # axial element index of max ᾱ
    x_max  = 0.5 * (((k_max) / naxial) + ((k_max + 1) / naxial)) * L  # ~axial position

    # neck radius: minimum current cross-section radius over axial layers (from the
    # deformed positions of the outer-boundary nodes of each layer).
    npc = ncross + 1; nlayer = npc * npc
    isbnd(i, j) = (i == 0 || i == ncross || j == 0 || j == ncross)
    neck_R = R0; neck_x = 0.0
    for k in 0:naxial
        rmax = 0.0
        for j in 0:ncross, i in 0:ncross
            isbnd(i, j) || continue
            nd = k * nlayer + j * npc + i + 1
            y = mesh.nodes[2, nd] + u[2, nd]
            z = mesh.nodes[3, nd] + u[3, nd]
            rmax = max(rmax, sqrt(y^2 + z^2))
        end
        if rmax < neck_R
            neck_R = rmax; neck_x = (k / naxial) * L
        end
    end

    println("=== necking of a notched titanium round bar (finite strain, F-bar) ===")
    println("mesh                 : $(ncross)×$(ncross)×$(naxial)  ($(mesh.nelem) elems, $(3*mesh.nnodes) DOFs)")
    println("material             : Ti-6Al-4V  E=113.8 GPa, ν=0.342, σy0=880 MPa, Hiso=1000")
    println("converged            : ", res.converged, "   (", length(res.iters), " load steps run)")
    println("max eq. plastic strn : ", maximum(ᾱ), "  at x ≈ ", round(x_max, digits=2), " (notch at L/2 = ", L/2, ")")
    println("neck radius (min)    : ", round(neck_R, digits=4), " at x ≈ ", round(neck_x, digits=2),
            "   vs R0 = ", R0, "  ⇒ ", round(100*(1 - neck_R/R0), digits=1), "% reduction")

    println("\nwrote: ", write_vtu(joinpath(@__DIR__, "necking_titanium_bar"), model))
    return res.converged
end

main()
