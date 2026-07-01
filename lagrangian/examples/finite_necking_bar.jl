# Finite-strain necking of a tension bar (3D, F-bar).
#
# A large-deformation J2 benchmark (de Souza Neto / Simo): a bar pulled in tension
# localizes into a neck once plastic flow dominates. The F-bar element relieves the
# volumetric locking of the trilinear Hex8 so the cross-section can contract.
# Run:  julia --project=. examples/finite_necking_bar.jl
#
# IMPERFECTION TRIGGER. A perfectly prismatic bar deforms HOMOGENEOUSLY — with no
# preferred neck site, localization never breaks symmetry numerically (the earlier
# version of this example showed exactly this: it converged but never necked). We
# seed a small geometric imperfection — a smooth ~2% reduction of the cross-section
# at mid-length — so plastic strain concentrates there and a neck forms at x = L/2.
#
# SCOPE / HONESTY. Full post-bifurcation necking is a SOFTENING instability: past
# the load maximum the response snaps back and a plain displacement-controlled
# full-Newton solver (no line search / arc-length continuation, which this code
# does not yet implement) overshoots into invalid configurations and diverges. So
# this example runs into the EARLY localization regime — enough to show the neck
# forming at the imperfection (the contraction and the equivalent-plastic-strain
# peak both localize at mid-length, not at the grips) — but not to rupture. A
# moderate hardening modulus keeps the path stable through the elongation shown.
# Driving it further (larger `elong`, lower `Hiso`) will diverge until an
# arc-length solver is added.
#
# Output: finite_necking_bar.vtu — open in ParaView, Warp By Vector on
# `Displacement` to see the necked shape; color by `EqPlasticStrain`.

using PlasticityFEM
using Printf

# slender bar [0,L]×[0,w]×[0,w]; refine along the axis so a neck can localize
L, w = 10.0, 1.0
mesh = box_mesh(L, w, w, 16, 3, 3)

# Geometric imperfection: shrink the transverse (y,z) coords toward the SYMMETRY
# planes (y=0, z=0 — the roller faces) by a factor that dips to (1 − amp) at
# x = L/2 (smooth Gaussian bell). Scaling about the origin (y → y·s) keeps the
# y=0 / z=0 boundary nodes exactly on the symmetry planes (rollers stay consistent)
# while pulling the outer surface (y=w → w·s) inward — a ~amp reduction of the
# mid-length cross-section that seeds the neck site.
let amp = 0.02, x0 = L / 2, xwid = L / 5
    @inbounds for n in 1:mesh.nnodes
        x = mesh.nodes[1, n]
        s = 1.0 - amp * exp(-((x - x0) / xwid)^2)   # ~1 at ends, (1−amp) at center
        mesh.nodes[2, n] *= s
        mesh.nodes[3, n] *= s
    end
end

# steel-like; a moderate hardening modulus keeps the localizing path stable for the
# displacement-controlled full-Newton solve (see SCOPE note above).
steel = J2Material(E = 210e3, ν = 0.3, σy0 = 250.0, Hiso = 2000.0)

model = Model(mesh, steel; element = :finite_fbar)

# symmetry rollers on the three min faces + axial pull on xmax (displacement control)
fix!(model, on_face(mesh, :xmin), :x)
fix!(model, on_face(mesh, :ymin), :y)
fix!(model, on_face(mesh, :zmin), :z)
prescribe!(model, on_face(mesh, :xmax), :x, 0.25)    # 2.5% nominal elongation

# solve! auto-selects the direct solver (the F-bar tangent is non-symmetric).
res = solve!(model; nsteps = 20, tol = 1e-7, maxiter = 40, verbose = false)

println("converged: ", res.converged, "  iters/step: ", res.iters)
u = nodal_displacements(model)
ᾱ = equivalent_plastic_strain(model)
println("max axial displacement: ", maximum(u[1, :]))
println("max equivalent plastic strain: ", maximum(ᾱ))

# --- neck localization diagnostics --------------------------------------------
# Bin the deformed transverse half-width and the Gauss-point ᾱ along the axis to
# confirm the neck localizes at mid-length rather than deforming homogeneously.
let nbin = 8
    edges = range(0, L; length = nbin + 1)
    halfw = fill(0.0, nbin)               # deformed transverse extent per axial bin
    @inbounds for n in 1:mesh.nnodes
        X = mesh.nodes[1, n]
        b = clamp(searchsortedlast(edges, X), 1, nbin)
        halfw[b] = max(halfw[b], mesh.nodes[2, n] + u[2, n])   # symmetry plane at y=0
    end
    αbin = fill(0.0, nbin)                 # peak ᾱ per axial bin
    @inbounds for e in 1:mesh.nelem
        xc = 0.0
        for a in 1:8
            xc += mesh.nodes[1, mesh.elements[a, e]]
        end
        xc /= 8
        b = clamp(searchsortedlast(edges, xc), 1, nbin)
        for g in 1:8
            αbin[b] = max(αbin[b], ᾱ[(e - 1) * 8 + g])
        end
    end
    println("\naxial bin |  half-width  |  peak ᾱ")
    for b in 1:nbin
        xm = 0.5 * (edges[b] + edges[b+1])
        @printf("  x=%4.1f   |   %.4f     |  %.4f\n", xm, halfw[b], αbin[b])
    end
    wmin, imin = findmin(halfw)
    αmax, imax = findmax(αbin)
    xneck = 0.5 * (edges[imin] + edges[imin+1])
    xαpk  = 0.5 * (edges[imax] + edges[imax+1])
    @printf("\nthinnest section at x=%.1f (half-width %.4f vs ends ~%.4f) — contraction %.1f%%\n",
            xneck, wmin, maximum(halfw), 100 * (1 - wmin / maximum(halfw)))
    @printf("peak plastic strain at x=%.1f (ᾱ=%.4f)\n", xαpk, αmax)
    midbins = (nbin ÷ 2, nbin ÷ 2 + 1)
    localized = wmin < maximum(halfw) && imin in midbins && imax in midbins
    println(localized ?
        "NECK LOCALIZED at mid-length ✓ (thinnest section and ᾱ peak both at x≈L/2)" :
        "deformation did not localize at mid-length")
end

out = write_vtu("finite_necking_bar", model)
println("wrote ", out)
