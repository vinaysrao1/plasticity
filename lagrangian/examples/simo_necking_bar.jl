# Simo (1988) round-bar necking benchmark — WITH SATURATION (Voce) HARDENING.
#
# This is the classic large-strain J2 localization benchmark, and the reason the
# saturation-hardening kernel exists: the published Simo necking curve uses a
# NONLINEAR (Voce) isotropic law, not the linear hardening the solver originally
# shipped with. With linear hardening the neck is qualitatively right but the
# load–elongation curve and neck geometry do not match Simo's published values;
# saturation hardening (Materials.jl) is what makes this benchmark faithful.
#
# Material (Simo 1988 / Simo & Armero 1992 — verify final numbers against the
# primary source before using as a hard pass/fail):
#   E = 206.9 GPa, ν = 0.29, σy0 = 450 MPa,
#   σy(ᾱ) = σy0 + (σ∞−σy0)(1−e^{−δᾱ}) + H·ᾱ,  σ∞ = 715 MPa, δ = 16.93, H = 129.24 MPa.
#
# Geometry: solid circular bar, R0 = 6.413 mm, L0 = 53.334 mm, with a small
# mid-length imperfection (radius dips to ≈0.982·R0, i.e. 1.8% reduction) to seed
# the neck at x = L/2. NOTE the imperfection here is a smooth Gaussian notch
# (reusing necking_titanium_bar.jl's O-grid mesh generator), whereas Simo's is a
# gentle linear taper — so the *material* is faithful but the imperfection SHAPE
# differs; expect the right qualitative curve and a comparable neck reduction, not
# a digit-for-digit match to the published figure.
#
# F-bar finite-strain element (near-incompressible plastic flow) ⇒ non-symmetric
# tangent ⇒ direct solver. This validates the new nonlinear consistent tangent in
# a FULL assembly: if Newton converges at its usual quadratic rate, the
# saturation-hardening algorithmic tangent (Materials.jl) is correct end-to-end.
#
#   julia -t auto --project=. examples/simo_necking_bar.jl

using PlasticityFEM
using Printf

# concentric square→disk map (same as cylinder_torsion.jl / necking_titanium_bar.jl)
@inline _square_to_disk(u, v) = (u * sqrt(1 - v * v / 2), v * sqrt(1 - u * u / 2))

# Solid circular bar (axis = x), Hex8 O-grid, mid-length Gaussian notch. Copied
# from necking_titanium_bar.jl (examples are standalone) — see there for details.
function necking_mesh(R0, L, ncross::Int, naxial::Int;
                      notch_amp = 0.018, notch_width = 2.0, grade = 1.03)
    @assert iseven(naxial) "naxial must be even (symmetric grading about the notch)"
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
        Rx = R0 * (1 - notch_amp * exp(-((x - L / 2) / notch_width)^2))
        for j in 0:ncross, i in 0:ncross
            u = -1.0 + 2.0 * i / ncross
            v = -1.0 + 2.0 * j / ncross
            my, mz = _square_to_disk(u, v)
            id = nid(i, j, k)
            nodes[1, id] = x; nodes[2, id] = Rx * my; nodes[3, id] = Rx * mz
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
    R0 = 6.413; L = 53.334               # Simo (1988) round-bar geometry
    ncross = 6                           # 6×6 = 36 cells per cross-section
    naxial = 40                          # 40 elements lengthwise (graded to the notch)
    mesh = necking_mesh(R0, L, ncross, naxial)

    # Simo saturation (Voce) hardening material.
    simo = J2Material(E = 206.9e3, ν = 0.29, σy0 = 450.0,
                      σsat = 715.0, δ = 16.93, Hiso = 129.24)

    model = Model(mesh, simo; element = :finite_fbar)

    # gripped tension: clamp x=0; pull x=L axially with lateral motion gripped
    fix!(model, on_face(mesh, :xmin))
    fix!(model, on_face(mesh, :xmax), :y)
    fix!(model, on_face(mesh, :xmax), :z)
    elong = 0.10 * L                     # 10% nominal elongation (well past neck onset)
    prescribe!(model, on_face(mesh, :xmax), :x, elong)

    res = solve!(model; nsteps = 50, tol = 1e-7, maxiter = 40)

    # --- postprocess ---
    u  = nodal_displacements(model)
    ᾱ  = equivalent_plastic_strain(model)

    # neck radius: min current cross-section radius over axial layers
    npc = ncross + 1; nlayer = npc * npc
    isbnd(i, j) = (i == 0 || i == ncross || j == 0 || j == ncross)
    neck_R = R0; neck_x = 0.0
    for k in 0:naxial
        rmax = 0.0
        for j in 0:ncross, i in 0:ncross
            isbnd(i, j) || continue
            nd = k * nlayer + j * npc + i + 1
            y = mesh.nodes[2, nd] + u[2, nd]; z = mesh.nodes[3, nd] + u[3, nd]
            rmax = max(rmax, sqrt(y^2 + z^2))
        end
        rmax < neck_R && (neck_R = rmax; neck_x = (k / naxial) * L)
    end

    # nominal axial-load indicator: mean axial Cauchy stress in a thin end slab
    # (near x=L). Not the true reaction, but tracks the load rise→drop signature.
    σ = gauss_stress(model)              # 6×ngp Cauchy (finite-strain model)
    endslab_sum = 0.0; nend = 0
    for e in 1:mesh.nelem
        kx = (e - 1) ÷ (ncross * ncross)   # axial element index
        if kx >= naxial - 2                # last few axial element layers
            for g in 1:8
                endslab_sum += σ[1, (e - 1) * 8 + g]; nend += 1
            end
        end
    end
    σ_end = endslab_sum / max(nend, 1)

    println("=== Simo round-bar necking (saturation/Voce hardening, F-bar) ===")
    @printf("mesh                 : %d×%d×%d  (%d elems, %d DOFs)\n",
            ncross, ncross, naxial, mesh.nelem, 3 * mesh.nnodes)
    println("material             : Simo Voce  E=206.9 GPa, ν=0.29, σy0=450, σ∞=715, δ=16.93, H=129.24")
    @printf("yield σy: ᾱ=0 → %.1f,  ᾱ=0.1 → %.1f,  ᾱ=0.5 → %.1f MPa\n",
            yield_stress(simo, 0.0), yield_stress(simo, 0.1), yield_stress(simo, 0.5))
    println("converged            : ", res.converged, "   (", length(res.iters), " load steps)")
    @printf("Newton iters/step    : min %d, max %d  (quadratic ⇒ nonlinear tangent OK)\n",
            minimum(res.iters), maximum(res.iters))
    @printf("elongation           : %.3f mm (%.1f%% of L)\n", elong, 100 * elong / L)
    @printf("max eq. plastic strn : %.4f  at x ≈ %.2f  (notch at L/2 = %.2f)\n",
            maximum(ᾱ), neck_x, L / 2)
    @printf("neck radius (min)    : %.4f at x ≈ %.2f  vs R0 = %.3f ⇒ %.1f%% reduction\n",
            neck_R, neck_x, R0, 100 * (1 - neck_R / R0))
    @printf("nominal end axial σ  : %.1f MPa (load indicator)\n", σ_end)

    println("\nwrote: ", write_vtu(joinpath(@__DIR__, "simo_necking_bar"), model))
    return res.converged
end

main()
