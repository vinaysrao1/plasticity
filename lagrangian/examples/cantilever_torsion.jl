# Torsion of a cantilever to first yield (companion to cantilever.jl).
#
# Same 10×1×1 cantilever, clamped at x=0, but loaded by a TORQUE about its own
# long axis (the x-axis) — i.e. the bar is twisted. This produces a much richer
# stress/strain field than bending: pure St-Venant torsion of a SQUARE shaft has
# shear stress that is maximal at the MID-POINTS of the four cross-section edges,
# ZERO at the centre and at the four corners, with out-of-plane "warping" of the
# cross-sections; the clamped end additionally restrains warping and concentrates
# stress near the root. Yielding therefore initiates as four lobes on the side
# faces (not the corners), spreading inward — a genuinely 3D elastoplastic field.
#
# Analytic check (St-Venant, square side a=1): the surface first yields at
#   T_yield ≈ 0.208·a³·τ_y ,  τ_y = σy0/√3   ⇒   T_yield ≈ 0.208·250/√3 ≈ 30.0.
# We ramp to T = 36 (~1.2× T_yield) so the onset and a small plastic zone are
# clearly captured while strains stay small (small-strain theory valid).
#
# Mesh is 160×16×16 (≈40.96k elements, ≈140k DOFs) — solved by the CG+AMG
# iterative solver. Run me with threads, e.g.:
#   JULIA_NUM_THREADS=4 julia --project=. examples/cantilever_torsion.jl

using PlasticityFEM

function main()
    # --- geometry & material --------------------------------------------------
    L, b, h = 10.0, 1.0, 1.0
    mesh = box_mesh(L, b, h, 160, 16, 16)
    mat  = J2Material(E = 210e3, ν = 0.3, σy0 = 250.0, Hiso = 1000.0)  # mild-steel-like
    model = Model(mesh, mat)

    # --- boundary conditions: clamp the x = 0 face ----------------------------
    fix!(model, on_face(mesh, :xmin))

    # --- loading: a pure torque T about the x-axis on the x = L face ----------
    # Applied as tangential nodal forces in the cross-section (y,z) plane. For a
    # node at offset (Δy,Δz) from the section centroid, the torque-producing force
    # is perpendicular to the radius: (Fy, Fz) = k·(−Δz, Δy). Then the net moment
    # about x is M_x = Σ(Δy·Fz − Δz·Fy) = k·Σ(Δy²+Δz²), with zero net force (by
    # symmetry), so k = T / Σ(Δy²+Δz²) applies exactly the torque T (ramped).
    T_total = 36.0                       # target torque (~1.2× the ≈30 yield torque)
    yc, zc  = 0.5b, 0.5h                  # section centroid
    endface = on_face(mesh, :xmax)

    S = 0.0
    for n in endface
        dy = mesh.nodes[2, n] - yc; dz = mesh.nodes[3, n] - zc
        S += dy * dy + dz * dz
    end
    k = T_total / S
    for n in endface
        dy = mesh.nodes[2, n] - yc; dz = mesh.nodes[3, n] - zc
        load!(model, [n], :y, -k * dz)
        load!(model, [n], :z,  k * dy)
    end

    # --- solve (load-stepped Newton + CG/AMG) ---------------------------------
    res = solve!(model; nsteps = 30, tol = 1e-8, maxiter = 60)

    # --- postprocess ----------------------------------------------------------
    u   = nodal_displacements(model)
    σ   = gauss_stress(model)
    ᾱ   = equivalent_plastic_strain(model)
    ngp = length(ᾱ)
    σvm_max = maximum(von_mises(@view σ[:, i]) for i in 1:ngp)

    # end-face twist angle θ (rigid-rotation fit: u_y=−θΔz, u_z=θΔy ⇒
    # θ ≈ Σ(Δy·u_z − Δz·u_y) / Σr²)
    num = 0.0; den = 0.0
    for n in endface
        dy = mesh.nodes[2, n] - yc; dz = mesh.nodes[3, n] - zc
        num += dy * u[3, n] - dz * u[2, n]
        den += dy * dy + dz * dz
    end
    twist = num / den          # radians of twist of the free end

    println("=== torsion of a 10×1×1 cantilever to first yield ===")
    println("mesh                 : 160×16×16  ($(mesh.nelem) elems, $(3*mesh.nnodes) DOFs)")
    println("applied torque T     : ", T_total, "   (analytic yield torque ≈ 30.0)")
    println("converged            : ", res.converged)
    println("end-face twist (rad) : ", twist)
    println("max von Mises stress : ", σvm_max, "   (yield σy0 = 250)")
    println("max eq. plastic strn : ", maximum(ᾱ))
    println("yielded Gauss points : ", count(>(0.0), ᾱ), " / ", ngp)

    # --- export the (complex) stress/strain field for ParaView ----------------
    # Colour by VonMises or EqPlasticStrain to see the four side-face yield lobes;
    # Warp By Vector on Displacement to see the twist.
    println("\nwrote: ", write_vtu(joinpath(@__DIR__, "cantilever_torsion"), model))
    return res.converged
end

main()
