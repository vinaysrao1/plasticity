# Finite-strain large-rotation cantilever.
#
# A slender cantilever under a transverse tip load undergoes large rotation /
# deflection — a regime where small-strain kinematics give a qualitatively wrong
# (over-stiff, non-rotating) answer and finite strain is required. Run:
#   julia --project=. examples/finite_large_rotation_cantilever.jl
#
# Output: finite_large_rotation_cantilever.vtu — Warp By Vector to see the bent
# shape; compare against a small-strain run to see the geometric-stiffening effect.

using PlasticityFEM
using LinearAlgebra

L, h = 8.0, 1.0
mesh = box_mesh(L, h, h, 16, 2, 2)
mat = J2Material(E = 210e3, ν = 0.3, σy0 = 1e9)      # stays elastic ⇒ pure large rotation

function run(elem)
    model = Model(mesh, mat; element = elem)
    fix!(model, on_face(mesh, :xmin), :all)            # clamp the root
    load!(model, on_face(mesh, :xmax), :z, -400.0; distribute = true)
    res = solve!(model; nsteps = 10, tol = 1e-7, maxiter = 40, linsolve = :direct)
    return model, res
end

ms, _  = run(:small)
mf, rf = run(:finite)

δs = maximum(abs, nodal_displacements(ms)[3, :])
δf = maximum(abs, nodal_displacements(mf)[3, :])
println("finite converged: ", rf.converged)
println("tip deflection  small = ", round(δs, digits = 4),
        "   finite = ", round(δf, digits = 4),
        "   (geometric stiffening makes finite stiffer at large rotation)")

out = write_vtu("finite_large_rotation_cantilever", mf)
println("wrote ", out)
