# Parameterized elastoplastic cantilever for mesh-refinement studies.
#   julia --project=. examples/cantilever_refine.jl NX NY NZ
# Defaults to 80 8 8 if no args are given.

using PlasticityFEM

L = 10.0; b = 1.0; h = 1.0
E = 210e3
I = b * h^3 / 12

nx = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 80
ny = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
nz = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8

mesh = box_mesh(L, b, h, nx, ny, nz)
println("mesh: ", nx, "×", ny, "×", nz, " = ", mesh.nelem, " elements, ",
        mesh.nnodes, " nodes (", 3 * mesh.nnodes, " DOFs)")
flush(stdout)

Pp      = 8.0
plastic = J2Material(E = E, ν = 0.3, σy0 = 250.0, Hkin = 2000.0)
pmodel  = Model(mesh, plastic)

fix!(pmodel, on_face(mesh, :xmin))                                 # clamp x=0
load!(pmodel, on_face(mesh, :xmax), :z, -Pp; distribute = true)    # total Fz = −Pp

t = @elapsed pres = solve!(pmodel; nsteps = 20, tol = 1e-8, maxiter = 50)

δp     = maximum(abs, nodal_displacements(pmodel)[3, :])
ᾱmax   = maximum(equivalent_plastic_strain(pmodel))
ngp    = length(equivalent_plastic_strain(pmodel))
δp_lin = Pp * L^3 / (3 * E * I)

println("\n=== elastoplastic cantilever ", nx, "×", ny, "×", nz, " (Pp = ", Pp, ") ===")
println("converged            : ", pres.converged)
println("solve time (s)       : ", round(t, digits = 2))
println("iters per step       : ", pres.iters)
println("δtip (FEM)           : ", δp, "  vs linear-elastic ", δp_lin)
println("max eq. plastic strn : ", ᾱmax)
println("yielded Gauss pts    : ", count(>(0.0), equivalent_plastic_strain(pmodel)), " / ", ngp)

tag = string(nx, "x", ny, "x", nz)
println("\nwrote: ", write_vtu(joinpath(@__DIR__, "cantilever_plastic_" * tag), pmodel))
flush(stdout)
