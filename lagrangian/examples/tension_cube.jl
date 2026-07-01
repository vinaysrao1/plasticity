# Uniaxial tension cube into the plastic regime (DESIGN §6.1).
#
# A 1×1×1 cube under roller BCs (one symmetry plane per axis) pulled in +x by a
# prescribed displacement. The roller BCs produce a uniaxial *stress* state, so
# the axial stress follows the analytical isotropic-hardening curve (DESIGN T2):
#
#   σ_xx = σy0 + (E·Hiso/(E+Hiso))·(ε − σy0/E),   ε > ε_y = σy0/E.

using PlasticityFEM

# 1) Mesh: single Hex8 cube
mesh = box_mesh(1.0, 1.0, 1.0, 1, 1, 1)

# 2) Material: steel-like with linear isotropic hardening (MPa, mm units)
steel = J2Material(E = 210e3, ν = 0.3, σy0 = 250.0, Hiso = 1000.0)

# 3) Model
model = Model(mesh, steel)

# 4) Boundary conditions by face predicates (rollers on the three min faces)
fix!(model, on_face(mesh, :xmin), :x)
fix!(model, on_face(mesh, :ymin), :y)
fix!(model, on_face(mesh, :zmin), :z)

# 5) Loading: prescribe x-displacement on the xmax face (1% nominal strain)
ε_target = 0.01
prescribe!(model, on_face(mesh, :xmax), :x, ε_target)

# 6) Solve with load stepping
result = solve!(model; nsteps = 20, tol = 1e-8, maxiter = 25)

# 7) Postprocess
σ = gauss_stress(model)
σxx = σ[1, 1]

E = steel.E; σy0 = steel.σy0; Hiso = steel.Hiso
εy = σy0 / E
Ht = E * Hiso / (E + Hiso)
σ_analytic = σy0 + Ht * (ε_target - εy)

println("converged        : ", result.converged)
println("iters per step   : ", result.iters)
println("σ_xx (FEM)       : ", σxx)
println("σ_xx (analytic)  : ", σ_analytic)
println("rel error        : ", abs(σxx - σ_analytic) / σ_analytic)
println("ᾱ (eq. pl. strn) : ", equivalent_plastic_strain(model)[1])

# 8) Export the stress/strain fields for ParaView (open tension_cube.vtu)
vtu = write_vtu(joinpath(@__DIR__, "tension_cube"), model)
println("wrote            : ", vtu)
