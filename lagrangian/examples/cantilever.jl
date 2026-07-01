# End-loaded cantilever beam (DESIGN §6.2).
#
# A slender beam clamped at x=0 with a distributed tip load (total Fz = −P) on
# the x=L face. For a small (elastic) load the tip deflection matches Euler–
# Bernoulli beam theory (DESIGN T14):
#
#   δ = P L³ / (3 E I),   I = b h³ / 12.
#
# Hex8 with shear adds a small Timoshenko correction, so the FEM value is a few
# percent larger than the Euler–Bernoulli target.

using PlasticityFEM

L = 10.0; b = 1.0; h = 1.0
E = 210e3
I = b * h^3 / 12
P = 100.0

# --- Part 1: elastic cantilever vs Euler–Bernoulli (DESIGN T14) ---------------
# With P=100 the max bending stress σ_max = (P L)(h/2)/I ≈ 6000 MPa, so we need a
# high yield stress to keep the response elastic (the regime T14 targets).
mesh  = box_mesh(L, b, h, 40, 4, 4)
elastic = J2Material(E = E, ν = 0.3, σy0 = 1.0e6)   # high yield → stays elastic
model = Model(mesh, elastic)

fix!(model, on_face(mesh, :xmin))                    # clamp the x=0 face
load!(model, on_face(mesh, :xmax), :z, -P; distribute = true)  # total Fz = −P

result = solve!(model; nsteps = 10, tol = 1e-8, maxiter = 25)

δtip   = maximum(abs, nodal_displacements(model)[3, :])
δ_beam = P * L^3 / (3 * E * I)

println("=== elastic cantilever (T14) ===")
println("converged       : ", result.converged)
println("iters per step  : ", result.iters)
println("δtip (FEM)      : ", δtip)
println("δ  (Euler-Bern) : ", δ_beam)
println("rel diff        : ", abs(δtip - δ_beam) / δ_beam)

# --- Part 2: same beam driven into (partial) plasticity -----------------------
# A modest load so only the highly-stressed root fibres yield while strains stay
# small (small-strain theory stays valid). The nominal max bending stress is
# σ_max ≈ (Pp·L)(h/2)/I = Pp·60; with σy0 = 250 a load Pp = 8 gives σ_max ≈ 480,
# ~1.9× yield — partial yielding near the clamp, small plastic strains, and a tip
# deflection modestly above the linear-elastic prediction for the same load.
Pp      = 8.0
plastic = J2Material(E = E, ν = 0.3, σy0 = 250.0, Hkin = 2000.0)
pmodel  = Model(mesh, plastic)

fix!(pmodel, on_face(mesh, :xmin))
load!(pmodel, on_face(mesh, :xmax), :z, -Pp; distribute = true)

pres   = solve!(pmodel; nsteps = 20, tol = 1e-8, maxiter = 50)
δp     = maximum(abs, nodal_displacements(pmodel)[3, :])
ᾱmax   = maximum(equivalent_plastic_strain(pmodel))
ngp    = length(equivalent_plastic_strain(pmodel))
δp_lin = Pp * L^3 / (3 * E * I)     # linear-elastic deflection at the same load

println("\n=== elastoplastic cantilever (Pp = ", Pp, ") ===")
println("converged            : ", pres.converged)
println("δtip (FEM)           : ", δp, "  vs linear-elastic ", δp_lin)
println("max eq. plastic strn : ", ᾱmax)
println("yielded Gauss pts    : ", count(>(0.0), equivalent_plastic_strain(pmodel)), " / ", ngp)

# Export both fields for ParaView. The plastic case shows the yielded zone near
# the clamp (color by VonMises or EqPlasticStrain; Warp By Vector on Displacement).
println("\nwrote: ", write_vtu(joinpath(@__DIR__, "cantilever_elastic"), model))
println("wrote: ", write_vtu(joinpath(@__DIR__, "cantilever_plastic"), pmodel))
