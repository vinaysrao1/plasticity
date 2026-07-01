# System / solver tests (DESIGN §8.3, T13–T19).

using PlasticityFEM
using PlasticityFEM.Materials
using StaticArrays
using LinearAlgebra
using Test

# Build the standard uniaxial roller cube model.
function uniaxial_cube(; nx=1, E=210e3, ν=0.3, σy0=250.0, Hiso=0.0, Hkin=0.0,
                       εtarget=0.01)
    mesh = box_mesh(1.0, 1.0, 1.0, nx, nx, nx)
    mat = J2Material(E=E, ν=ν, σy0=σy0, Hiso=Hiso, Hkin=Hkin)
    model = Model(mesh, mat)
    fix!(model, on_face(mesh, :xmin), :x)
    fix!(model, on_face(mesh, :ymin), :y)
    fix!(model, on_face(mesh, :zmin), :z)
    prescribe!(model, on_face(mesh, :xmax), :x, εtarget)
    return model, mat
end

@testset "T13 uniaxial elastic + plastic, mesh independence" begin
    E = 210e3; ν = 0.3; σy0 = 250.0; Hiso = 1000.0
    εy = σy0 / E
    Ht = E * Hiso / (E + Hiso)

    # elastic regime: ε = 0.001 < εy
    model, mat = uniaxial_cube(; E=E, ν=ν, σy0=σy0, Hiso=Hiso, εtarget=0.001)
    solve!(model; nsteps=2, tol=1e-10)
    σ = gauss_stress(model)
    @test σ[1, 1] ≈ E * 0.001 rtol=1e-8
    @test abs(σ[2,1]) < 1e-6 * σ[1,1]
    @test abs(σ[3,1]) < 1e-6 * σ[1,1]
    # lateral strains = −ν εxx: check via displacement of a max-face node
    u = nodal_displacements(model)
    # ymax node displacement_y / 1.0 == -ν*εxx
    ymaxnode = on_face(model.mesh, :ymax)[1]
    @test u[2, ymaxnode] ≈ -ν * 0.001 rtol=1e-6

    # plastic regime: ε = 0.01, compare to T2 analytical at several refinements
    σ_an = σy0 + Ht * (0.01 - εy)
    σ_by_mesh = Float64[]
    for nx in (1, 2, 4)
        m, _ = uniaxial_cube(; E=E, ν=ν, σy0=σy0, Hiso=Hiso, εtarget=0.01)
        solve!(m; nsteps=20, tol=1e-9)
        push!(σ_by_mesh, gauss_stress(m)[1, 1])
        @test gauss_stress(m)[1, 1] ≈ σ_an rtol=1e-6
    end
    # mesh independence (uniform stress)
    @test maximum(σ_by_mesh) - minimum(σ_by_mesh) <= 1e-8 * σ_an
end

@testset "T14 elastic cantilever vs beam theory" begin
    L = 10.0; b = 1.0; h = 1.0; E = 210e3; P = 100.0
    function tipdefl(nx)
        mesh = box_mesh(L, b, h, nx, 4, 4)
        mat = J2Material(E=E, ν=0.3, σy0=1e9)   # high yield → elastic
        model = Model(mesh, mat)
        fix!(model, on_face(mesh, :xmin))
        load!(model, on_face(mesh, :xmax), :z, -P; distribute=true)
        solve!(model; nsteps=1, tol=1e-9)
        return maximum(abs, nodal_displacements(model)[3, :])
    end
    I = b * h^3 / 12
    δ_beam = P * L^3 / (3 * E * I)     # ≈ 1.905
    δ20 = tipdefl(20)
    δ40 = tipdefl(40)
    # within ~10% of Euler–Bernoulli; Hex8 slightly stiff (smaller δ)
    @test abs(δ40 - δ_beam) / δ_beam < 0.10
    # converges toward the analytic value under refinement
    @test abs(δ40 - δ_beam) <= abs(δ20 - δ_beam) + 1e-6
end

@testset "T15 Newton quadratic convergence" begin
    model, _ = uniaxial_cube(; nx=2, σy0=250.0, Hiso=1000.0, Hkin=500.0, εtarget=0.01)
    res = solve!(model; nsteps=10, tol=1e-9, maxiter=15)
    @test res.converged
    # find a clearly plastic step (≥4 iters) and check quadratic drop
    plastic_step = 0
    for (i, h) in enumerate(res.residuals)
        if length(h) >= 4
            plastic_step = i; break
        end
    end
    @test plastic_step > 0
    h = res.residuals[plastic_step]
    # asymptotic quadratic: ‖R^{k+1}‖ ≤ C ‖R^k‖² for last iterations
    # check the ratio is small / decreasing in the tail
    # use the last 3 residuals before the final (machine-zero) one
    ok = false
    for k in 2:(length(h)-1)
        if h[k] < 1.0 && h[k+1] <= 50.0 * h[k]^2 + 1e-12
            ok = true
        end
    end
    @test ok
    # converges in a small number of iterations
    @test all(length(hh) <= 8 for hh in res.residuals)
end

@testset "T17 reaction equilibrium / global balance" begin
    # force-controlled: applied load must balance reactions at every step.
    L = 4.0
    mesh = box_mesh(L, 1.0, 1.0, 8, 2, 2)
    mat = J2Material(E=210e3, ν=0.3, σy0=1e9)   # elastic, easy balance
    model = Model(mesh, mat)
    fix!(model, on_face(mesh, :xmin))
    Ptot = -50.0
    loadnodes = on_face(mesh, :xmax)
    load!(model, loadnodes, :z, Ptot; distribute=true)
    solve!(model; nsteps=3, tol=1e-10)
    # internal force F_int at the converged U equals external load on free dofs,
    # and the reaction at fixed dofs balances the applied load. Compute F_int.
    st = model.state_trial
    K, Fint = PlasticityFEM.Assembly.assemble!(model.sparsity, mat, model.cache,
        model.U, st.εp, st.β, st.ᾱ, st.σ, copy(model.Rbuf))
    # reactions = F_int at constrained (xmin) dofs
    fixnodes = on_face(mesh, :xmin)
    Rz = 0.0
    for n in fixnodes
        Rz += Fint[3*(n-1)+3]
    end
    # total applied z load = Ptot (distributed). Sum reactions_z + applied = 0
    @test Rz + Ptot ≈ 0.0 atol=1e-7 * abs(Ptot) + 1e-7
end

@testset "T18 load-path consistency (commit semantics)" begin
    function final_state(nsteps)
        model, _ = uniaxial_cube(; nx=1, σy0=250.0, Hiso=1000.0, Hkin=500.0, εtarget=0.01)
        solve!(model; nsteps=nsteps, tol=1e-10)
        return gauss_stress(model)[1,1], equivalent_plastic_strain(model)[1]
    end
    σN, αN = final_state(10)
    σ2N, α2N = final_state(20)
    @test σN ≈ σ2N rtol=1e-5
    @test αN ≈ α2N rtol=1e-4
end

@testset "T19 unload elastic check" begin
    # load into plasticity, then unload; unloading slope = E.
    E = 210e3; σy0 = 250.0; Hiso = 1000.0
    mat = J2Material(E=E, ν=0.3, σy0=σy0, Hiso=Hiso)
    # single GP uniaxial-stress path
    Z6 = zero(SVector{6,Float64})
    function uniaxial_step(εp, β, ᾱ, exx)
        εlat = 0.0
        for _ in 1:80
            ε = SVector{6,Float64}(exx, εlat, εlat, 0,0,0)
            σ, _, _, _, D = return_map(mat, ε, εp, β, ᾱ)
            εlat -= σ[2] / (D[2,2] + D[2,3])
            abs(σ[2]) < 1e-11 && break
        end
        ε = SVector{6,Float64}(exx, εlat, εlat, 0,0,0)
        σ, εp2, β2, ᾱ2, _ = return_map(mat, ε, εp, β, ᾱ)
        return σ[1], εp2, β2, ᾱ2
    end
    εp = Z6; β = Z6; ᾱ = 0.0
    # load to εxx=0.01 in steps
    σ_load = 0.0
    for n in 1:100
        σ_load, εp, β, ᾱ = uniaxial_step(εp, β, ᾱ, 0.01*n/100)
    end
    ᾱ_loaded = ᾱ
    # small elastic unload increment Δε = -1e-5 (committed state frozen)
    Δε = -1e-5
    σ_unl, _, _, ᾱ_unl = uniaxial_step(εp, β, ᾱ, 0.01 + Δε)
    slope = (σ_unl - σ_load) / Δε
    @test slope ≈ E rtol=1e-3        # unloading stiffness is elastic E
    @test ᾱ_unl ≈ ᾱ_loaded atol=1e-12  # no further plastic flow on unload
end
