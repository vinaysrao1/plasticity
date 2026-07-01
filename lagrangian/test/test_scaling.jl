# Scaling-redesign tests (SCALING.md §6). Phase-gated:
#   Phase 1 — C1 (CG+AMG == direct), C3 (inexact Newton keeps the Newton tail).
#   Phase 2 — allocation gate, nnz/N flat, Int32 indices, no COO transient.
#   Phase 3 — 1-vs-4-thread result equality (race-free), threaded SpMV correctness.
#
# These tests use moderate meshes that run in the 15 GB sandbox; the 10M / 64 GB
# target is validated separately by extrapolation.

using PlasticityFEM
using PlasticityFEM.Materials
using PlasticityFEM.Assembly
using SparseArrays
using LinearAlgebra
using Test

# Standard uniaxial roller cube (mirrors test_solver.jl) with a chosen linsolve.
function _scaling_cube(nx; E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=0.0,
                       εtarget=0.01)
    mesh = box_mesh(1.0, 1.0, 1.0, nx, nx, nx)
    mat = J2Material(E=E, ν=ν, σy0=σy0, Hiso=Hiso, Hkin=Hkin)
    model = Model(mesh, mat)
    fix!(model, on_face(mesh, :xmin), :x)
    fix!(model, on_face(mesh, :ymin), :y)
    fix!(model, on_face(mesh, :zmin), :z)
    prescribe!(model, on_face(mesh, :xmax), :x, εtarget)
    return model
end

@testset "C1 CG+AMG matches direct solve (displacements & stresses)" begin
    # Solve the SAME plastic problem with the direct (UMFPACK) and CG+AMG paths
    # on a ~16³ mesh and assert the answers agree to ~10·rtol (SCALING.md §6.1).
    rtol = 1e-8
    nx = 16

    md = _scaling_cube(nx)
    solve!(md; nsteps=12, tol=1e-9, linsolve=:direct)
    Ud = copy(md.U)
    σd = copy(gauss_stress(md))

    mc = _scaling_cube(nx)
    solve!(mc; nsteps=12, tol=1e-9, linsolve=:cg, amg=:sa)
    Uc = copy(mc.U)
    σc = copy(gauss_stress(mc))

    uerr = norm(Uc - Ud) / max(norm(Ud), eps())
    serr = norm(σc - σd) / max(norm(σd), eps())
    @info "C1 match" uerr serr
    @test uerr <= 10 * rtol
    @test serr <= 10 * rtol

    # Ruge–Stüben fallback also matches.
    mr = _scaling_cube(nx)
    solve!(mr; nsteps=12, tol=1e-9, linsolve=:cg, amg=:rs)
    @test norm(mr.U - Ud) / norm(Ud) <= 10 * rtol
end

@testset "C3 inexact Newton preserves the quadratic Newton tail (analog of T15)" begin
    # Same as T15 but through the CG+AMG inexact-Newton path: the last residuals
    # of a plastic step must still drop quadratically and the step must converge
    # in a small number of Newton iterations (SCALING.md §1.4, §6.1).
    model = _scaling_cube(2; σy0=250.0, Hiso=1000.0, Hkin=500.0, εtarget=0.01)
    res = solve!(model; nsteps=10, tol=1e-9, maxiter=15, linsolve=:cg)
    @test res.converged

    plastic_step = 0
    for (i, h) in enumerate(res.residuals)
        if length(h) >= 4
            plastic_step = i; break
        end
    end
    @test plastic_step > 0
    h = res.residuals[plastic_step]
    # quadratic tail: ‖R^{k+1}‖ ≤ C ‖R^k‖² for some tail iteration
    ok = false
    for k in 2:(length(h) - 1)
        if h[k] < 1.0 && h[k+1] <= 50.0 * h[k]^2 + 1e-12
            ok = true
        end
    end
    @info "C3 plastic residual tail" plastic_step h
    @test ok
    # small Newton iteration count preserved under inexact CG
    @test all(length(hh) <= 8 for hh in res.residuals)
end

@testset "C2 mesh-independent CG count (near-null-space SA-AMG)" begin
    # THE decisive O(N)-flops check (SCALING.md §4, §6.3): with the rigid-body
    # near-null-space, the CG iteration count must be ~flat in N. Without it,
    # scalar SA-AMG grows ~N^0.3 (→ O(N^1.3)). We sweep elastic cubes and fit the
    # max CG-iters-per-Newton-step vs N in log-log; the exponent must be small.
    Ns = Int[]; maxcg = Int[]
    for n in (8, 12, 16, 20)
        m = _scaling_cube(n; σy0=1e9, εtarget=1e-3)   # elastic ⇒ clean CG counts
        res = solve!(m; nsteps=1, tol=1e-9, linsolve=:cg, amg=:sa, threaded=false)
        push!(Ns, 3 * m.mesh.nnodes)
        push!(maxcg, maximum(res.cg_iters[1]))
    end
    # log-log slope of max_cg vs N
    lx = log.(Ns); ly = log.(maxcg)
    x̄ = sum(lx)/length(lx); ȳ = sum(ly)/length(ly)
    slope = sum((lx .- x̄) .* (ly .- ȳ)) / sum((lx .- x̄).^2)
    @info "C2 CG-iter scaling" Ns maxcg slope
    @test slope <= 0.15            # ~flat: near-null-space delivers O(N) flops
    @test maximum(maxcg) <= 25     # absolute sanity: few iterations at every size
end

@testset "Phase 2: index type, nnz/N stencil bound, O(1) assemble alloc" begin
    # Default index type is Int (Int64): the SA-AMG near-null-space preconditioner
    # needs Int64 hierarchies, so one shared Int64 K is leaner than Int32+copy
    # (SCALING.md §2.5). Int32 remains forceable via the Ti kwarg.
    sp = build_sparsity(box_mesh(1.0, 1.0, 1.0, 4, 4, 4))
    @test eltype(sp.K.rowval) == Int
    @test eltype(sp.K.colptr) == Int
    @test eltype(build_sparsity(box_mesh(1.0,1.0,1.0,2,2,2); Ti=Int32).K.rowval) == Int32

    # nnz/N is bounded by the 81-entry interior Hex8 stencil and approaches it
    # under refinement (boundary fraction shrinks) — i.e. nnz = O(N) with the
    # mesh-independent constant 81 (SCALING.md §0.1, §6.3).
    ratios = Float64[]
    for n in (4, 8, 12)
        spn = build_sparsity(box_mesh(1.0, 1.0, 1.0, n, n, n))
        push!(ratios, nnz(spn.K) / size(spn.K, 1))
    end
    @info "Phase2 nnz/N" ratios
    @test all(r -> r <= 81.0 + 1e-9, ratios)   # never exceeds the interior stencil
    @test issorted(ratios)                      # grows toward 81 with refinement
    @test ratios[end] > ratios[1]

    # assemble! allocation is a bounded constant independent of nelem (the
    # on-the-fly CSC scatter must not allocate per element — preserves T20).
    function asm_alloc(n)
        m = _scaling_cube(n; σy0=1e9)
        st = m.state_trial
        assemble!(m.sparsity, m.material, m.cache, m.U, st.εp, st.β, st.ᾱ, st.σ, m.Rbuf)
        return @allocated assemble!(m.sparsity, m.material, m.cache, m.U,
                                    st.εp, st.β, st.ᾱ, st.σ, m.Rbuf)
    end
    a4 = asm_alloc(4); a8 = asm_alloc(8)   # 64 vs 512 elements
    @info "Phase2 assemble! alloc" a4 a8
    @test a8 <= a4 + 1024                   # bounded constant, not O(nelem)
end

@testset "Phase 3: threaded assembly == serial; threaded SpMV == K*x" begin
    m = _scaling_cube(6; σy0=1e9)
    # impose a nontrivial displacement so K and R are both nonzero
    m.U .= [1e-4 * sin(0.7 * i) for i in 1:length(m.U)]
    st = m.state_trial

    Ks, Rs = assemble!(m.sparsity, m.material, m.cache, m.U,
                       st.εp, st.β, st.ᾱ, st.σ, m.Rbuf)
    Ks_nz = copy(Ks.nzval); Rs_c = copy(Rs)

    # threaded 8-color assembly into the same pattern — race-free, so the result
    # must match the serial assembly to round-off (different summation order).
    Kt, Rt = assemble_threaded!(m.sparsity, m.material, m.cache, m.U,
                                st.εp, st.β, st.ᾱ, st.σ, m.Rbuf)
    @info "Phase3 threaded vs serial" nthreads=Threads.nthreads() dK=norm(Kt.nzval - Ks_nz) dR=norm(Rt - Rs_c)
    @test norm(Kt.nzval - Ks_nz) <= 1e-12 * (norm(Ks_nz) + 1)
    @test norm(Rt - Rs_c) <= 1e-12 * (norm(Rs_c) + 1)

    # row-parallel symmetric SpMV (K = Kᵀ) matches the plain sparse product
    A = PlasticityFEM.Solver.SymThreadedK(Ks)
    x = Float64[sin(0.3 * i) for i in 1:size(Ks, 2)]
    @test norm(A * x - Ks * x) <= 1e-10 * norm(Ks * x)
end
