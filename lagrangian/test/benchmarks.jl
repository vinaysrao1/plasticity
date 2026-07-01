# =============================================================================
# benchmarks.jl — performance regression baselines (NOT asserted, recorded).
#
# Run with:  jl --project=. test/benchmarks.jl
#
# Prints @btime / @timed numbers for the hot kernels and a global assemble, plus
# allocation and scaling tables. These are baselines for regression tracking,
# per DESIGN §7.1; they are informational and not part of the pass/fail suite.
# Falls back to manual timing if BenchmarkTools is unavailable.
# =============================================================================

using PlasticityFEM
using PlasticityFEM.Materials
using PlasticityFEM.Elements
using PlasticityFEM.Assembly
using StaticArrays
using SparseArrays
using LinearAlgebra

const HAVE_BT = try
    @eval using BenchmarkTools
    true
catch
    @warn "BenchmarkTools not available; using manual timing"
    false
end

function bench(f, args...; reps=10_000)
    f(args...)  # warmup
    if HAVE_BT
        b = @benchmark $f($(args)...) samples=2000 evals=1
        return minimum(b).time, minimum(b).memory   # ns, bytes
    else
        best = Inf
        for _ in 1:5
            t0 = time_ns()
            for _ in 1:reps; f(args...); end
            best = min(best, (time_ns() - t0) / reps)
        end
        return best, (@allocated f(args...))
    end
end

println("="^70)
println("PlasticityFEM performance baselines")
println("="^70)

# --- return_map ------------------------------------------------------------
let
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)
    Z = zero(SVector{6,Float64})
    εpl = SVector{6,Float64}(0.01, 0, 0, 0, 0, 0)
    t, m = bench(return_map, mat, εpl, Z, Z, 0.0)
    println("return_map (plastic):            $(round(t, digits=1)) ns,  $m bytes")
end

# --- element_force_tangent! ------------------------------------------------
let
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)
    mesh = box_mesh(1.0, 1.0, 1.0, 1, 1, 1)
    cache = precompute_cache(mesh.nodes, mesh.elements)
    εp = zeros(6, 8); β = zeros(6, 8); ᾱ = zeros(8); σ = zeros(6, 8)
    ue = SVector{24,Float64}(ntuple(i -> 1e-3 * i, 24))
    B1, Jw1 = element_geometry(cache, 1)
    f() = element_force_tangent!(mat, B1, Jw1, ue, εp, β, ᾱ, 1, σ, Val(false))
    t, m = bench(f)
    println("element_force_tangent! (1 elem): $(round(t, digits=1)) ns,  $m bytes")
end

# --- global assemble! over a sweep -----------------------------------------
println("-"^70)
println("assemble! scaling (one global assembly):")
println(rpad("nelem", 10), rpad("ndof", 10), rpad("nnz(K)", 12),
        rpad("time [µs]", 14), rpad("alloc [B]", 12), "ns/elem")
for nx in (4, 8, 10, 12)
    mesh = box_mesh(1.0, 1.0, 1.0, nx, nx, nx)
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0)
    model = Model(mesh, mat)
    U = zeros(3 * mesh.nnodes); st = model.state_trial
    f() = assemble!(model.sparsity, mat, model.cache, U, st.εp, st.β, st.ᾱ, st.σ, model.Rbuf)
    t, m = bench(f; reps=200)
    println(rpad(mesh.nelem, 10), rpad(3 * mesh.nnodes, 10),
            rpad(nnz(model.sparsity.K), 12),
            rpad(round(t / 1e3, digits=2), 14), rpad(m, 12),
            round(t / mesh.nelem, digits=1))
end

# --- one Newton iteration on a 10×10×10 plastic mesh -----------------------
println("-"^70)
let
    mesh = box_mesh(1.0, 1.0, 1.0, 10, 10, 10)
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)
    model = Model(mesh, mat)
    fix!(model, on_face(mesh, :xmin), :x)
    fix!(model, on_face(mesh, :ymin), :y)
    fix!(model, on_face(mesh, :zmin), :z)
    prescribe!(model, on_face(mesh, :xmax), :x, 0.01)
    r = @timed solve!(model; nsteps=5, tol=1e-8, maxiter=20)
    println("10×10×10 plastic solve (5 steps): $(round(r.time, digits=3)) s, ",
            "$(round(r.bytes/1e6, digits=1)) MB, iters/step = ", r.value.iters)
end
println("="^70)
