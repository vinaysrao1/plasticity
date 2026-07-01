# =============================================================================
# scaling_validation.jl — independent V&V + performance validation for the
# Route-A (assembled CG+AMG) scaling of PlasticityFEM toward 10M DOFs / 64 GB.
#
# Standalone, re-runnable. NOT part of the default unit suite (heavy sweeps).
# Run:  JULIA_NUM_THREADS=4 jl --project=. test/scaling_validation.jl
#       JULIA_NUM_THREADS=4 jl --project=. test/scaling_validation.jl 56  (max n)
#
# Validates, with NUMBERS:
#   A. Correctness  — CG+AMG vs direct (elastic & plastic), threaded==serial,
#                     uniaxial closed form, equilibrium.
#   B. O(flops)     — flat CG-iters vs N (WITH vs WITHOUT rigid-body nullspace +
#                     Jacobi control), total-time exponent, flops/DOF.
#   C. Memory       — bytes/DOF flatness, nnz/N≈81, AMG operator complexity,
#                     build-time transient, assemble! allocation gate.
#   D. Deliverable  — extrapolated 10M-on-64GB line-item budget (<64 GB assert)
#                     and projected wall-time / load step.
# Prints clean tables; ends with a PASS/FAIL summary.
# =============================================================================

import PlasticityFEM
using PlasticityFEM: box_mesh, on_face, J2Material, Model, fix!, prescribe!, load!,
                     solve!, gauss_stress, nodal_displacements
using PlasticityFEM.Assembly: assemble!, assemble_threaded!, build_sparsity
using PlasticityFEM.Materials: return_map
using PlasticityFEM.Elements: element_force_tangent!
using AlgebraicMultigrid
using Krylov
using SparseArrays
using StaticArrays
using LinearAlgebra
using Printf

const SOLVER = PlasticityFEM.Solver
const BC     = PlasticityFEM.BoundaryConditions
const NMAX   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 48   # max elements/side
const TARGET_N = 1.033e7                                      # 150^3 box, 3 DOF/node
const TARGET_NGP = 2.70e7
const BUDGET_GB = 64.0

# ---------- helpers ----------------------------------------------------------

# Standard uniaxial roller cube (mirrors test_solver.jl / test_scaling.jl).
function cube(nx; E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=0.0, εtarget=0.01)
    mesh = box_mesh(1.0, 1.0, 1.0, nx, nx, nx)
    mat = J2Material(E=E, ν=ν, σy0=σy0, Hiso=Hiso, Hkin=Hkin)
    model = Model(mesh, mat)
    fix!(model, on_face(mesh, :xmin), :x)
    fix!(model, on_face(mesh, :ymin), :y)
    fix!(model, on_face(mesh, :zmin), :z)
    prescribe!(model, on_face(mesh, :xmax), :x, εtarget)
    return model
end

function loglog_slope(x, y)
    lx = log.(float.(x)); ly = log.(float.(y))
    x̄ = sum(lx)/length(lx); ȳ = sum(ly)/length(ly)
    return sum((lx .- x̄).*(ly .- ȳ)) / sum((lx .- x̄).^2)
end

# Assemble + impose Dirichlet to reproduce the exact operator CG solves, return K.
function imposed_operator(m)
    st = m.state_trial
    K, R = assemble!(m.sparsity, m.material, m.cache, m.U, st.εp,st.β,st.ᾱ,st.σ, m.Rbuf)
    dr, df, _ = SOLVER._freeze_bcs(m)
    BC.impose_dirichlet!(K, R, dr, 1.0, m.U)
    BC.impose_dirichlet!(K, R, df, 1.0, m.U)
    return K, R
end

# Full resident bytes of the AMG hierarchy: operator A on every level + final_A
# + prolongation P + restriction R (nzval 8B + index arrays). Returns (bytes, OC, levels).
function amg_hierarchy_bytes(K::SparseMatrixCSC, nullspace)
    ml = nullspace === nothing ? smoothed_aggregation(K) :
                                 smoothed_aggregation(K; B=nullspace)
    spbytes(M) = nnz(M)*(8 + sizeof(eltype(M.colptr))) + (size(M,2)+1)*sizeof(eltype(M.colptr))
    bytes = 0
    for lv in ml.levels
        bytes += spbytes(lv.A) + spbytes(lv.P) + spbytes(sparse(lv.R))
    end
    bytes += spbytes(ml.final_A)
    oc = AlgebraicMultigrid.operator_complexity(ml)
    return bytes, oc, length(ml.levels) + 1
end

# CG iters using SA WITHOUT a near-null-space, on the same imposed elastic operator.
function cg_iters_no_nullspace(n)
    m = cube(n; σy0=1e9, εtarget=1e-3)
    K, R = imposed_operator(m)
    b = -R
    Pl = aspreconditioner(smoothed_aggregation(K))   # NO B nullspace
    _, stats = Krylov.cg(K, b; M=Pl, ldiv=true, atol=0.0, rtol=1e-9, itmax=400)
    return stats.niter
end

rss_gb() = Sys.maxrss() / 2^30

println("="^78)
println("PlasticityFEM scaling validation   (Route A: assembled CG+AMG)")
@printf("threads=%d   free=%.1f GB   NMAX=%d   target N=%.2e\n",
        Threads.nthreads(), Sys.free_memory()/2^30, NMAX, TARGET_N)
println("="^78)

results = Tuple{String,Bool}[]
addresult(name, ok) = push!(results, (name, ok))

# =============================================================================
# A. CORRECTNESS
# =============================================================================
println("\n## A. CORRECTNESS\n")

let rtol = 1e-8, nx = 20
    println("A1. CG+AMG vs direct  (nx=$nx, $(3*(nx+1)^3) DOFs)")
    @printf("    %-10s %-14s %-14s\n", "case", "‖ΔU‖/‖U‖", "‖Δσ‖/‖σ‖")
    okA1 = true
    for (label, kw) in (("elastic", (σy0=1e9, εtarget=1e-3)),
                        ("plastic", (σy0=250.0, Hiso=1000.0, εtarget=0.01)))
        md = cube(nx; kw...); solve!(md; nsteps=8, tol=1e-9, linsolve=:direct)
        Ud = copy(md.U); σd = copy(gauss_stress(md))
        mc = cube(nx; kw...); solve!(mc; nsteps=8, tol=1e-9, linsolve=:cg, amg=:sa)
        ue = norm(mc.U - Ud)/max(norm(Ud), eps())
        se = norm(gauss_stress(mc) - σd)/max(norm(σd), eps())
        @printf("    %-10s %-14.3e %-14.3e\n", label, ue, se)
        okA1 &= (ue <= 10rtol && se <= 10rtol)
    end
    addresult("A1 CG+AMG == direct (≤1e-7)", okA1)
end

let nx = 12
    println("\nA2. threaded(4) == serial(1) result equality  (nx=$nx)")
    m1 = cube(nx; σy0=250.0, Hiso=1000.0); solve!(m1; nsteps=6, tol=1e-9, linsolve=:cg, threaded=false)
    m4 = cube(nx; σy0=250.0, Hiso=1000.0); solve!(m4; nsteps=6, tol=1e-9, linsolve=:cg, threaded=true)
    uerr = norm(m4.U - m1.U)/max(norm(m1.U), eps())
    m = cube(nx; σy0=1e9); m.U .= [1e-4*sin(0.7i) for i in 1:length(m.U)]
    st = m.state_trial
    Ks,Rs = assemble!(m.sparsity, m.material, m.cache, m.U, st.εp,st.β,st.ᾱ,st.σ, m.Rbuf)
    Knz = copy(Ks.nzval); Rc = copy(Rs)
    Kt,Rt = assemble_threaded!(m.sparsity, m.material, m.cache, m.U, st.εp,st.β,st.ᾱ,st.σ, m.Rbuf)
    dK = norm(Kt.nzval - Knz)/(norm(Knz)+1); dR = norm(Rt - Rc)/(norm(Rc)+1)
    @printf("    full-solve ‖ΔU‖/‖U‖ = %.3e   assembled ΔK=%.3e  ΔR=%.3e  (threads=%d)\n",
            uerr, dK, dR, Threads.nthreads())
    addresult("A2 threaded == serial", uerr <= 1e-7 && dK <= 1e-12 && dR <= 1e-12)
end

let nx = 4, E=210e3, σy0=250.0, Hiso=1000.0, εt=0.01
    println("\nA3. uniaxial post-yield closed form through CG path  (nx=$nx)")
    m = cube(nx; E=E, σy0=σy0, Hiso=Hiso, εtarget=εt)
    solve!(m; nsteps=20, tol=1e-10, linsolve=:cg)
    Ht = E*Hiso/(E+Hiso); εy = σy0/E
    σ_exact = σy0 + Ht*(εt - εy)
    σg = gauss_stress(m); σxx = sum(@view σg[1,:]) / size(σg,2)
    relerr = abs(σxx - σ_exact)/σ_exact
    @printf("    σxx(mean)=%.4f  closed-form=%.4f  rel.err=%.2e\n", σxx, σ_exact, relerr)
    addresult("A3 uniaxial post-yield (T2)", relerr <= 1e-4)
end

let nx = 6
    println("\nA4. equilibrium residual on free DOFs through CG path (nx=$nx)")
    m = cube(nx; σy0=250.0, Hiso=1000.0, εtarget=0.01)
    solve!(m; nsteps=10, tol=1e-10, linsolve=:cg)
    st = m.state_committed
    _, R = assemble!(m.sparsity, m.material, m.cache, m.U, st.εp,st.β,st.ᾱ,st.σ, m.Rbuf)
    constrained = falses(length(m.U)); for d in m.dir_dofs; constrained[d] = true; end
    free_resid = norm(R[.!constrained]); scale = norm(R) + 1
    @printf("    ‖F_int(free)‖ = %.3e   (‖F_int‖=%.3e)\n", free_resid, norm(R))
    addresult("A4 equilibrium on free DOFs", free_resid <= 1e-6*scale)
end

# =============================================================================
# B+C. SCALING SWEEP
# =============================================================================
println("\n## B+C. SCALING SWEEP  (elastic, 1 load step, serial timing)\n")

ns = filter(n -> n <= NMAX, [8, 16, 24, 32, 40, 48, 56, 64])
rec = NamedTuple[]
@printf("%-4s %-9s %-11s %-7s %-7s %-9s %-10s %-10s %-11s %-8s %-9s\n",
        "n", "N", "nnz", "nnz/N", "maxCG", "asm[ms]", "setup[ms]", "cgiter[ms]",
        "cg/iter[µs]", "AMG_OC", "bytes/DOF")
for n in ns
    if Sys.free_memory()/2^30 < 2.0
        println("  <2GB free — stopping sweep at n=$n"); break
    end
    m = cube(n; σy0=1e9, εtarget=1e-3)
    N = 3*m.mesh.nnodes; nnzK = nnz(m.sparsity.K)
    st = m.state_trial
    # assembly time (warm, median of 3)
    assemble!(m.sparsity, m.material, m.cache, m.U, st.εp,st.β,st.ᾱ,st.σ, m.Rbuf)
    t_asm = minimum(@elapsed(assemble!(m.sparsity, m.material, m.cache, m.U,
                                       st.εp,st.β,st.ᾱ,st.σ, m.Rbuf)) for _ in 1:3)
    # Decompose the linear solve: AMG SETUP (amortizable across Newton iters by
    # the reuse policy, SCALING §1.3) vs CG ITERATION work (the per-V-cycle SpMV+
    # smoothing — the true O(N)/iter cost). This separates the two so the O(N)
    # claim is tested on the iteration work, not on the (superlinear, amortized)
    # setup phase.
    nullspace = SOLVER._rigid_body_modes(m.mesh.nodes)
    K2, R2 = imposed_operator(m); b = -R2
    GC.gc()
    smoothed_aggregation(K2; B=nullspace)                    # warm
    t_setup = minimum(@elapsed(smoothed_aggregation(K2; B=nullspace)) for _ in 1:2)
    Pl = aspreconditioner(smoothed_aggregation(K2; B=nullspace))
    Krylov.cg(K2, b; M=Pl, ldiv=true, atol=0.0, rtol=1e-9, itmax=300)  # warm
    t_cg = @elapsed (_, st2) = Krylov.cg(K2, b; M=Pl, ldiv=true, atol=0.0, rtol=1e-9, itmax=300)
    maxcg = st2.niter
    t_cgiter = t_cg / max(maxcg, 1)
    amg_bytes, amg_oc, amg_lv = amg_hierarchy_bytes(K2, nullspace)
    rss = rss_gb()
    # bytes/DOF of the persistent objects we can measure exactly:
    Kbytes = nnzK*16 + (N+1)*8
    statebytes = 2 * (8*m.mesh.nelem*8) * 19   # 2 copies × ngp × 19 Float64
    cgbytes = 5*N*8
    edofbytes = 24*m.mesh.nelem*8
    persist = Kbytes + amg_bytes + statebytes + cgbytes + edofbytes
    bpd = persist / N
    push!(rec, (n=n, N=N, nnz=nnzK, maxcg=maxcg, t_asm=t_asm, t_setup=t_setup,
                t_cg=t_cg, t_cgiter=t_cgiter, amg_bytes=amg_bytes, amg_oc=amg_oc,
                amg_lv=amg_lv, rss=rss, Kbytes=Kbytes, statebytes=statebytes,
                persist=persist, bpd=bpd, ngp=8*m.mesh.nelem))
    @printf("%-4d %-9d %-11d %-7.2f %-7d %-9.2f %-10.1f %-10.1f %-11.3f %-8.3f %-9.0f\n",
            n, N, nnzK, nnzK/N, maxcg, t_asm*1e3, t_setup*1e3, t_cg*1e3,
            t_cgiter*1e6, amg_oc, bpd)
    m = nothing; K2 = nothing; GC.gc()
end

# ---------- B-control: WITHOUT near-null-space (must grow ~N^1/3) -------------
println("\nB-control. max CG-iters: SA+nullspace (O(N)) vs SA-no-nullspace vs Jacobi:")
@printf("%-4s %-9s %-12s %-12s %-9s\n", "n", "N", "SA+null", "SA-nonull", "Jacobi")
ctrl = NamedTuple[]
for n in filter(<=(NMAX), [8, 16, 24, 32])
    m = cube(n; σy0=1e9, εtarget=1e-3); N = 3*m.mesh.nnodes
    cg_with = maximum(solve!(m; nsteps=1, tol=1e-9, linsolve=:cg, amg=:sa, threaded=false).cg_iters[1])
    cg_nonull = cg_iters_no_nullspace(n)
    m2 = cube(n; σy0=1e9, εtarget=1e-3)
    cg_jac = maximum(solve!(m2; nsteps=1, tol=1e-9, linsolve=:cg, amg=:sa,
                            smoother=:jacobi, threaded=false).cg_iters[1])
    push!(ctrl, (n=n, N=N, with=cg_with, nonull=cg_nonull, jac=cg_jac))
    @printf("%-4d %-9d %-12d %-12d %-9d\n", n, N, cg_with, cg_nonull, cg_jac)
end

# =============================================================================
# Derived scaling exponents
# =============================================================================
println("\n## SCALING EXPONENTS (log-log slope vs N)\n")
Ns = [r.N for r in rec]
# tail = the larger meshes (drop the smallest 1-2 cache-resident points where
# wall-time/iter is cache-inflated, not bandwidth-bound — the asymptotic regime
# is what extrapolates to 10M).
tail = length(rec) >= 4 ? rec[max(1,end-3):end] : rec
Ntail = [r.N for r in tail]

slope_cg       = loglog_slope(Ns, [r.maxcg for r in rec])
slope_cgiter   = loglog_slope(Ns, [r.t_cgiter for r in rec])
slope_cgiter_t = loglog_slope(Ntail, [r.t_cgiter for r in tail])   # asymptotic
slope_setup    = loglog_slope(Ns, [r.t_setup for r in rec])
slope_tasm     = loglog_slope(Ns, [r.t_asm for r in rec])
slope_bpd      = loglog_slope(Ntail, [r.bpd for r in tail])
nnzN_min, nnzN_max = extrema([r.nnz/r.N for r in rec])
bpd_min, bpd_max = extrema([r.bpd for r in rec])
bpd_tail_spread = (maximum(r.bpd for r in tail) - minimum(r.bpd for r in tail)) /
                   minimum(r.bpd for r in tail)

slope_cg_nonull = length(ctrl) >= 2 ? loglog_slope([c.N for c in ctrl], [c.nonull for c in ctrl]) : NaN
slope_cg_jac    = length(ctrl) >= 2 ? loglog_slope([c.N for c in ctrl], [c.jac for c in ctrl]) : NaN

println("  -- O(flops) (the headline claim) --")
@printf("  CG-iters vs N           : slope = %+.3f   (target ≲ 0.15, flat ⇒ O(N))  *DECISIVE*\n", slope_cg)
@printf("    control SA-no-null    : slope = %+.3f   (grows ⇒ nullspace is what buys O(N))\n", slope_cg_nonull)
@printf("    control Jacobi        : slope = %+.3f\n", slope_cg_jac)
@printf("  CG time/iter vs N       : slope = %+.3f (full) / %+.3f (tail n≥%d) ⇒ O(N) work/iter\n",
        slope_cgiter, slope_cgiter_t, tail[1].n)
@printf("  assembly time vs N      : slope = %+.3f   (target ≈ 1.0, O(N))\n", slope_tasm)
@printf("  AMG SETUP time vs N      : slope = %+.3f   (mildly superlinear; AMORTIZED by\n", slope_setup)
@printf("                                          reuse policy across Newton iters, §1.3)\n")
println("  -- O(memory) --")
@printf("  bytes/DOF vs N (tail)   : slope = %+.3f   (target ≈ 0, flat ⇒ linear mem)\n", slope_bpd)
@printf("  nnz/N range             : %.2f … %.2f → 81 (boundary fraction shrinks; saturating)\n", nnzN_min, nnzN_max)
@printf("  bytes/DOF range         : %.0f … %.0f B (full %.0f%%, tail %.0f%%); rises toward asymptote\n",
        bpd_min, bpd_max, 100*(bpd_max-bpd_min)/bpd_min, 100*bpd_tail_spread)

# flops/DOF/Newton-iter estimate: assembly (~const) + cg_iters·(SpMV+Vcycle)
let r = rec[end]
    # SpMV ≈ 2·nnz/N flops/DOF; V-cycle ≈ OC·(smoother sweeps≈2)·2·nnz/N flops/DOF
    nnzN = r.nnz/r.N
    spmv = 2*nnzN
    vcyc = r.amg_oc * 4 * nnzN     # ~2 GS sweeps (pre+post) × 2 flop/nz
    asm  = 5000.0                  # ~few×10³ flops/DOF for return map + 24×24 products
    flops_dof = asm + r.maxcg*(spmv + vcyc)
    @printf("  flops/DOF/Newton-iter ≈ %.2e  (asm≈%.0f + %d CG×(SpMV %.0f + Vcyc %.0f)); target ~1e4\n",
            flops_dof, asm, r.maxcg, spmv, vcyc)
    global FLOPS_DOF = flops_dof
end

# =============================================================================
# C. Build-time transient + assemble! allocation gate + nnz/N
# =============================================================================
println("\n## C. BUILD-TIME TRANSIENT & ALLOCATION GATES\n")

# build-time alloc of Model construction across sizes (must be O(N)-small, no blowup)
println("Model construction allocation (build_sparsity + precompute_cache):")
@printf("  %-5s %-10s %-12s %-12s %-12s\n", "n", "N", "@alloc[MB]", "alloc/DOF", "K bytes[MB]")
build_ratios = Float64[]
for n in [8, 16, 24, 32]
    n > NMAX && continue
    mesh = box_mesh(1.0,1.0,1.0,n,n,n); mat = J2Material(E=210e3, ν=0.3, σy0=250.0)
    Model(mesh, mat)   # warm
    a = @allocated Model(mesh, mat)
    N = 3*mesh.nnodes
    m = Model(mesh, mat); Kb = nnz(m.sparsity.K)*16 + (N+1)*8
    @printf("  %-5d %-10d %-12.1f %-12.1f %-12.1f\n", n, N, a/2^20, a/N, Kb/2^20)
    push!(build_ratios, a/N)
end
# build alloc/DOF should be roughly flat (O(N)); flag the 295 GB B-cache / 47 GB COO regressions
build_flat = (length(build_ratios) >= 2) && (maximum(build_ratios)/minimum(build_ratios) < 4)
@printf("  alloc/DOF range %.0f … %.0f B  ⇒ %s (no super-linear build blowup)\n",
        minimum(build_ratios), maximum(build_ratios), build_flat ? "FLAT" : "GROWS")
addresult("C build alloc O(N)-small (no 31/47/295 GB transient)", build_flat)

# assemble! allocation gate at ≥3 sizes — bounded constant, independent of nelem
println("\nassemble! allocation (on-the-fly scatter; must be O(1) in nelem):")
function asm_alloc(n)
    m = cube(n; σy0=1e9); st = m.state_trial
    assemble!(m.sparsity, m.material, m.cache, m.U, st.εp,st.β,st.ᾱ,st.σ, m.Rbuf)
    return @allocated assemble!(m.sparsity, m.material, m.cache, m.U,
                                st.εp,st.β,st.ᾱ,st.σ, m.Rbuf)
end
asm_allocs = [(n, asm_alloc(n)) for n in [4, 8, 16, 24] if n <= NMAX]
for (n,a) in asm_allocs
    @printf("  n=%-3d nelem=%-7d  @allocated assemble! = %d B\n", n, n^3, a)
end
asm_bounded = maximum(a for (_,a) in asm_allocs) <= minimum(a for (_,a) in asm_allocs) + 2048
addresult("C assemble! O(1) alloc (bounded, indep of nelem)", asm_bounded)

# element kernel + return_map allocation == 0
let m = cube(4; σy0=250.0, Hiso=1000.0)
    st = m.state_trial
    ue = ones(SVector{24,Float64}) .* 1e-4
    a_elem = @allocated element_force_tangent!(m.material, m.cache.Bref, m.cache.detJwref,
                                               ue, st.εp, st.β, st.ᾱ, 1, st.σ, Val(false))
    rm_alloc = @allocated return_map(m.material, (@SVector zeros(6)), (@SVector zeros(6)),
                                     (@SVector zeros(6)), 0.0)
    @printf("\n  @allocated element_force_tangent! = %d B ;  return_map = %d B (target 0)\n",
            a_elem, rm_alloc)
    addresult("C element kernel + return_map 0 alloc", a_elem == 0 && rm_alloc == 0)
end

# =============================================================================
# D. EXTRAPOLATED 10M-ON-64GB BUDGET
# =============================================================================
println("\n" * "="^78)
println("## D. EXTRAPOLATED 10M-on-64GB BUDGET")
println("="^78)

# Measured per-DOF / per-GP constants (from the largest mesh in the sweep)
rL = rec[end]
nnz_per_N   = rL.nnz / rL.N                       # → 81
amg_oc      = maximum(r.amg_oc for r in rec)      # worst-case operator complexity
amg_b_per_N = rL.amg_bytes / rL.N                 # measured AMG bytes/DOF
t_cgiter_per_N = rL.t_cgiter / rL.N               # s/DOF/CG-iteration (serial)
t_setup_per_N  = rL.t_setup / rL.N                # s/DOF AMG setup (serial)
t_asm_per_N    = rL.t_asm / rL.N                  # s/DOF assembly (serial)
maxcg       = maximum(r.maxcg for r in rec)

# state bytes/GP and ngp/N are exact (geometry), not extrapolated guesses
ngp_per_N = TARGET_NGP / TARGET_N                 # = 8·nelem / (3·nnodes) ≈ 2.61
state_b_per_gp = 19 * 8                           # εp,β,σ (6 each) + ᾱ

# Line items at N = TARGET_N
N = TARGET_N
nnzT = nnz_per_N * N
GB(x) = x / 2^30
lines = Tuple{String,Float64,String}[]
push!(lines, ("K nzval (Float64)",      nnzT*8,                  @sprintf("nnz·8, nnz=%.2e", nnzT)))
push!(lines, ("K rowval (Int64)",       nnzT*8,                  "nnz·8 (Int64 default, §9.2)"))
push!(lines, ("K colptr (Int64)",       (N+1)*8,                 "(N+1)·8"))
push!(lines, ("GaussState committed",   TARGET_NGP*state_b_per_gp, "ngp·152"))
push!(lines, ("GaussState trial",       TARGET_NGP*state_b_per_gp, "ngp·152"))
push!(lines, ("AMG hierarchy",          amg_b_per_N*N,           @sprintf("measured %.0f B/DOF, OC=%.2f", amg_b_per_N, amg_oc)))
push!(lines, ("CG work vectors (5N)",   5*N*8,                   "Krylov CG workspace"))
push!(lines, ("edofs (Int64)",          24*(N/3)*8,              "24·nelem·8"))
push!(lines, ("nullspace B (6 modes)",  6*N*8,                   "6·N·8 (rigid-body modes)"))
push!(lines, ("Mesh + colors + misc",   (3*(N/3) + 8*(N/3) + (N/3))*8 + 4*N*8, "nodes/elements/coloring + Fext,U,δU,Rbuf"))

println()
@printf("  %-28s %-12s %s\n", "Line item", "bytes (GB)", "formula / source")
println("  " * "-"^72)
total = 0.0
for (name, b, note) in lines
    @printf("  %-28s %-12.2f %s\n", name, GB(b), note)
    global total += b
end
println("  " * "-"^72)
@printf("  %-28s %-12.2f\n", "TOTAL (steady-state)", GB(total))
@printf("  %-28s %-12.2f\n", "BUDGET", BUDGET_GB)
@printf("  %-28s %-12.2f GB  free\n", "HEADROOM", BUDGET_GB - GB(total))

# transient peak (pattern build): K + O(N) adjacency scratch (~adj vectors)
adj_scratch = 27 * (N/3) * 8                       # ≤27 neighbor ints/node, Int64
pattern_peak = (nnzT*16 + (N+1)*8) + adj_scratch
@printf("\n  Pattern-build transient peak ≈ %.1f GB  (K + O(N) adjacency; vs 47 GB COO path)\n",
        GB(pattern_peak))

under_budget = GB(total) < BUDGET_GB
addresult("D extrapolated 10M total < 64 GB", under_budget)

# projected wall-time per load step from the DECOMPOSED measured per-DOF constants.
# Per load step (reuse policy §1.3): 1 AMG setup + n_newton × (assembly + maxcg ×
# cg/iter). Use n_newton=5 (typical plastic step) and the worst-case maxcg.
n_newton = 5
proj_setup    = t_setup_per_N  * N
proj_asm      = t_asm_per_N    * N
proj_cgiter   = t_cgiter_per_N * N
proj_step = proj_setup + n_newton * (proj_asm + maxcg * proj_cgiter)
@printf("\n  Projected wall-time @10M per load step (serial, from decomposed constants):\n")
@printf("    AMG setup       %.2e s/DOF ⇒ %6.0f s  (once/step, amortized)\n", t_setup_per_N, proj_setup)
@printf("    assembly        %.2e s/DOF ⇒ %6.0f s  × %d Newton = %.0f s\n",
        t_asm_per_N, proj_asm, n_newton, n_newton*proj_asm)
@printf("    CG iter         %.2e s/DOF ⇒ %6.1f s/iter × %d iters × %d Newton = %.0f s\n",
        t_cgiter_per_N, proj_cgiter, maxcg, n_newton, n_newton*maxcg*proj_cgiter)
@printf("    -> ~%.0f s/load step SERIAL  (~%.1f min). Threaded(SpMV+assembly) cuts the\n",
        proj_step, proj_step/60)
@printf("       Newton-loop portion ~2-6×; AMG setup is the long pole (parallelize coarsening).\n")
@printf("    flops/DOF/Newton-iter ≈ %.2e (mesh-independent) ⇒ ~%.2e flops/step @10M\n",
        FLOPS_DOF, FLOPS_DOF*N)
println("\n  MEASURED vs ASSUMED:")
println("    MEASURED (this sweep): nnz/DOF, AMG bytes/DOF & operator-complexity, CG-iters,")
println("                           CG time/iter, AMG setup time, assembly time, bytes/DOF.")
println("    EXACT (geometry):      GaussState (152 B/GP), ngp, edofs, nullspace, mesh sizes.")
println("    ASSUMED:               n_newton=5/step; linear extrapolation of per-DOF constants;")
println("                           AMG OC at 10M ≈ measured (it is flat/mildly varying in sweep).")

# =============================================================================
# SUMMARY
# =============================================================================
println("\n" * "="^78)
println("## VALIDATION SUMMARY")
println("="^78)
# scaling-exponent gate results (gate the algorithmically-meaningful quantities)
addresult("B CG-iters slope ≲ 0.15 (flat ⇒ O(N)) *DECISIVE*", slope_cg <= 0.15)
addresult("B CG time/iter slope (tail) ≲ 1.15 (O(N) work/iter)", slope_cgiter_t <= 1.15)
addresult("B assembly time slope ≲ 1.2 (O(N))", slope_tasm <= 1.2)
addresult("B control: SA-no-null grows (nullspace buys O(N))", isnan(slope_cg_nonull) ? true : slope_cg_nonull > slope_cg + 0.10)
addresult("C nnz/N ≤ 81 (mesh-indep stencil)", nnzN_max <= 81 + 1e-6)
addresult("C bytes/DOF saturating (tail spread ≤12%)", bpd_tail_spread <= 0.12)

allok = true
for (name, ok) in results
    @printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name)
    global allok &= ok
end
println("="^78)
println(allok ? "  OVERALL: PASS — correct AND optimal (memory / O(flops) / speed)" :
                "  OVERALL: some checks FAILED — see above")
println("="^78)
