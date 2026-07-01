"""
    Solver

Newton–Raphson with load stepping and history commit. See DESIGN.md §1.5, §5.1.

Linear solve (SCALING.md §1): the default path is preconditioned Conjugate
Gradient (`Krylov.cg!`) with an Algebraic Multigrid preconditioner
(`AlgebraicMultigrid`), driven by an inexact-Newton (Eisenstat–Walker) forcing
term. A `:direct` (UMFPACK `\\`) path is kept for tiny/degenerate systems and as a
correctness reference. Preconditioner reuse policy: rebuild the AMG hierarchy
once per load step and reuse it across that step's Newton iterations (always
correctness-safe — CG always multiplies by the true current K).
"""
module Solver

using SparseArrays
using LinearAlgebra
using Krylov: CgWorkspace, cg!
using AlgebraicMultigrid: smoothed_aggregation, ruge_stuben, aspreconditioner, Jacobi
using ..MeshMod: GaussState
using ..FiniteStrain: Hex8Small, Hex8Finite, Hex8FiniteFbar
using ..BoundaryConditions: DirichletBC, NeumannBC, impose_dirichlet!, assemble_neumann!
using ..Assembly: assemble!, assemble_threaded!
using ..ModelMod: Model, reset!

export solve!, newton!, SolveResult, LinearSolveState

# Row-parallel SpMV for a SYMMETRIC CSC matrix (SCALING.md §3.2). Because K = Kᵀ,
# CSC column i holds exactly the entries of row i, so y[i] = Σ_{k in col i}
# nzval[k]·x[rowval[k]] lets each thread write a disjoint y[i] — race-free, no
# atomics. Used as the CG operator when threading is on; CG still applies the AMG
# preconditioner built from the underlying K. Falls back to a serial loop when
# single-threaded.
struct SymThreadedK{Tv,Ti}
    K::SparseMatrixCSC{Tv,Ti}
end
Base.eltype(::SymThreadedK{Tv}) where {Tv} = Tv
Base.size(A::SymThreadedK, d::Integer) = size(A.K, d)
Base.size(A::SymThreadedK) = size(A.K)
function LinearAlgebra.mul!(y::AbstractVector, A::SymThreadedK, x::AbstractVector)
    K = A.K; cp = K.colptr; rv = K.rowval; nz = K.nzval
    Threads.@threads for i in 1:size(K, 2)
        s = zero(eltype(y))
        @inbounds for k in cp[i]:(cp[i+1] - 1)
            s += nz[k] * x[rv[k]]
        end
        @inbounds y[i] = s
    end
    return y
end
Base.:*(A::SymThreadedK, x::AbstractVector) = mul!(similar(x, promote_type(eltype(A), eltype(x))), A, x)

"""
    SolveResult

Per-load-step iteration counts and residual histories (DESIGN §6.3). Used by
tests to assert Newton convergence rate (T15). `cg_iters` records, per load step,
the inner CG iteration count of every Newton iteration (empty for the `:direct`
path) — used by the scaling sweep to assert mesh-independent CG counts.
"""
struct SolveResult
    converged::Bool
    iters::Vector{Int}
    residuals::Vector{Vector{Float64}}
    cg_iters::Vector{Vector{Int}}
end

# Backwards-compatible constructor (older call sites / tests that omit cg_iters).
SolveResult(c, i, r) = SolveResult(c, i, r, Vector{Int}[])

"""
    LinearSolveState

Persistent workspace for the iterative linear solve (SCALING.md §1.2): the Krylov
CG workspace (so the inner solve is ~O(1) alloc, not O(N), per Newton iteration)
and the cached AMG preconditioner for the current load step. Created once for a
`solve!` call and reused across all load steps / Newton iterations.

Fields:
- `method` — `:cg` (CG+AMG, default) or `:direct` (UMFPACK `\\`).
- `amg` — `:sa` (smoothed aggregation, default) or `:rs` (Ruge–Stüben).
- `smoother` — `:gs` (Gauss–Seidel, AMG default) or `:jacobi` (thread-parallel).
- `ws` — Krylov `CgWorkspace` sized to ndof (allocated lazily).
- `Pl` — the cached AMG preconditioner (one V-cycle), rebuilt per load step.
"""
mutable struct LinearSolveState
    method::Symbol
    amg::Symbol
    smoother::Symbol
    cg_itmax::Int
    η_min::Float64
    η_max::Float64
    ew_γ::Float64
    ew_α::Float64
    ndof::Int
    threaded::Bool   # use row-parallel SpMV + 8-color threaded assembly
    nullspace::Union{Matrix{Float64},Nothing}  # rigid-body modes for SA-AMG (R1)
    ws::Any          # CgWorkspace{Float64,Float64,Vector{Float64}} | Nothing
    Pl::Any          # AlgebraicMultigrid.Preconditioner | Nothing
    cg_iters_hist::Vector{Int}   # rolling history for the refresh trigger
end

function LinearSolveState(ndof::Int; method::Symbol=:cg, amg::Symbol=:sa,
                          smoother::Symbol=:gs, cg_itmax::Int=200,
                          η_min::Float64=1e-8, η_max::Float64=0.1,
                          ew_γ::Float64=0.9, ew_α::Float64=1.5,
                          threaded::Bool=(Threads.nthreads() > 1),
                          nullspace::Union{Matrix{Float64},Nothing}=nothing)
    return LinearSolveState(method, amg, smoother, cg_itmax, η_min, η_max,
                            ew_γ, ew_α, ndof, threaded, nullspace, nothing, nothing, Int[])
end

# Rigid-body near-null-space (6 modes: 3 translations + 3 rotations) of the
# 3-DOF-per-node elasticity operator, as columns of an ndof×6 matrix. Supplying
# these to smoothed-aggregation AMG flattens the CG iteration count from ~N^0.3 to
# ~constant (SCALING.md §1.3 R1) — the difference between O(N^1.3) and O(N) flops.
function _rigid_body_modes(nodes::Matrix{Float64})
    nn = size(nodes, 2)
    B = zeros(3 * nn, 6)
    @inbounds for n in 1:nn
        x = nodes[1, n]; y = nodes[2, n]; z = nodes[3, n]
        ix = 3 * (n - 1) + 1; iy = ix + 1; iz = ix + 2
        B[ix, 1] = 1.0; B[iy, 2] = 1.0; B[iz, 3] = 1.0   # translations
        B[iy, 4] = -z;  B[iz, 4] = y                      # rotation about x
        B[ix, 5] = z;   B[iz, 5] = -x                     # rotation about y
        B[ix, 6] = -y;  B[iy, 6] = x                      # rotation about z
    end
    return B
end

# Build an AMG hierarchy from K and wrap it as a (one V-cycle) preconditioner.
# Smoothed aggregation (default) uses the rigid-body near-null-space `ls.nullspace`
# when present, which flattens the CG iteration count for 3D vector elasticity
# (SCALING.md §1.3 R1). Ruge–Stüben is the documented fallback. A Jacobi smoother
# (fully thread-parallel) is selectable; the AMG default is symmetric Gauss–Seidel.
function _build_amg(K::SparseMatrixCSC, ls::LinearSolveState)
    amg = ls.amg; B = ls.nullspace
    sm = ls.smoother === :jacobi ? Jacobi(2.0 / 3.0; iter=1) : nothing
    if amg === :rs
        # AlgebraicMultigrid's Ruge–Stüben needs Int-indexed input (its Int32
        # path is broken); copy only if K is not already Int-indexed.
        Krs = eltype(K.colptr) === Int ? K : SparseMatrixCSC{eltype(K),Int}(K)
        ml = sm === nothing ? ruge_stuben(Krs) : ruge_stuben(Krs; presmoother=sm, postsmoother=sm)
    elseif B !== nothing
        # SA with a near-null-space builds Int64 prolongators, so K must be
        # Int-indexed (build_sparsity defaults to Int for exactly this reason).
        ml = sm === nothing ? smoothed_aggregation(K; B=B) :
                              smoothed_aggregation(K; B=B, presmoother=sm, postsmoother=sm)
    else
        ml = sm === nothing ? smoothed_aggregation(K) :
                              smoothed_aggregation(K; presmoother=sm, postsmoother=sm)
    end
    return aspreconditioner(ml)
end

# Eisenstat–Walker choice 2 forcing term (SCALING.md §1.4). `rk`,`rkm1` are the
# current/previous Newton residual norms; `η_prev` the previous forcing term.
# Returns the CG rtol for this Newton iteration.
function _forcing_term(ls::LinearSolveState, it::Int, rk::Float64,
                       rkm1::Float64, η_prev::Float64)
    if it == 1 || rkm1 <= 0
        return ls.η_max
    end
    η = ls.ew_γ * (rk / rkm1)^ls.ew_α
    # safeguard against oversolving oscillation
    safe = ls.ew_γ * η_prev^ls.ew_α
    if safe > 0.1
        η = max(η, safe)
    end
    return clamp(η, ls.η_min, ls.η_max)
end

# --- tangent symmetry (SCALING §3.2 / FINITE_STRAIN §4.2) ---
#
# CG + AMG (with the rigid-body near-null-space) + the row-parallel `SymThreadedK`
# SpMV ALL assume K = Kᵀ. The finite-strain consistent tangent is symmetric only
# for small strain, or finite strain with NO kinematic hardening (isotropic /
# perfect plasticity). It is NON-symmetric for:
#   • F-bar (Hex8FiniteFbar) — always (the centroid J₀ coupling is non-symmetric);
#   • finite strain with kinematic hardening (Hkin > 0) — the objective back-stress
#     rotation contributes a non-symmetric ∂τ/∂β·∂β_sp/∂F term (FINITE_STRAIN §4.6).
# For those, CG is invalid (cg! raises an SPD error); we must use a direct solve.
@inline _tangent_symmetric(model::Model) =
    _kind_symmetric(model.kind, model.material.Hkin)
@inline _kind_symmetric(::Hex8Small, ::Float64)      = true
@inline _kind_symmetric(::Hex8Finite, Hkin::Float64) = Hkin == 0.0
@inline _kind_symmetric(::Hex8FiniteFbar, ::Float64) = false

# Build the frozen BC objects from the model accumulators.
# Dirichlet is split into ramped (prescribed) and non-ramped (fixed) sets so a
# single `ramp` bool per DirichletBC suffices (DESIGN §4.5).
function _freeze_bcs(model::Model)
    # Deduplicate Dirichlet constraints per DOF (last assignment wins) so a DOF
    # that was constrained more than once does not appear twice in the imposed
    # system. Warn if two *conflicting* values are prescribed on the same DOF.
    last = Dict{Int,Tuple{Float64,Bool}}()   # dof => (value, ramp)
    @inbounds for i in eachindex(model.dir_dofs)
        d = model.dir_dofs[i]
        v = model.dir_vals[i]
        prev = get(last, d, nothing)
        if prev !== nothing && prev[1] != v
            @warn "conflicting Dirichlet values on dof $d ($(prev[1]) vs $v); using the last" maxlog=5
        end
        last[d] = (v, model.dir_ramp[i])
    end
    rd = Int[]; rv = Float64[]; fd = Int[]; fv = Float64[]
    for (d, (v, ramp)) in last
        if ramp
            push!(rd, d); push!(rv, v)
        else
            push!(fd, d); push!(fv, v)
        end
    end
    dir_ramp = DirichletBC(rd, rv, true)
    dir_fix = DirichletBC(fd, fv, false)
    neu = NeumannBC(copy(model.neu_dofs), copy(model.neu_vals))
    return dir_ramp, dir_fix, neu
end

# Classify a CG error: true iff it is the non-SPD / "positive definite" failure
# `cg!` raises on a non-symmetric / indefinite tangent (the only case for which a
# direct fallback is the right recovery). InterruptException (Ctrl-C) and genuine
# CG misconfiguration errors are NOT this and must propagate (S1). We match on the
# rendered message because Krylov surfaces this as a plain error type.
function _is_non_spd_error(err::Exception)
    err isa InterruptException && return false
    msg = sprint(showerror, err)
    return occursin("positive definite", msg) || occursin("not symmetric", msg) ||
           occursin("indefinite", msg)
end

# Solve K δU = b into δU. CG+AMG by default, with a refresh-on-stall fallback;
# `:direct` uses UMFPACK. Returns the number of CG iterations (0 for direct).
# `rebuild_pc` forces an AMG rebuild before solving (start of a load step).
function _linear_solve!(δU::Vector{Float64}, K::SparseMatrixCSC, b::Vector{Float64},
                        ls::LinearSolveState, rtol::Float64, rebuild_pc::Bool)
    if ls.method === :direct
        δU .= K \ b
        return 0
    end

    # (re)build AMG preconditioner per the reuse policy
    if rebuild_pc || ls.Pl === nothing
        ls.Pl = _build_amg(K, ls)
    end
    if ls.ws === nothing || ls.ws.n != length(b)
        ls.ws = CgWorkspace(length(b), length(b), Vector{Float64})
    end

    # CG operator: the row-parallel symmetric SpMV wrapper when threading is on
    # (K is symmetric after `impose_dirichlet!`), else the plain CSC matrix. The
    # AMG preconditioner is always built from the underlying sparse K.
    A = ls.threaded ? SymThreadedK(K) : K
    # Guard: cg! raises an SPD error (rather than returning !solved) if K or M is
    # not symmetric positive definite. Catch it and fall back to a direct solve so
    # an unexpectedly non-symmetric tangent can never crash the Newton loop.
    try
        cg!(ls.ws, A, b; M=ls.Pl, ldiv=true, atol=0.0, rtol=rtol, itmax=ls.cg_itmax)
    catch err
        _is_non_spd_error(err) || rethrow()
        @warn "CG raised $(typeof(err)) ($(sprint(showerror, err))); falling back to direct solve" maxlog=5
        δU .= K \ b
        return 0
    end
    stats = ls.ws.stats
    niter = stats.niter

    # Fallback 1 (SCALING.md §1.7): refresh the preconditioner from the current K
    # and retry once if CG failed to converge (stall / itmax hit).
    if !stats.solved
        ls.Pl = _build_amg(K, ls)
        try
            cg!(ls.ws, A, b; M=ls.Pl, ldiv=true, atol=0.0, rtol=rtol, itmax=ls.cg_itmax)
        catch err
            _is_non_spd_error(err) || rethrow()
            @warn "CG raised $(typeof(err)) on retry ($(sprint(showerror, err))); falling back to direct solve" maxlog=5
            δU .= K \ b
            return niter
        end
        stats = ls.ws.stats
        niter += stats.niter
        if !stats.solved
            @warn "CG failed to converge after preconditioner refresh ($(stats.status)); falling back to direct solve" maxlog=5
            δU .= K \ b
            return niter
        end
    end
    copyto!(δU, ls.ws.x)
    return niter
end

# Assemble K and F_int, choosing the race-free 8-color threaded path when enabled
# (SCALING.md §3.1); otherwise the serial assembler. Same result either way.
@inline function _assemble_KR!(model::Model, U::Vector{Float64}, st::GaussState,
                               R::Vector{Float64}, threaded::Bool, commit::Bool)
    if threaded
        return assemble_threaded!(model.sparsity, model.material, model.cache, U,
                                  st.εp, st.β, st.ᾱ, st.σ, R; commit=commit,
                                  kind=model.kind, Cp_inv=st.Cp_inv)
    else
        return assemble!(model.sparsity, model.material, model.cache, U,
                         st.εp, st.β, st.ᾱ, st.σ, R; commit=commit,
                         kind=model.kind, Cp_inv=st.Cp_inv)
    end
end

"""
    newton!(model, dir_ramp, dir_fix, neu, λ, Fext, ls; tol, maxiter, verbose)
        -> (converged, residual_history, cg_history)

One load step at load factor λ. Iterates K_T δU = −R to equilibrium using the
consistent tangent (quadratic convergence, DESIGN §1.5). The inner linear solve
is CG+AMG with an inexact-Newton (Eisenstat–Walker) forcing term via `ls`
(SCALING.md §1.4). Writes the trial per-GP state; the caller commits on
convergence. The AMG preconditioner is rebuilt once at the start of the step.
"""
function newton!(model::Model, dir_ramp::DirichletBC, dir_fix::DirichletBC,
                 neu::NeumannBC, λ::Float64, Fext::Vector{Float64},
                 ls::LinearSolveState;
                 tol::Float64=1e-8, maxiter::Int=25, verbose::Bool=false)
    st = model.state_trial
    R = model.Rbuf
    U = model.U

    # external force for this step
    fill!(Fext, 0.0)
    assemble_neumann!(Fext, neu, λ)

    res_hist = Float64[]
    cg_hist = Int[]
    converged = false
    # Reference scale for a *relative* convergence test (DESIGN §6.3). See the v1
    # rationale: set from the first-iteration residual + external load norm with a
    # floor of 1. UNCHANGED by Phase 1 (outer Newton test is preserved).
    ref = 1.0
    fext_norm = sqrt(sum(abs2, Fext))

    rkm1 = 0.0      # previous Newton residual norm (for Eisenstat–Walker)
    η_prev = ls.η_max
    first_step = true

    for it in 1:maxiter
        # reset trial state from committed before recomputing (path-dependent)
        copyto!(model.state_trial, model.state_committed)

        # assemble F_int (into R) and K (into sparsity.nzval)
        K, _ = _assemble_KR!(model, U, st, R, ls.threaded, false)
        # residual R = F_int − F_ext
        @inbounds @. R = R - Fext

        # impose Dirichlet symmetrically (modifies K, R): on free rows this
        # carries the known-column contributions; on constrained rows R holds
        # the constraint violation −δg = −(g − U[d]) (DESIGN §5).
        impose_dirichlet!(K, R, dir_ramp, λ, U)
        impose_dirichlet!(K, R, dir_fix, λ, U)

        # convergence: full residual of the imposed system (UNCHANGED from v1).
        rnorm = sqrt(sum(abs2, R))
        push!(res_hist, rnorm)
        if it == 1
            ref = max(rnorm, fext_norm, 1.0)
        end

        if rnorm <= tol * ref
            converged = true
            verbose && println("    iter $it  |R| = $rnorm  -> converged")
            break
        end

        # inexact-Newton forcing term = inner CG rtol (SCALING.md §1.4)
        η = _forcing_term(ls, it, rnorm, rkm1, η_prev)
        # rebuild the AMG preconditioner once at the start of the load step
        # (reuse policy SCALING.md §1.3); reuse it for the rest of the iterations.
        rebuild = first_step
        first_step = false
        # RHS b = −R (the convergence test above is already done, and R is fully
        # re-assembled next iteration, so negating it in place is safe & alloc-free)
        @inbounds @. R = -R
        niter = _linear_solve!(model.δU, K, R, ls, η, rebuild)
        push!(cg_hist, niter)
        verbose && println("    iter $it  |R| = $rnorm  η = $η  cg_iters = $niter")

        @inbounds @. U = U + model.δU
        rkm1 = rnorm
        η_prev = η
    end

    return converged, res_hist, cg_hist
end

"""
    solve!(model; nsteps=10, tol=1e-8, maxiter=25, verbose=false,
           linsolve=:auto, amg=:sa, smoother=:gs, cg_itmax=200) -> SolveResult

Load-stepped Newton driver (DESIGN §1.5, §6.3). Ramps λ = 1/N … 1, Newton-
iterates each step, and commits the per-GP state on convergence (plasticity is
path dependent).

Linear-solver keywords (SCALING.md §1):
- `linsolve` — `:auto` (default), `:cg` (PCG+AMG) or `:direct` (UMFPACK `\\`).
  CG + AMG + the row-parallel `SymThreadedK` SpMV all assume a SYMMETRIC tangent
  K = Kᵀ. `:auto` inspects the element kind + material and picks the right solver:
    • `:small`, or `:finite` with **no kinematic hardening** (Hkin = 0) ⇒ symmetric
      ⇒ CG + AMG;
    • `:finite_fbar` (always), or `:finite` with `Hkin > 0` ⇒ NON-symmetric tangent
      (F-bar's centroid coupling; the objective back-stress rotation, FINITE_STRAIN
      §4.2/§4.6) ⇒ `:direct` (UMFPACK).
  Forcing `linsolve=:cg` on a non-symmetric configuration is unsafe (`cg!` raises
  an SPD error); `solve!` WARNS and overrides to `:direct` in that case. The inner
  solve is additionally guarded so any thrown SPD error falls back to direct.
- `amg` — `:sa` smoothed aggregation (default) or `:rs` Ruge–Stüben.
- `smoother` — `:gs` Gauss–Seidel (AMG default) or `:jacobi` (thread-parallel).
- `cg_itmax` — inner CG iteration cap (default 200).
- `threaded` — 8-color threaded assembly + row-parallel SpMV (default: on when
  `Threads.nthreads() > 1`). The row-parallel SpMV is only used on the symmetric
  CG path; the direct path assembles the full (possibly non-symmetric) K. Results
  are identical to the serial path.
"""
function solve!(model::Model; nsteps::Int=10, tol::Float64=1e-8,
                maxiter::Int=25, verbose::Bool=false,
                linsolve::Symbol=:auto, amg::Symbol=:sa, smoother::Symbol=:gs,
                cg_itmax::Int=200, threaded::Bool=(Threads.nthreads() > 1))
    reset!(model)   # idempotent: start from the undeformed, unhardened state
    dir_ramp, dir_fix, neu = _freeze_bcs(model)
    Fext = zeros(length(model.U))

    # Symmetry-aware solver selection (B1 / FINITE_STRAIN §4.2). CG/AMG/SymThreadedK
    # require K = Kᵀ; a non-symmetric tangent must use the direct solve.
    sym = _tangent_symmetric(model)
    if linsolve === :auto
        linsolve = sym ? :cg : :direct
    elseif linsolve === :cg && !sym
        @warn "linsolve=:cg requested but the tangent is non-symmetric for this " *
              "element/material (F-bar, or finite strain with kinematic hardening); " *
              "overriding to :direct (CG/AMG assume K=Kᵀ)." maxlog=5
        linsolve = :direct
    end
    # `ls.threaded` drives both the 8-color threaded assembly and the row-parallel
    # `SymThreadedK` CG SpMV. The SpMV assumes K = Kᵀ, but it is ONLY reached on the
    # `:cg` path, and a non-symmetric tangent has already been forced to `:direct`
    # above — so the SpMV never sees a non-symmetric K. Threaded assembly is
    # race-free regardless of symmetry, so keep it on for the direct path too.

    # rigid-body near-null-space for smoothed-aggregation AMG (flattens CG iters)
    nullspace = (linsolve === :cg && amg === :sa) ? _rigid_body_modes(model.mesh.nodes) : nothing
    ls = LinearSolveState(length(model.U); method=linsolve, amg=amg,
                          smoother=smoother, cg_itmax=cg_itmax, threaded=threaded,
                          nullspace=nullspace)

    iters = Int[]
    residuals = Vector{Float64}[]
    cg_iters = Vector{Int}[]
    allconv = true

    for n in 1:nsteps
        λ = n / nsteps
        verbose && println("Load step $n/$nsteps  (λ=$λ)")
        conv, hist, chist = newton!(model, dir_ramp, dir_fix, neu, λ, Fext, ls;
                                    tol=tol, maxiter=maxiter, verbose=verbose)
        push!(iters, length(hist))
        push!(residuals, hist)
        push!(cg_iters, chist)
        if !conv
            allconv = false
            @warn "load step $n did not converge in $maxiter iterations"
            break
        end
        # commit: re-run assembly once with commit=true to write GP state,
        # then copy trial → committed (DESIGN §9 commit semantics).
        _assemble_KR!(model, model.U, model.state_trial, model.Rbuf, ls.threaded, true)
        copyto!(model.state_committed, model.state_trial)
    end

    return SolveResult(allconv, iters, residuals, cg_iters)
end

end # module
