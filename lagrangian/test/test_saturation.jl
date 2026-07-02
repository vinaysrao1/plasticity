# Saturation (Voce) isotropic hardening kernel tests (T9–T15).
#
# The nonlinear isotropic law σy(ᾱ) = σy0 + (σsat−σy0)(1−e^{−δᾱ}) + Hiso·ᾱ turns
# the closed-form return into a scalar local Newton on Δγ (Materials.jl). These
# tests pin: (i) it reduces to the linear kernel bit-for-bit when σsat=σy0;
# (ii) the returned stress lands on the *updated* yield surface (Newton actually
# converged); (iii) a uniaxial curve matches an INDEPENDENT closed-form solve;
# (iv) the consistent tangent (whose plastic modulus is now state-dependent)
# still matches finite differences; (v) it stays allocation-free; (vi) the
# unsupported saturation+kinematic combination is rejected at construction.
#
# Simo (1988) round-bar-necking material: E=206.9 GPa, ν=0.29, σy0=450 MPa,
# σsat(σ∞)=715 MPa, δ=16.93, Hiso(H)=129.24 MPa.

using PlasticityFEM
using PlasticityFEM.Materials
using StaticArrays
using LinearAlgebra
using Test

const Z6 = zero(SVector{6,Float64})

# Simo necking material parameters, reused across the tests below.
const SIMO = (E=206.9e3, ν=0.29, σy0=450.0, σsat=715.0, δ=16.93, H=129.24)

@testset "T9 saturation reduces to linear (bit-identical) when σsat=σy0" begin
    matL = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0)
    # σsat=σy0 ⇒ saturation term ≡ 0; δ arbitrary (multiplies a zero coefficient)
    matS = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, σsat=250.0, δ=25.0)
    for ε in (SVector{6,Float64}(0.01, -0.002, 0.001, 0.003, 0, 0),
              SVector{6,Float64}(0.02, -0.005, -0.005, 0.0, 0.004, 0.0),
              SVector{6,Float64}(1e-4, -2e-5, 3e-5, 1e-5, -1e-5, 2e-5))  # elastic
        rL = return_map(matL, ε, Z6, Z6, 0.0)
        rS = return_map(matS, ε, Z6, Z6, 0.0)
        @test rL[1] == rS[1]           # stress: bit-identical, not just ≈
        @test rL[4] == rS[4]           # ᾱ: bit-identical
        @test Matrix(rL[5]) == Matrix(rS[5])   # tangent: bit-identical
    end
end

@testset "T10 saturation return lands on the updated yield surface" begin
    mat = J2Material(E=SIMO.E, ν=SIMO.ν, σy0=SIMO.σy0, σsat=SIMO.σsat, δ=SIMO.δ, Hiso=SIMO.H)
    s32 = sqrt(1.5)
    for ε in (SVector{6,Float64}(0.02, -0.003, -0.003, 0.004, 0, 0),
              SVector{6,Float64}(0.05, -0.01, -0.01, 0.0, 0.002, 0.0),
              SVector{6,Float64}(0.10, -0.02, -0.02, 0.01, 0.0, 0.0))
        σ, εp, β, ᾱ, _ = return_map(mat, ε, Z6, Z6, 0.0)
        # von Mises of the (deviatoric) stress must equal σy(ᾱ_new): the Newton
        # consistency condition, i.e. the point sits exactly on the yield surface.
        p = (σ[1] + σ[2] + σ[3]) / 3
        s = σ - SVector{6,Float64}(p, p, p, 0, 0, 0)
        q = s32 * sqrt(s[1]^2+s[2]^2+s[3]^2 + 2*(s[4]^2+s[5]^2+s[6]^2))
        @test q ≈ yield_stress(mat, ᾱ) rtol=1e-12
        @test ᾱ > 0                    # genuinely plastic
    end
end

@testset "T11 uniaxial saturation curve vs independent closed form" begin
    mat = J2Material(E=SIMO.E, ν=SIMO.ν, σy0=SIMO.σy0, σsat=SIMO.σsat, δ=SIMO.δ, Hiso=SIMO.H)
    E = SIMO.E
    # uniaxial-STRESS drive (Newton on lateral strains to zero σ_yy=σ_zz), same
    # pattern as T2, incrementally so the plastic path is followed correctly.
    function uniaxial(mat, εxx; nsteps=80)
        εp = Z6; β = Z6; ᾱ = 0.0; εlat = 0.0
        for n in 1:nsteps
            exx = εxx * n / nsteps
            for _ in 1:60
                ε = SVector{6,Float64}(exx, εlat, εlat, 0, 0, 0)
                σ, _, _, _, D = return_map(mat, ε, εp, β, ᾱ)
                r = σ[2]
                εlat -= r / (D[2,2] + D[2,3])
                abs(r) < 1e-10 && break
            end
            ε = SVector{6,Float64}(exx, εlat, εlat, 0, 0, 0)
            σ, εp, β, ᾱ, _ = return_map(mat, ε, εp, β, ᾱ)
        end
        ε = SVector{6,Float64}(εxx, εlat, εlat, 0, 0, 0)
        σ, _, _, ᾱ, _ = return_map(mat, ε, εp, β, ᾱ)
        return σ[1], abs(σ[2]), ᾱ
    end
    # Independent reference: in uniaxial stress past yield, σ = σy(εp_axial) with
    # εp_axial = εxx − σ/E (and ᾱ = εp_axial). Solve this scalar fixed point
    # directly, with NO reference to return_map's internals.
    function uniaxial_ref(εxx)
        σy(a) = SIMO.σy0 + (SIMO.σsat - SIMO.σy0)*(1 - exp(-SIMO.δ*a)) + SIMO.H*a
        σ = SIMO.σy0
        for _ in 1:200
            εp = εxx - σ/E
            εp < 0 && (εp = 0.0)
            σ = 0.5σ + 0.5*σy(εp)     # damped fixed point (robust)
        end
        return σ
    end
    for εxx in (0.01, 0.03, 0.08, 0.15)
        σ1, σ2, ᾱ = uniaxial(mat, εxx)
        σref = uniaxial_ref(εxx)
        @test σ1 ≈ σref rtol=1e-4
        @test σ2 < 1e-5 * σ1           # uniaxial stress: lateral ≈ 0
        @test σ1 < SIMO.σsat + SIMO.H*ᾱ + 1e-6   # never exceeds the saturated+tail bound
    end
end

@testset "T12 saturation consistent tangent vs finite difference" begin
    mat = J2Material(E=SIMO.E, ν=SIMO.ν, σy0=SIMO.σy0, σsat=SIMO.σsat, δ=SIMO.δ, Hiso=SIMO.H)
    states = (
        SVector{6,Float64}(0.02, 0, 0, 0, 0, 0),
        SVector{6,Float64}(0.03, -0.006, 0, 0.004, 0.002, 0),
        SVector{6,Float64}(0.05, -0.012, 0.004, 0.006, 0.001, 0.003),
        SVector{6,Float64}(0.0, 0.0, 0.0, 0.02, 0, 0),
    )
    for ε in states
        _, _, _, _, D = return_map(mat, ε, Z6, Z6, 0.0)
        h = 1e-7
        Dfd = zeros(6, 6)
        for j in 1:6
            e = SVector{6,Float64}(ntuple(k -> k == j ? h : 0.0, 6))
            sp = return_map(mat, ε + e, Z6, Z6, 0.0)[1]
            sm = return_map(mat, ε - e, Z6, Z6, 0.0)[1]
            Dfd[:, j] = (sp - sm) / (2h)
        end
        @test norm(Matrix(D) - Dfd) / norm(Matrix(D)) <= 1e-6
    end
    # tangent must also stay symmetric (associative J2, isotropic hardening)
    D = return_map(mat, SVector{6,Float64}(0.04,-0.01,0,0.005,0.002,0.001), Z6, Z6, 0.0)[5]
    @test norm(Matrix(D) - Matrix(D)') <= 1e-12 * norm(Matrix(D))
end

@testset "T13 monotone, saturating yield growth" begin
    mat = J2Material(E=SIMO.E, ν=SIMO.ν, σy0=SIMO.σy0, σsat=SIMO.σsat, δ=SIMO.δ, Hiso=SIMO.H)
    @test yield_stress(mat, 0.0) ≈ SIMO.σy0
    # monotone increasing, and the Voce part is bounded by (σsat−σy0)
    as = 0.0:0.02:1.0
    ys = [yield_stress(mat, a) for a in as]
    @test all(diff(ys) .> 0)                     # strictly increasing
    voce(a) = yield_stress(mat, a) - SIMO.H*a     # strip the linear tail
    @test voce(10.0) ≈ SIMO.σsat rtol=1e-5        # saturates to σsat far out
    # yield_slope is the analytic derivative
    a = 0.05; dh = 1e-6
    fd = (yield_stress(mat, a+dh) - yield_stress(mat, a-dh))/(2dh)
    @test yield_slope(mat, a) ≈ fd rtol=1e-6
end

@testset "T14 saturation return_map allocation-free" begin
    mat = J2Material(E=SIMO.E, ν=SIMO.ν, σy0=SIMO.σy0, σsat=SIMO.σsat, δ=SIMO.δ, Hiso=SIMO.H)
    ε = SVector{6,Float64}(0.05, -0.012, 0.004, 0.006, 0.001, 0.003)
    function alloc_test(mat, ε)
        return_map(mat, ε, Z6, Z6, 0.0)          # warmup (plastic branch = Newton)
        return @allocated return_map(mat, ε, Z6, Z6, 0.0)
    end
    @test alloc_test(mat, ε) == 0
end

@testset "T15 construction guards" begin
    # saturation + kinematic hardening is not yet supported ⇒ rejected
    @test_throws ArgumentError J2Material(E=SIMO.E, ν=SIMO.ν, σy0=SIMO.σy0,
                                          σsat=SIMO.σsat, δ=SIMO.δ, Hiso=SIMO.H, Hkin=100.0)
    # σsat < σy0 is unphysical (yield would drop below initial) ⇒ rejected
    @test_throws ArgumentError J2Material(E=210e3, ν=0.3, σy0=450.0, σsat=400.0, δ=10.0)
    # δ < 0 ⇒ rejected
    @test_throws ArgumentError J2Material(E=210e3, ν=0.3, σy0=450.0, σsat=715.0, δ=-1.0)
    # linear-only construction (no saturation kwargs) still works and is linear
    m = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0)
    @test m.σsat == m.σy0 && m.δ == 0.0
end
