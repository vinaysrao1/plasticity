# Material kernel unit tests (DESIGN §8.1, T1–T8).

using PlasticityFEM
using PlasticityFEM.Materials
using StaticArrays
using LinearAlgebra
using Test

const Z6 = zero(SVector{6,Float64})

@testset "T1 elastic round-trip" begin
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)
    # strain well below yield
    ε = SVector{6,Float64}(1e-4, -2e-5, 3e-5, 1e-5, -1e-5, 2e-5)
    σ, εp, β, ᾱ, D = return_map(mat, ε, Z6, Z6, 0.0)
    @test σ ≈ mat.Cmat * ε rtol=1e-12
    @test εp == Z6
    @test β == Z6
    @test ᾱ == 0.0
    @test Matrix(D) ≈ Matrix(mat.Cmat) rtol=1e-12
end

@testset "T2 uniaxial post-yield analytical curve" begin
    E = 210e3; σy0 = 250.0; Hiso = 1000.0
    mat = J2Material(E=E, ν=0.3, σy0=σy0, Hiso=Hiso)
    εy = σy0 / E
    Ht = E * Hiso / (E + Hiso)          # uniaxial tangent (DESIGN T2)
    # drive a single point in uniaxial STRESS by Newton on lateral strains
    function uniaxial(mat, εxx; nsteps=50)
        εp = Z6; β = Z6; ᾱ = 0.0; εlat = 0.0
        for n in 1:nsteps
            exx = εxx * n / nsteps
            for _ in 1:60
                ε = SVector{6,Float64}(exx, εlat, εlat, 0, 0, 0)
                σ, _, _, _, D = return_map(mat, ε, εp, β, ᾱ)
                r = σ[2]
                εlat -= r / (D[2,2] + D[2,3])
                abs(r) < 1e-11 && break
            end
            ε = SVector{6,Float64}(exx, εlat, εlat, 0, 0, 0)
            σ, εp, β, ᾱ, _ = return_map(mat, ε, εp, β, ᾱ)
        end
        ε = SVector{6,Float64}(εxx, εlat, εlat, 0, 0, 0)
        return return_map(mat, ε, εp, β, ᾱ)[1]
    end
    for εxx in (0.004, 0.007, 0.01)
        σ = uniaxial(mat, εxx)
        σ_an = σy0 + Ht * (εxx - εy)
        @test σ[1] ≈ σ_an rtol=1e-6
        @test abs(σ[2]) < 1e-6 * σ[1]   # uniaxial stress: lateral ≈ 0
    end
end

@testset "T3 perfect plasticity cap" begin
    σy0 = 250.0
    mat = J2Material(E=210e3, ν=0.3, σy0=σy0, Hiso=0.0, Hkin=0.0)
    # uniaxial-stress drive; σ must asymptote to σy0 and never exceed it
    εp = Z6; β = Z6; ᾱ = 0.0; εlat = 0.0
    σxx = 0.0
    for n in 1:200
        exx = 0.02 * n / 200
        for _ in 1:60
            ε = SVector{6,Float64}(exx, εlat, εlat, 0, 0, 0)
            σ, _, _, _, D = return_map(mat, ε, εp, β, ᾱ)
            d = D[2,2] + D[2,3]
            d == 0 && break          # fully plastic, D deviatoric singular in lat
            εlat -= σ[2] / d
            abs(σ[2]) < 1e-9 && break
        end
        ε = SVector{6,Float64}(exx, εlat, εlat, 0, 0, 0)
        σ, εp, β, ᾱ, _ = return_map(mat, ε, εp, β, ᾱ)
        σxx = σ[1]
        @test σxx <= σy0 * (1 + 1e-10)
    end
    @test σxx ≈ σy0 rtol=1e-3
end

@testset "T4 kinematic hardening / Bauschinger" begin
    σy0 = 250.0; Hkin = 2000.0
    mat = J2Material(E=210e3, ν=0.3, σy0=σy0, Hiso=0.0, Hkin=Hkin)
    # uniaxial-stress monotonic load in +x then reverse; record forward yield
    # stress on unload and the reverse-yield stress; their gap is 2σy0.
    function uniaxial_path(mat, εpath)
        εp = Z6; β = Z6; ᾱ = 0.0; εlat = 0.0
        σxxs = Float64[]; ᾱs = Float64[]
        for exx in εpath
            for _ in 1:80
                ε = SVector{6,Float64}(exx, εlat, εlat, 0, 0, 0)
                σ, _, _, _, D = return_map(mat, ε, εp, β, ᾱ)
                d = D[2,2] + D[2,3]
                εlat -= σ[2] / d
                abs(σ[2]) < 1e-10 && break
            end
            ε = SVector{6,Float64}(exx, εlat, εlat, 0, 0, 0)
            σ, εp, β, ᾱ, _ = return_map(mat, ε, εp, β, ᾱ)
            push!(σxxs, σ[1]); push!(ᾱs, ᾱ)
        end
        return σxxs, ᾱs
    end
    # forward to εxx=0.01, then reverse down to −0.01. Use fine increments so the
    # discrete reverse-yield point (which lands between samples) resolves the
    # 2σy0 elastic span tightly.
    fwd = range(0, 0.01, length=500)
    rev = range(0.01, -0.01, length=1000)
    σf, ᾱf = uniaxial_path(mat, vcat(collect(fwd), collect(rev)))
    σpeak = maximum(σf)
    # reverse yield: first point on the reversal where plastic flow restarts.
    # detect by where σ deviates from the elastic unload line by > tol.
    # Simpler robust check: the elastic span (peak stress − reverse yield stress)
    # equals 2σy0 for pure kinematic hardening (Bauschinger).
    # Find reverse yield stress = stress at which ᾱ starts increasing again.
    n_fwd = length(fwd)
    rstart = 0
    for k in (n_fwd+2):length(σf)
        if ᾱf[k] > ᾱf[k-1] + 1e-12
            rstart = k; break
        end
    end
    @test rstart > 0
    σ_revyield = σf[rstart-1]    # last elastic point before reverse flow
    @test (σpeak - σ_revyield) ≈ 2σy0 rtol=2e-2
    # back stress magnitude grew as (2/3)Hkin·ᾱ in axial terms:
    # at peak, β_axial deviatoric; check von Mises of β ≈ Hkin*εp_eq*(2/3)*?
    # equivalent check: peak stress = σy0 + back-stress contribution
    @test σpeak > σy0    # hardened above initial yield
end

@testset "T5 consistent tangent vs finite difference" begin
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=800.0)
    states = (
        SVector{6,Float64}(0.01, 0, 0, 0, 0, 0),
        SVector{6,Float64}(0.005, -0.001, 0, 0.004, 0.002, 0),
        SVector{6,Float64}(0.003, -0.002, 0.001, 0.002, 0.001, 0.0015),
        SVector{6,Float64}(0.0, 0.0, 0.0, 0.006, 0, 0),
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
        @test norm(Matrix(D) - Dfd) / norm(Matrix(D)) <= 1e-5
    end
end

@testset "T6 symmetry of D_alg" begin
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=800.0)
    for ε in (SVector{6,Float64}(0.01,0,0,0,0,0),
              SVector{6,Float64}(0.004,-0.001,0,0.003,0.001,0.002))
        D = return_map(mat, ε, Z6, Z6, 0.0)[5]
        @test norm(Matrix(D) - Matrix(D)') <= 1e-12 * norm(Matrix(D))
    end
end

@testset "T7 plastic incompressibility" begin
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)
    ε = SVector{6,Float64}(0.01, -0.002, 0.001, 0.003, 0, 0)
    σ, εp, β, ᾱ, _ = return_map(mat, ε, Z6, Z6, 0.0)
    @test εp[1] + εp[2] + εp[3] ≈ 0.0 atol=1e-12
    # mean stress unaffected by plastic correction: p(σ) == p(σ_trial)
    σ_tr = mat.Cmat * ε
    p = (σ[1] + σ[2] + σ[3]) / 3
    p_tr = (σ_tr[1] + σ_tr[2] + σ_tr[3]) / 3
    @test p ≈ p_tr atol=1e-9
end

@testset "T8 return_map allocation" begin
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0, Hkin=500.0)
    ε = SVector{6,Float64}(0.01, 0, 0, 0, 0, 0)
    function alloc_test(mat, ε)
        return_map(mat, ε, Z6, Z6, 0.0)   # warmup
        return @allocated return_map(mat, ε, Z6, Z6, 0.0)
    end
    @test alloc_test(mat, ε) == 0
end
