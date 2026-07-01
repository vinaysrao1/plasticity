# Visualization / VTK export tests.

using PlasticityFEM
using PlasticityFEM.Materials
using StaticArrays
using LinearAlgebra
using Test

@testset "von_mises matches Voigt definition" begin
    # uniaxial stress σxx=S → von Mises = |S|
    @test von_mises(SVector{6,Float64}(123.0, 0, 0, 0, 0, 0)) ≈ 123.0
    # hydrostatic stress → von Mises = 0
    @test von_mises(SVector{6,Float64}(50, 50, 50, 0, 0, 0)) ≈ 0.0 atol = 1e-12
    # pure shear τ → von Mises = √3·τ
    @test von_mises(SVector{6,Float64}(0, 0, 0, 10.0, 0, 0)) ≈ sqrt(3) * 10.0
end

@testset "gauss_strain reproduces uniaxial strain field" begin
    # roller cube pulled to ε_xx = 0.001 (elastic): every GP strain must match.
    mesh = box_mesh(1.0, 1.0, 1.0, 2, 2, 2)
    mat = J2Material(E=210e3, ν=0.3, σy0=1e9)   # elastic
    model = Model(mesh, mat)
    fix!(model, on_face(mesh, :xmin), :x)
    fix!(model, on_face(mesh, :ymin), :y)
    fix!(model, on_face(mesh, :zmin), :z)
    prescribe!(model, on_face(mesh, :xmax), :x, 0.001)
    solve!(model; nsteps=1, tol=1e-10)

    ε = gauss_strain(model)
    @test size(ε) == (6, mesh.nelem * 8)
    # uniaxial-stress state ⇒ εxx = 0.001, εyy = εzz = -ν εxx, shears ≈ 0
    @test all(abs.(ε[1, :] .- 0.001) .< 1e-9)
    @test all(abs.(ε[2, :] .+ 0.3 * 0.001) .< 1e-9)
    @test all(abs.(ε[3, :] .+ 0.3 * 0.001) .< 1e-9)
    @test maximum(abs, ε[4:6, :]) < 1e-10
end

@testset "write_vtu produces a valid, consistent file" begin
    mesh = box_mesh(2.0, 1.0, 1.0, 3, 2, 2)
    mat = J2Material(E=210e3, ν=0.3, σy0=250.0, Hiso=1000.0)
    model = Model(mesh, mat)
    fix!(model, on_face(mesh, :xmin), :x)
    fix!(model, on_face(mesh, :ymin), :y)
    fix!(model, on_face(mesh, :zmin), :z)
    prescribe!(model, on_face(mesh, :xmax), :x, 0.01)   # into the plastic regime
    solve!(model; nsteps=10)

    dir = mktempdir()
    path = write_vtu(joinpath(dir, "out"), model)        # extension appended
    @test endswith(path, ".vtu")
    @test isfile(path)
    txt = read(path, String)

    # structural sanity
    @test occursin("type=\"UnstructuredGrid\"", txt)
    @test occursin("NumberOfPoints=\"$(mesh.nnodes)\"", txt)
    @test occursin("NumberOfCells=\"$(mesh.nelem)\"", txt)
    for fld in ("Displacement", "Stress", "Strain", "VonMises", "MeanStress", "EqPlasticStrain")
        @test occursin("Name=\"$fld\"", txt)
    end
    # one VTK_HEXAHEDRON (type 12) per element
    @test count(==("12"), strip.(split(txt, '\n'))) == mesh.nelem

    # connectivity / offsets must match the mesh exactly (0-based, VTK ordering).
    # This guards the writer's most dangerous silent failure: a permuted or
    # off-by-one Hex8 ordering would distort every cell in ParaView yet pass the
    # name/count checks above.
    between(name) = match(Regex("Name=\"$name\"[^>]*>(.*?)</DataArray>", "s"), txt).captures[1]
    conn = parse.(Int, split(between("connectivity")))
    @test conn == vec(mesh.elements .- 1)                    # 8 nodes/cell, 0-based
    offs = parse.(Int, split(between("offsets")))
    @test offs == collect(8:8:8*mesh.nelem)
    # plastic problem ⇒ some yielding recorded in the export
    @test occursin("EqPlasticStrain", txt)
    @test maximum(equivalent_plastic_strain(model)) > 0

    # extension already present → not doubled
    path2 = write_vtu(joinpath(dir, "out2.vtu"), model)
    @test endswith(path2, ".vtu") && !endswith(path2, ".vtu.vtu")
end
