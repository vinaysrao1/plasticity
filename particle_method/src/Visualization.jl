"""
    Visualization

Dependency-free VTK export (`.vtu`, XML unstructured grid, VTK point cloud —
`VTK_VERTEX` cells) for ParaView / VisIt, modeled on `lagrangian`'s
`Visualization.jl` writer style. No external packages; plain ASCII XML.
"""
module Visualization

using StaticArrays
using ..ParticlesMod: Particles
using PlasticityFEM.FiniteStrain: voigt_to_sym3
using PlasticityFEM.Visualization: von_mises

export write_particles_vtu

const _VOIGT_NAMES = ("xx", "yy", "zz", "xy", "yz", "zx")

function _write_vector(io, name, np, f)
    println(io, "        <DataArray type=\"Float64\" Name=\"$name\" ",
            "NumberOfComponents=\"3\" format=\"ascii\">")
    @inbounds for p in 1:np
        v = f(p)
        println(io, "          ", v[1], " ", v[2], " ", v[3])
    end
    println(io, "        </DataArray>")
end

function _write_scalar(io, name, np, f)
    println(io, "        <DataArray type=\"Float64\" Name=\"$name\" ",
            "NumberOfComponents=\"1\" format=\"ascii\">")
    @inbounds for p in 1:np
        println(io, "          ", f(p))
    end
    println(io, "        </DataArray>")
end

function _write_voigt6(io, name, np, f)
    comps = join(("ComponentName$(i-1)=\"$(_VOIGT_NAMES[i])\"" for i in 1:6), " ")
    println(io, "        <DataArray type=\"Float64\" Name=\"$name\" ",
            "NumberOfComponents=\"6\" $comps format=\"ascii\">")
    @inbounds for p in 1:np
        v = f(p)
        println(io, "          ", v[1], " ", v[2], " ", v[3], " ", v[4], " ", v[5], " ", v[6])
    end
    println(io, "        </DataArray>")
end

"""
    write_particles_vtu(filename, particles) -> String

Write a `.vtu` VTK point cloud (`VTK_VERTEX` cells, one per particle) for
ParaView / VisIt and return the path written. Point data: `Velocity`,
`CauchyStress` (6-Voigt, `σ = τ/J`), `VonMises`, `MeanStress`,
`EqPlasticStrain`, `J` (`det F`), `Mass`.
"""
function write_particles_vtu(filename::AbstractString, pts::Particles)
    fn = endswith(filename, ".vtu") ? String(filename) : filename * ".vtu"
    np = length(pts)

    open(fn, "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io, "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">")
        println(io, "  <UnstructuredGrid>")
        println(io, "    <Piece NumberOfPoints=\"$np\" NumberOfCells=\"$np\">")

        println(io, "      <Points>")
        println(io, "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">")
        @inbounds for p in 1:np
            x = pts.x[p]
            println(io, "          ", x[1], " ", x[2], " ", x[3])
        end
        println(io, "        </DataArray>")
        println(io, "      </Points>")

        println(io, "      <Cells>")
        println(io, "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">")
        @inbounds for p in 1:np
            println(io, "          ", p - 1)
        end
        println(io, "        </DataArray>")
        println(io, "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">")
        @inbounds for p in 1:np
            println(io, "          ", p)
        end
        println(io, "        </DataArray>")
        println(io, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">")
        for _ in 1:np
            println(io, "          1")   # VTK_VERTEX
        end
        println(io, "        </DataArray>")
        println(io, "      </Cells>")

        println(io, "      <PointData Vectors=\"Velocity\" Scalars=\"VonMises\">")
        _write_vector(io, "Velocity", np, p -> pts.v[p])
        _write_voigt6(io, "CauchyStress", np, p -> pts.τ[p] / pts.J[p])
        _write_scalar(io, "VonMises", np, p -> von_mises(pts.τ[p] / pts.J[p]))
        _write_scalar(io, "MeanStress", np, p -> begin
            σ = pts.τ[p] / pts.J[p]
            (σ[1] + σ[2] + σ[3]) / 3
        end)
        _write_scalar(io, "EqPlasticStrain", np, p -> pts.ᾱ[p])
        _write_scalar(io, "J", np, p -> pts.J[p])
        _write_scalar(io, "Mass", np, p -> pts.m[p])
        println(io, "      </PointData>")

        println(io, "    </Piece>")
        println(io, "  </UnstructuredGrid>")
        println(io, "</VTKFile>")
    end
    return fn
end

end # module
