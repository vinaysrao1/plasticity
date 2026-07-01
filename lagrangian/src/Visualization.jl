"""
    Visualization

Dependency-free VTK export (`.vtu`, XML unstructured grid) for ParaView / VisIt.
Writes the mesh with per-node displacements and per-element (Gauss-point-averaged)
stress, strain, von Mises stress, mean stress, and equivalent plastic strain, so
the full 3D field distribution can be inspected interactively.

No external packages are used; the writer emits plain ASCII XML.
"""
module Visualization

using StaticArrays
using ..MeshMod: Mesh
using ..ModelMod: Model, gauss_stress
using ..Elements: element_geometry

export write_vtu, gauss_strain, von_mises

"""
    von_mises(σ) -> Float64

von Mises equivalent stress from a Voigt stress 6-vector `[xx,yy,zz,xy,yz,zx]`
(physical shear components):
`σvm = sqrt( ½[(σxx−σyy)²+(σyy−σzz)²+(σzz−σxx)²] + 3(σxy²+σyz²+σzx²) )`.
"""
@inline function von_mises(σ)
    sxx, syy, szz = σ[1], σ[2], σ[3]
    sxy, syz, szx = σ[4], σ[5], σ[6]
    return sqrt(0.5 * ((sxx - syy)^2 + (syy - szz)^2 + (szz - sxx)^2) +
               3.0 * (sxy^2 + syz^2 + szx^2))
end

"""
    gauss_strain(model) -> Matrix (6 × ngp)

Total small-strain at every Gauss point (engineering-shear Voigt
`[εxx,εyy,εzz,γxy,γyz,γzx]`), computed from the current displacement solution via
ε = B·u_e using the cached element B-matrices (DESIGN §3.4).
"""
function gauss_strain(model::Model)
    mesh = model.mesh
    U = model.U
    edofs = model.sparsity.edofs
    ε = Matrix{Float64}(undef, 6, mesh.nelem * 8)
    @inbounds for e in 1:mesh.nelem
        ue = SVector{24,Float64}(ntuple(i -> U[edofs[i, e]], Val(24)))
        Bs, _ = element_geometry(model.cache, e)
        for g in 1:8
            εg = Bs[g] * ue
            gp = (e - 1) * 8 + g
            for k in 1:6
                ε[k, gp] = εg[k]
            end
        end
    end
    return ε
end

# Average a 6×ngp Gauss-point field to one value per element (6 × nelem).
function _cell_avg6(field::AbstractMatrix{Float64}, nelem::Int)
    out = zeros(6, nelem)
    @inbounds for e in 1:nelem
        for g in 1:8, k in 1:6
            out[k, e] += field[k, (e - 1) * 8 + g]
        end
        for k in 1:6
            out[k, e] /= 8
        end
    end
    return out
end

# Average a length-ngp Gauss-point scalar field to one value per element.
function _cell_avg1(field::AbstractVector{Float64}, nelem::Int)
    out = zeros(nelem)
    @inbounds for e in 1:nelem
        s = 0.0
        for g in 1:8
            s += field[(e - 1) * 8 + g]
        end
        out[e] = s / 8
    end
    return out
end

const _VOIGT_NAMES = ("xx", "yy", "zz", "xy", "yz", "zx")

function _write_cell6(io, name, data, ne)
    comps = join(("ComponentName$(i-1)=\"$(_VOIGT_NAMES[i])\"" for i in 1:6), " ")
    println(io, "        <DataArray type=\"Float64\" Name=\"$name\" ",
            "NumberOfComponents=\"6\" $comps format=\"ascii\">")
    @inbounds for e in 1:ne
        println(io, "          ", data[1, e], " ", data[2, e], " ", data[3, e],
                " ", data[4, e], " ", data[5, e], " ", data[6, e])
    end
    println(io, "        </DataArray>")
end

function _write_cell_scalar(io, name, vals)
    println(io, "        <DataArray type=\"Float64\" Name=\"$name\" ",
            "NumberOfComponents=\"1\" format=\"ascii\">")
    for v in vals
        println(io, "          ", v)
    end
    println(io, "        </DataArray>")
end

"""
    write_vtu(filename, model) -> String

Write an ASCII VTK unstructured-grid file (`.vtu`) for ParaView / VisIt and
return the path written (the `.vtu` extension is appended if missing).

Fields:
- point data — `Displacement` (vector; use ParaView's *Warp By Vector* to see the
  deformed shape);
- cell data (Gauss-point-averaged per element) — `Stress` and `Strain` (Voigt
  6-component, with `xx,yy,zz,xy,yz,zx` component names), `VonMises`,
  `MeanStress`, and `EqPlasticStrain`.

Call after a converged `solve!`: `Stress`/`EqPlasticStrain` come from the
committed state and `Strain` is recomputed from the current displacement `U`, so
the two agree only at a converged solution. Example:

    solve!(model; nsteps=20)
    write_vtu("tension_cube", model)   # -> "tension_cube.vtu", open in ParaView
"""
function write_vtu(filename::AbstractString, model::Model)
    fn = endswith(filename, ".vtu") ? String(filename) : filename * ".vtu"
    mesh = model.mesh
    nn = mesh.nnodes
    ne = mesh.nelem

    # `gauss_stress` reports Cauchy for finite-strain models (Kirchhoff/J),
    # the committed engineering stress for small strain (FINITE_STRAIN §6.4).
    σc = _cell_avg6(gauss_stress(model), ne)
    εc = _cell_avg6(gauss_strain(model), ne)
    ᾱc = _cell_avg1(model.state_committed.ᾱ, ne)
    U = model.U

    open(fn, "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io, "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">")
        println(io, "  <UnstructuredGrid>")
        println(io, "    <Piece NumberOfPoints=\"$nn\" NumberOfCells=\"$ne\">")

        # --- points ---
        println(io, "      <Points>")
        println(io, "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">")
        @inbounds for n in 1:nn
            println(io, "          ", mesh.nodes[1, n], " ", mesh.nodes[2, n], " ", mesh.nodes[3, n])
        end
        println(io, "        </DataArray>")
        println(io, "      </Points>")

        # --- cells (Hex8 = VTK_HEXAHEDRON, type 12; our node order matches VTK) ---
        println(io, "      <Cells>")
        println(io, "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">")
        @inbounds for e in 1:ne
            print(io, "         ")
            for a in 1:8
                print(io, " ", mesh.elements[a, e] - 1)   # 0-based for VTK
            end
            println(io)
        end
        println(io, "        </DataArray>")
        println(io, "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">")
        for e in 1:ne
            println(io, "          ", 8 * e)
        end
        println(io, "        </DataArray>")
        println(io, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">")
        for _ in 1:ne
            println(io, "          12")
        end
        println(io, "        </DataArray>")
        println(io, "      </Cells>")

        # --- point data ---
        println(io, "      <PointData Vectors=\"Displacement\">")
        println(io, "        <DataArray type=\"Float64\" Name=\"Displacement\" NumberOfComponents=\"3\" format=\"ascii\">")
        @inbounds for n in 1:nn
            println(io, "          ", U[3*(n-1)+1], " ", U[3*(n-1)+2], " ", U[3*(n-1)+3])
        end
        println(io, "        </DataArray>")
        println(io, "      </PointData>")

        # --- cell data (element-averaged Gauss-point fields) ---
        println(io, "      <CellData Scalars=\"VonMises\" Tensors=\"Stress\">")
        _write_cell6(io, "Stress", σc, ne)
        _write_cell6(io, "Strain", εc, ne)
        _write_cell_scalar(io, "VonMises", [von_mises(view(σc, :, e)) for e in 1:ne])
        _write_cell_scalar(io, "MeanStress", [(σc[1, e] + σc[2, e] + σc[3, e]) / 3 for e in 1:ne])
        _write_cell_scalar(io, "EqPlasticStrain", ᾱc)
        println(io, "      </CellData>")

        println(io, "    </Piece>")
        println(io, "  </UnstructuredGrid>")
        println(io, "</VTKFile>")
    end
    return fn
end

end # module
