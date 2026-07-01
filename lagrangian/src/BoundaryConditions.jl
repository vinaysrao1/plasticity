"""
    BoundaryConditions

Dirichlet (displacement) and Neumann (nodal force) boundary conditions, plus
symmetric Dirichlet imposition. See DESIGN.md §4.5, §5.1.
"""
module BoundaryConditions

using SparseArrays

export DirichletBC, NeumannBC, impose_dirichlet!, assemble_neumann!

"""
    DirichletBC

Constrained global DOFs and their prescribed values. `ramp=true` scales values
by the load factor λ; `ramp=false` applies fully at step 1 (DESIGN §4.5).
"""
struct DirichletBC
    dofs::Vector{Int}
    values::Vector{Float64}
    ramp::Bool
end

DirichletBC() = DirichletBC(Int[], Float64[], true)

"""
    NeumannBC

Loaded global DOFs and nodal force magnitudes (ramped by λ) (DESIGN §4.5).
"""
struct NeumannBC
    dofs::Vector{Int}
    values::Vector{Float64}
end

NeumannBC() = NeumannBC(Int[], Float64[])

"""
    assemble_neumann!(Fext, bc, λ)

Scatter ramped nodal forces into the external force vector (DESIGN §5.1).
"""
function assemble_neumann!(Fext::Vector{Float64}, bc::NeumannBC, λ::Float64)
    @inbounds for i in eachindex(bc.dofs)
        Fext[bc.dofs[i]] += λ * bc.values[i]
    end
    return Fext
end

"""
    impose_dirichlet!(K, R, bc, λ, U)

Impose Dirichlet BCs keeping K symmetric: for each constrained DOF d with
prescribed value g (the *target* nodal value, scaled by λ if ramp), set the
residual-system so that the Newton correction drives U[d] → g while moving the
known column to the RHS and placing 1 on the diagonal (DESIGN §5, §7).

Operates on the residual system K·δU = −R. The correction at a constrained DOF
must be δU[d] = g − U[d]. We enforce this with symmetric row/col elimination.
"""
function impose_dirichlet!(K::SparseMatrixCSC{Float64,<:Integer}, R::Vector{Float64},
                           bc::DirichletBC, λ::Float64, U::Vector{Float64})
    nzval = K.nzval
    rowval = K.rowval
    colptr = K.colptr
    n = size(K, 1)

    # prescribed correction value per constrained dof
    isbc = falses(n)
    δg = zeros(n)
    @inbounds for i in eachindex(bc.dofs)
        d = bc.dofs[i]
        g = bc.ramp ? λ * bc.values[i] : bc.values[i]
        isbc[d] = true
        δg[d] = g - U[d]
    end

    # Move known columns to RHS: for every column d that is constrained,
    # R[r] += K[r,d] * δg[d] for all rows r (then zero that column).
    # Symmetric elimination: zero constrained rows & columns, 1 on diagonal,
    # set R[d] = -δg[d] so the solve yields δU[d] = δg[d].
    @inbounds for col in 1:n
        for k in colptr[col]:(colptr[col+1]-1)
            row = rowval[k]
            if isbc[col] && !isbc[row]
                # known column contributes to RHS of free rows
                R[row] += nzval[k] * δg[col]
            end
        end
    end

    # Now zero rows and columns of constrained dofs, set diagonal to 1.
    @inbounds for col in 1:n
        for k in colptr[col]:(colptr[col+1]-1)
            row = rowval[k]
            if isbc[col] || isbc[row]
                nzval[k] = (row == col && isbc[row]) ? 1.0 : 0.0
            end
        end
    end

    # RHS for constrained dofs: K δU = −R, want δU[d] = δg[d] ⇒ R[d] = −δg[d]
    @inbounds for i in eachindex(bc.dofs)
        d = bc.dofs[i]
        R[d] = -δg[d]
    end
    return K, R
end

end # module
