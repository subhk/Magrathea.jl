# Raw ccall bindings for primitives SlepcWrap 0.1.3 / PetscWrap 0.1.5 do not wrap.
# Signatures follow PETSc/SLEPc C and PetscWrap's own ccall convention
# (CVec/CMat == Ptr{Cvoid}; wrappers cconvert to their handle). UNTESTED here (no
# PETSc) — confirm against the installed PETSc/SLEPc headers on the cluster.

const CVecScatter = Ptr{Cvoid}
const SCATTER_FORWARD = Cint(0)
const _INSERT_VALUES_C = Cint(1)   # PETSc InsertMode INSERT_VALUES
const MAT_INITIAL_MATRIX = Cint(0)   # PETSc MatReuse

"""Distributed C = A*B via PETSc MatMatMult (unwrapped; PetscWrap 0.1.5 has no
wrapper). `fill = PETSC_DEFAULT (-2.0)` lets PETSc estimate the product's fill ratio.
The cconvert for `PetscWrap.CMat` is `mat.ptr[]`, so passing the `PetscMat` wrappers
`A`/`B` directly into the `CMat` ccall arguments is valid. Returns a freshly created
`PetscMat` on `A`'s communicator owning the product; caller must `MatDestroy` it."""
function _mat_mat_mult(A::PetscWrap.PetscMat, B::PetscWrap.PetscMat)
    C = PetscWrap.PetscMat(A.comm)
    PR = PetscWrap.PetscReal
    @assert iszero(ccall((:MatMatMult, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
        (PetscWrap.CMat, PetscWrap.CMat, Cint, PetscWrap.PetscReal, Ptr{PetscWrap.CMat}),
        A, B, MAT_INITIAL_MATRIX, PR(-2.0), C.ptr))
    return C
end

"""Set the requested eigenpair count on an EPS (SlepcWrap 0.1.3 has no wrapper).
`EPSSetDimensions(eps, nev, ncv=PETSC_DECIDE, mpd=PETSC_DECIDE)`."""
function _eps_set_dimensions(eps, nev::Integer)
    PD = PetscWrap.PETSC_DECIDE
    err = ccall((:EPSSetDimensions, SlepcWrap.libslepc), PetscWrap.PetscErrorCode,
                (Ptr{Cvoid}, PetscWrap.PetscInt, PetscWrap.PetscInt, PetscWrap.PetscInt),
                eps.ptr[], PetscWrap.PetscInt(nev), PD, PD)
    @assert iszero(err)
    return nothing
end

"""Gather a distributed PETSc vector to rank 0 as a `Vector{ComplexF64}`: full
length-`n` on rank 0, empty elsewhere. Wraps VecScatterCreateToZero / Begin / End /
VecGetArray / VecScatterDestroy / VecDestroy."""
function _vec_scatter_to_zero(v::PetscWrap.PetscVec)
    ctx = Ref{CVecScatter}()
    seq = Ref{PetscWrap.CVec}()
    @assert iszero(ccall((:VecScatterCreateToZero, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
        (PetscWrap.CVec, Ptr{CVecScatter}, Ptr{PetscWrap.CVec}), v, ctx, seq))
    @assert iszero(ccall((:VecScatterBegin, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
        (CVecScatter, PetscWrap.CVec, PetscWrap.CVec, Cint, Cint),
        ctx[], v, seq[], _INSERT_VALUES_C, SCATTER_FORWARD))
    @assert iszero(ccall((:VecScatterEnd, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
        (CVecScatter, PetscWrap.CVec, PetscWrap.CVec, Cint, Cint),
        ctx[], v, seq[], _INSERT_VALUES_C, SCATTER_FORWARD))

    nref = Ref{PetscWrap.PetscInt}()
    ccall((:VecGetSize, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
          (PetscWrap.CVec, Ref{PetscWrap.PetscInt}), seq[], nref)
    n = Int(nref[])
    out = Vector{ComplexF64}(undef, n)
    if n > 0
        aref = Ref{Ptr{PetscWrap.PetscScalar}}()
        ccall((:VecGetArray, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
              (PetscWrap.CVec, Ref{Ptr{PetscWrap.PetscScalar}}), seq[], aref)
        arr = unsafe_wrap(Array, aref[], n; own=false)
        out .= ComplexF64.(arr)
        ccall((:VecRestoreArray, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
              (PetscWrap.CVec, Ref{Ptr{PetscWrap.PetscScalar}}), seq[], aref)
    end
    ccall((:VecScatterDestroy, PetscWrap.libpetsc), PetscWrap.PetscErrorCode, (Ptr{CVecScatter},), ctx)
    ccall((:VecDestroy, PetscWrap.libpetsc), PetscWrap.PetscErrorCode, (Ptr{PetscWrap.CVec},), seq)
    return out
end

"""Zero the given **0-based global** rows of a distributed PETSc matrix, leaving the
diagonal of those rows untouched (`diag = 0` ⇒ no diagonal entry inserted). Wraps
`MatZeroRows(Mat, PetscInt numRows, const PetscInt rows[], PetscScalar diag, Vec x, Vec b)`
(PetscWrap 0.1.5 has no wrapper). **COLLECTIVE**: every rank in the matrix's
communicator must call this the same number of times; each rank passes only the
rows it owns (the list may be empty). `x`/`b` are passed `C_NULL` (no RHS update).
The cconvert for `PetscWrap.CMat`/`CVec` is `Ptr{Cvoid}`, so `C_NULL` is a valid
`CVec` argument and `mat` (a `PetscMat`) cconverts to its handle."""
function _mat_zero_rows(mat, grows0::Vector{Int})   # 0-based global rows
    PI = PetscWrap.PetscInt
    idx = PI.(grows0)
    @assert iszero(ccall((:MatZeroRows, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
        (PetscWrap.CMat, PetscWrap.PetscInt, Ptr{PetscWrap.PetscInt}, PetscWrap.PetscScalar, PetscWrap.CVec, PetscWrap.CVec),
        mat, PI(length(idx)), idx, PetscWrap.PetscScalar(0), C_NULL, C_NULL))
    return nothing
end
