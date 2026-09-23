# Raw ccall bindings for primitives SlepcWrap 0.1.3 / PetscWrap 0.1.5 do not wrap.
# Signatures follow PETSc/SLEPc C and PetscWrap's own ccall convention
# (CVec/CMat/ST == Ptr{Cvoid}; wrappers cconvert to their handle). Exercised by the
# SLEPc coverage workflow; confirm against the installed PETSc/SLEPc headers.

const CVecScatter = Ptr{Cvoid}
const CST = Ptr{Cvoid}
const SCATTER_FORWARD = Cint(0)
const _INSERT_VALUES_C = Cint(1)   # PETSc InsertMode INSERT_VALUES
const MAT_INITIAL_MATRIX = Cint(0)   # PETSc MatReuse

"""Throw if a PETSc/SLEPc call returned a nonzero error code. Used instead of
`@assert`, which may be compiled out, around calls whose side effects matter."""
function _check_petsc(err, call::AbstractString)
    iszero(err) || error("$call failed with PETSc error code $err")
    return nothing
end

"""Distributed C = A*B via PETSc MatMatMult (unwrapped; PetscWrap 0.1.5 has no
wrapper). `fill = PETSC_DEFAULT (-2.0)` lets PETSc estimate the product's fill ratio.
The cconvert for `PetscWrap.CMat` is `mat.ptr[]`, so passing the `PetscMat` wrappers
`A`/`B` directly into the `CMat` ccall arguments is valid. Returns a freshly created
`PetscMat` on `A`'s communicator owning the product; caller must `MatDestroy` it."""
function _mat_mat_mult(A::PetscWrap.PetscMat, B::PetscWrap.PetscMat)
    C = PetscWrap.PetscMat(A.comm)
    PR = PetscWrap.PetscReal
    _check_petsc(ccall((:MatMatMult, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
        (PetscWrap.CMat, PetscWrap.CMat, Cint, PetscWrap.PetscReal, Ptr{PetscWrap.CMat}),
        A, B, MAT_INITIAL_MATRIX, PR(-2.0), C.ptr), "MatMatMult")
    return C
end

"""Set the requested eigenpair count on an EPS (SlepcWrap 0.1.3 has no wrapper).
`EPSSetDimensions(eps, nev, ncv=PETSC_DECIDE, mpd=PETSC_DECIDE)`."""
function _eps_set_dimensions(eps, nev::Integer)
    PD = PetscWrap.PETSC_DECIDE
    err = ccall((:EPSSetDimensions, SlepcWrap.libslepc), PetscWrap.PetscErrorCode,
                (Ptr{Cvoid}, PetscWrap.PetscInt, PetscWrap.PetscInt, PetscWrap.PetscInt),
                eps.ptr[], PetscWrap.PetscInt(nev), PD, PD)
    _check_petsc(err, "EPSSetDimensions")
    return nothing
end

"""Set EPS convergence controls (SlepcWrap 0.1 has no setter wrapper)."""
function _eps_set_tolerances(eps, tol::Real, maxiter::Integer)
    err = ccall((:EPSSetTolerances, SlepcWrap.libslepc), PetscWrap.PetscErrorCode,
                (Ptr{Cvoid}, PetscWrap.PetscReal, PetscWrap.PetscInt),
                eps.ptr[], PetscWrap.PetscReal(tol), PetscWrap.PetscInt(maxiter))
    _check_petsc(err, "EPSSetTolerances")
    return nothing
end

"""Return the EPS spectral transformation handle (SlepcWrap 0.1 has no `EPSGetST`)."""
function _eps_get_st(eps)
    st = Ref{CST}()
    _check_petsc(ccall((:EPSGetST, SlepcWrap.libslepc), PetscWrap.PetscErrorCode,
                       (Ptr{Cvoid}, Ptr{CST}), eps.ptr[], st), "EPSGetST")
    return st[]
end

"""Set the EPS spectral transformation type, e.g. `"sinvert"` (`STSetType`)."""
function _eps_set_st_type(eps, type::AbstractString)
    _check_petsc(ccall((:STSetType, SlepcWrap.libslepc), PetscWrap.PetscErrorCode,
                       (CST, Cstring), _eps_get_st(eps), type), "STSetType")
    return nothing
end

"""Return the EPS spectral transformation type name (`STGetType`)."""
function _eps_get_st_type(eps)
    t = Ref{Ptr{Cchar}}(C_NULL)
    _check_petsc(ccall((:STGetType, SlepcWrap.libslepc), PetscWrap.PetscErrorCode,
                       (CST, Ptr{Ptr{Cchar}}), _eps_get_st(eps), t), "STGetType")
    return t[] == C_NULL ? "" : unsafe_string(t[])
end

"""Gather a distributed PETSc vector to rank 0 as a `Vector{ComplexF64}`: full
length-`n` on rank 0, empty elsewhere. Wraps VecScatterCreateToZero / Begin / End /
VecGetArray / VecScatterDestroy / VecDestroy."""
function _vec_scatter_to_zero(v::PetscWrap.PetscVec)
    ctx = Ref{CVecScatter}()
    seq = Ref{PetscWrap.CVec}()
    _check_petsc(ccall((:VecScatterCreateToZero, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
        (PetscWrap.CVec, Ptr{CVecScatter}, Ptr{PetscWrap.CVec}), v, ctx, seq),
        "VecScatterCreateToZero")
    try
        _check_petsc(ccall((:VecScatterBegin, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
            (CVecScatter, PetscWrap.CVec, PetscWrap.CVec, Cint, Cint),
            ctx[], v, seq[], _INSERT_VALUES_C, SCATTER_FORWARD), "VecScatterBegin")
        _check_petsc(ccall((:VecScatterEnd, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
            (CVecScatter, PetscWrap.CVec, PetscWrap.CVec, Cint, Cint),
            ctx[], v, seq[], _INSERT_VALUES_C, SCATTER_FORWARD), "VecScatterEnd")

        nref = Ref{PetscWrap.PetscInt}()
        _check_petsc(ccall((:VecGetSize, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
              (PetscWrap.CVec, Ref{PetscWrap.PetscInt}), seq[], nref), "VecGetSize")
        n = Int(nref[])
        out = Vector{ComplexF64}(undef, n)
        if n > 0
            aref = Ref{Ptr{PetscWrap.PetscScalar}}()
            _check_petsc(ccall((:VecGetArray, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
                  (PetscWrap.CVec, Ref{Ptr{PetscWrap.PetscScalar}}), seq[], aref), "VecGetArray")
            arr = unsafe_wrap(Array, aref[], n; own=false)
            out .= ComplexF64.(arr)
            _check_petsc(ccall((:VecRestoreArray, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
                  (PetscWrap.CVec, Ref{Ptr{PetscWrap.PetscScalar}}), seq[], aref), "VecRestoreArray")
        end
        return out
    finally
        _check_petsc(ccall((:VecScatterDestroy, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
                           (Ptr{CVecScatter},), ctx), "VecScatterDestroy")
        _check_petsc(ccall((:VecDestroy, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
                           (Ptr{PetscWrap.CVec},), seq), "VecDestroy")
    end
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
    _check_petsc(ccall((:MatZeroRows, PetscWrap.libpetsc), PetscWrap.PetscErrorCode,
        (PetscWrap.CMat, PetscWrap.PetscInt, Ptr{PetscWrap.PetscInt}, PetscWrap.PetscScalar, PetscWrap.CVec, PetscWrap.CVec),
        mat, PI(length(idx)), idx, PetscWrap.PetscScalar(0), C_NULL, C_NULL), "MatZeroRows")
    return nothing
end
