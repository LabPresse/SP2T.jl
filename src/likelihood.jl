# using vec makes GPU sum much faster

# logℒ(data::Data, U::AbstractArray{T,N}, S::AbstractArray{T,N}) where {T,N} =
#     _logℒ(data.frames, U, data.filter, data.batchsize, S)

# _logℒ(
#     W::AbstractArray{Bool,N},
#     U::AbstractArray{T,N},
#     F::AbstractMatrix{Bool},
# ) where {T,N} = sum(logexpm1.(U[W.&F])) - sum(U .* F)

function logℒ(
    W::AbstractArray{UInt16,3},
    F::AbstractMatrix{Bool},
    B::Integer,
    U::AbstractArray{T,3},
    Sₐ::AbstractArray{T,3},
    Sᵥ::AbstractVector{T},
) where {T}
    @. Sₐ = W * logexpm1(U) - B * U
    mul!(Sᵥ, transpose(reshape(Sₐ, length(F), :)), vec(F))
    sum(Sᵥ)
end

logℒ(D::Data, A::AuxiliaryVariables) =
    logℒ(D.frames, D.filter, D.batchsize, A.U, A.Sₐ, A.Sᵥ)

# dangerous hack
# function unsafe_Δlogℒ!(
#     logratio::AbstractVector{T},
#     W::AbstractArray{UInt16,N},
#     U::AbstractArray{T,N},
#     V::AbstractArray{T},
#     M::AbstractMatrix{Bool},
# ) where {T}
#     Δlnℒ = (U .- V) .* M
#     @. Δlnℒ[W] += logexpm1(V[W]) - logexpm1(U[W])
#     sum!(logratio, Δlnℒ, init = false)
# end

# function Δlogℒ!(
#     Δlogℒ::AbstractArray{T,N},
#     W::AbstractArray{Bool,N},
#     U::AbstractArray{T,N},
#     V::AbstractArray{T,N},
#     F::AbstractMatrix{Bool},
#     S::AbstractArray{T,N},
# ) where {T,N}
#     @. S = (U - V) * F
#     @. S[W&F] += logexpm1(V[W&F]) - logexpm1(U[W&F])
#     sum!(Δlogℒ, S)
# end

function Δlogℒ!(
    Δlogℒ::AbstractVector{T},
    W::AbstractArray{UInt16,3},
    F::AbstractMatrix{Bool},
    B::Integer,
    U::AbstractArray{T,3},
    V::AbstractArray{T,3},
    S::AbstractArray{T,3},
) where {T}
    @. S = W * (logexpm1(V) - logexpm1(U)) - B * (V - U)
    mul!(Δlogℒ, transpose(reshape(S, length(F), :)), vec(F))
end

Δlogℒ!(D::Data, A::AuxiliaryVariables) =
    Δlogℒ!(A.Sᵥ, D.frames, D.filter, D.batchsize, A.U, A.V, A.Sₐ)

anneal(logℒ::T, 𝑇::T) where {T} = logℒ / 𝑇

anneal!(logℒ::AbstractVector{T}, 𝑇::T) where {T} = logℒ ./= 𝑇