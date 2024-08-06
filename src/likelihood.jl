# using vec makes GPU sum much faster

logℒ(data::Data, U::AbstractArray{T,N}, S::AbstractArray{T,N}) where {T,N} =
    _logℒ(data.frames, U, data.mask, S, data.batchsize)

# _logℒ(
#     W::AbstractArray{Bool,N},
#     U::AbstractArray{T,N},
#     F::AbstractMatrix{Bool},
# ) where {T,N} = sum(logexpm1.(U[W.&F])) - sum(U .* F)

function _logℒ(
    W::AbstractArray{UInt16,3},
    U::AbstractArray{T,3},
    F::AbstractMatrix{Bool},
    𝐴::AbstractArray{T,3},
    B::Integer=1,
) where {T}
    @. 𝐴 = W * logexpm1(U) - B * U
    sum(transpose(reshape(𝐴, length(F), :)) * vec(F))
end

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
    U::AbstractArray{T,3},
    V::AbstractArray{T,3},
    F::AbstractMatrix{Bool},
    𝐴::AbstractArray{T,3},
    B::Integer=1,
) where {T}
    @. 𝐴 = W * (logexpm1(V) - logexpm1(U)) - B * (V - U)
    mul!(Δlogℒ, transpose(reshape(𝐴, length(F), :)), vec(F))
end

anneal(logℒ::T, 𝑇::T) where {T} = logℒ / 𝑇

anneal!(logℒ::AbstractVector{T}, 𝑇::T) where {T} = logℒ ./= 𝑇