# using vec makes GPU sum much faster
# S stands for scratch within this file 

_logℒ(
    W::AbstractArray{Bool,N},
    U::AbstractArray{T,N},
    M::AbstractMatrix{Bool},
    S::AbstractArray{T,N},
) where {T,N} = sum(logexpm1.(U[W.&M])) - sum(U .* M)

function _logℒ(
    W::AbstractArray{UInt16,3},
    U::AbstractArray{T,3},
    M::AbstractMatrix{Bool},
    S::AbstractArray{T,3},
) where {T}
    @. S = W * logexpm1(U) - U
    sum(transpose(reshape(S, length(M), :)) * vec(M))
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

function Δlogℒ!(
    Δlogℒ::AbstractArray{T,N},
    W::AbstractArray{Bool,N},
    U::AbstractArray{T,N},
    V::AbstractArray{T,N},
    M::AbstractMatrix{Bool},
    S::AbstractArray{T,N},
) where {T,N}
    @. S = (U - V) * M
    @. S[W&M] += logexpm1(V[W&M]) - logexpm1(U[W&M])
    sum!(Δlogℒ, S)
end

function Δlogℒ!(
    Δlogℒ::AbstractVector{T},
    W::AbstractArray{UInt16,3},
    U::AbstractArray{T,3},
    V::AbstractArray{T,3},
    M::AbstractMatrix{Bool},
    S::AbstractArray{T,3},
) where {T}
    @. S = W * (logexpm1(V) - logexpm1(U)) - (V - U)
    Δlogℒ .= transpose(reshape(S, length(M), :)) * vec(M)
end

anneal(logℒ::T, 𝑇::T) where {T} = logℒ / 𝑇

anneal!(logℒ::AbstractVector{T}, 𝑇::T) where {T} = logℒ ./= 𝑇