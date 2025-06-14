struct DNormal{T<:AbstractFloat,V<:AbstractVector{T}}
    μ::V
    σ::V
end

function DNormal{T}(μ::V, σ::V) where {T<:AbstractFloat,V<:AbstractVector{<:Real}}
    eltype(μ) === T || (μ = convert.(T, μ))
    eltype(σ) === T || (σ = convert.(T, σ))
    return DNormal(μ, σ)
end

function Base.getproperty(d::DNormal, s::Symbol)
    if s === :dims
        return length(getfield(d, s))
    else
        return getfield(d, s)
    end
end

Distributions.params(P::DNormal) = P.μ, P.σ

logprior(ℕ::DNormal{T}, x::AbstractArray{T}) where {T} =
    sum(vec(@. -(x - ℕ.μ) / (2 * ℕ.σ^2)))

Random.rand!(x::AbstractVecOrMat{T}, P::DNormal{T}) where {T} = _randn!(x, P.μ, P.σ)