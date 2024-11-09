abstract type SP2TDistribution{T} end

struct DNormal{T<:AbstractFloat,V<:AbstractVector{T}} <: SP2TDistribution{T}
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

Distributions.params(n::DNormal) = n.μ, n.σ