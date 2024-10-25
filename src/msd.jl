mutable struct MeanSquaredDisplacement{T,P}
    value::T
    prior::P
end

MeanSquaredDisplacement{T}(
    value::Real,
    prior::ContinuousUnivariateDistribution,
) where {T<:AbstractFloat} = MeanSquaredDisplacement(
    convert(T, value),
    unionalltypeof(prior)(convert.(T, params(prior))...),
)

MeanSquaredDisplacement{T}(;
    value::Real,
    prior::ContinuousUnivariateDistribution,
) where {T<:AbstractFloat} = MeanSquaredDisplacement{T}(value, prior)

function _rand(prior::P, Δparams::NTuple{2,T}) where {T,P}
    postparams = params(prior) .+ Δparams
    return rand(P(postparams...))
end

function sample!(msd::MeanSquaredDisplacement{T}, Δx²::AbstractArray{T,3}, 𝑇::T) where {T}
    Δparams = (length(Δx²), sum(vec(Δx²))) ./ (2 * 𝑇)
    msd.value = _rand(msd.prior, Δparams)
    return msd
end