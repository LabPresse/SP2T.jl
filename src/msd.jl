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

logprior(msd::MeanSquaredDisplacement{T,P}) where {T,P<:InverseGamma{T}} =
    -(shape(msd.prior) + 1) * log(msd.value) - scale(msd.prior) / msd.value

logprior_norm(msd::MeanSquaredDisplacement{T,P}) where {T,P} = logpdf(msd.prior, msd.value)

function update!(
    msd::MeanSquaredDisplacement{T},
    displacement²::AbstractArray{T,3},
    𝑇::T,
) where {T}
    Δparams = (length(displacement²), sum(vec(displacement²))) ./ (2 * 𝑇)
    msd.value = _rand(msd.prior, Δparams)
    return msd
end