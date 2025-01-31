# MeanSquaredDisplacement{T}(
#     value::Real,
#     prior::ContinuousUnivariateDistribution,
#     fixed::Bool,
# ) where {T<:AbstractFloat} = MeanSquaredDisplacement(
#     convert(T, value),
#     unionalltypeof(prior)(convert.(T, params(prior))...),
#     fixed,
# )

MeanSquaredDisplacement{T}(;
    guess::Real,
    prior::ContinuousUnivariateDistribution,
    fixed::Bool = false,
) where {T<:AbstractFloat} = MeanSquaredDisplacement(
    convert(T, guess),
    unionalltypeof(prior)(convert.(T, params(prior))...),
    fixed,
)

# MeanSquaredDisplacement{T}(guess, prior, fixed)

logprior(msd::MeanSquaredDisplacement{T,P}) where {T,P<:InverseGamma{T}} =
    -(shape(msd.prior) + 1) * log(msd.value) - scale(msd.prior) / msd.value

logprior_norm(msd::MeanSquaredDisplacement{T,P}) where {T,P} = logpdf(msd.prior, msd.value)

function update!(
    msd::MeanSquaredDisplacement{T},
    displacement²::AbstractArray{T,3},
    𝑇::T,
) where {T}
    isfixed(msd) && return msd
    Δparams = (length(displacement²), sum(vec(displacement²))) ./ (2 * 𝑇)
    msd.value = _rand(msd.prior, Δparams)
    return msd
end