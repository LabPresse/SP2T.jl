"""
    MeanSquaredDisplacement{T<:AbstractFloat, P<:InverseGamma{T}} <: RandomVariable{T}

A mutable struct representing the mean squared displacement random variable. `value::T` is the value, `prior::P` is an inverse gamma distrtibution, `fixed::Bool` determines whether its `value` gets updated.
"""
mutable struct MeanSquaredDisplacement{T<:AbstractFloat,P<:InverseGamma{T}} <:
               RandomVariable{T}
    value::T
    prior::P
    fixed::Bool
end

"""
    MeanSquaredDisplacement{T}(; guess::Real, priorparams::Tuple{<:Real,<:Real}, fixed::Bool)

The constructor for `MeanSquaredDisplacement`. By default, `priorparams = (2, 1e-5)' and `fixed = false`.
"""
MeanSquaredDisplacement{T}(;
    guess::Real,
    priorparams::Tuple{<:Real,<:Real} = (2, 1e-5),
    fixed::Bool = false,
) where {T<:AbstractFloat} = MeanSquaredDisplacement(
    convert(T, guess),
    InverseGamma(convert.(T, priorparams)...),
    fixed,
)

"""
    logprior(msd::MeanSquaredDisplacement; normalization::Bool)

Calculate the log density for `msd`'s prior given its current value. `normalization = false` by default. 
"""
logprior(msd::MeanSquaredDisplacement; normalization::Bool = false) =
    normalization ? logpdf(msd.prior, msd.value) :
    -(shape(msd.prior) + 1) * log(msd.value) - scale(msd.prior) / msd.value

"""
    update!(msd::MeanSquaredDisplacement{T}, displacement²::AbstractArray{T,3}, 𝑇::T)

Directly sample `msd`'s new value from its posterior given the `displacement²` array.
Only the detector likelihood is tempered, so this conditional does not depend on `𝑇`.
"""
function update!(
    msd::MeanSquaredDisplacement{T},
    displacement²::AbstractArray{T,3},
    𝑇::T,
) where {T}
    isfixed(msd) && return msd
    Δparams = (length(displacement²), sum(vec(displacement²))) ./ (2 * one(T))
    msd.value = _rand(msd.prior, Δparams)
    return msd
end
