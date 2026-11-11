"""
    MeanSquaredDisplacement{T<:AbstractFloat, P<:InverseGamma{T}} <: RandomVariable{T}

A mutable struct representing the mean squared displacement random variable. `value::T` is the value, `prior::P` is an inverse gamma distribution, `fixed::Bool` determines whether its `value` gets updated. `proposalscale::T` is the standard deviation of `log(msd′/msd)` in the rescaling move, and `counter::Vector{Int}` records its acceptances and proposals.
"""
mutable struct MeanSquaredDisplacement{T<:AbstractFloat,P<:InverseGamma{T}} <:
               RandomVariable{T}
    value::T
    prior::P
    fixed::Bool
    proposalscale::T
    counter::Vector{Int}
end

"""
    MeanSquaredDisplacement{T}(; guess::Real, priorparams::Tuple{<:Real,<:Real}, fixed::Bool, proposalscale::Real)

The constructor for `MeanSquaredDisplacement`. By default, `priorparams = (2, 1e-5)`, `fixed = false` and `proposalscale = 0.05`.
"""
MeanSquaredDisplacement{T}(;
    guess::Real,
    priorparams::Tuple{<:Real,<:Real} = (2, 1e-5),
    fixed::Bool = false,
    proposalscale::Real = 0.05,
) where {T<:AbstractFloat} = MeanSquaredDisplacement(
    convert(T, guess),
    InverseGamma(convert.(T, priorparams)...),
    fixed,
    convert(T, proposalscale),
    zeros(Int, 2),
)

"""
    logprior(msd::MeanSquaredDisplacement; normalization::Bool)

Calculate the log density for `msd`'s prior given its current value. `normalization = false` by default. 
"""
logprior(msd::MeanSquaredDisplacement; normalization::Bool = false) =
    normalization ? logpdf(msd.prior, msd.value) :
    -(shape(msd.prior) + 1) * log(msd.value) - scale(msd.prior) / msd.value

"""
    update!(rng::AbstractRNG, msd::MeanSquaredDisplacement{T}, displacement²::AbstractArray{T,3}, 𝑇::T)

Directly sample `msd`'s new value from its posterior given `displacement²`.
Only the detector likelihood is tempered, so the motion-prior sufficient
statistics and this conditional do not depend on `𝑇`.
"""
function update!(
    rng::AbstractRNG,
    msd::MeanSquaredDisplacement{T},
    displacement²::AbstractArray{T,3},
    𝑇::T,
) where {T}
    isfixed(msd) && return msd
    Δparams = (T(length(displacement²)) / 2, sum(vec(displacement²)) / 2)
    msd.value = _rand(rng, msd.prior, Δparams)
    return msd
end
