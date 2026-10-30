"""
    Brightness{T<:AbstractFloat, P} <: RandomVariable{T}

A mutable struct representing the brightness random variable. `value::T` is the value, `prior::P` is a gamma distribution, `fixed::Bool` determines whether its `value` gets updated. `proposal::Beta{T}` is a beta proposal distribution for the multiplicative random walk (see Pressé, Data Modeling for the Sciences, 2023, p195). `counter::Vector{Int}` records the number of acceptances and the number of proposals.
"""
mutable struct Brightness{T<:AbstractFloat,P} <: RandomVariable{T}
    value::T
    prior::P
    fixed::Bool
    proposal::Beta{T}
    counter::Vector{Int}
end

"""
    Brightness{T}(; guess::Real, priorparams::Tuple{<:Real,<:Real}, fixed::Bool, proposalparams::Tuple{<:Real,<:Real})

The constructor for `Brightness`. By default, `priorparams = (1, 1)`, `fixed = false`, and `proposalparams = (1, 10)`.
"""
Brightness{T}(;
    guess::Real,
    priorparams::Tuple{<:Real,<:Real} = (1, 1),
    fixed::Bool = false,
    proposalparams::Tuple{<:Real,<:Real} = (1, 10),
) where {T<:AbstractFloat} = Brightness(
    convert(T, guess),
    Gamma(convert.(T, priorparams)...),
    fixed,
    Beta(convert.(T, proposalparams)...),
    zeros(Int, 2),
)

"""
    logprior(b::Brightness{T,P}; normalization::Bool)

Calculate the log density for `b`'s prior given its current value. `normalization = false` by default. 
"""
logprior(b::Brightness{T,P}; normalization::Bool = false) where {T,P<:Gamma{T}} =
    normalization ? logpdf(b.prior, b.value) :
    (shape(b.prior) - 1) * log(b.value) - b.value / scale(b.prior)

"""
    Δlogprior(b::Brightness{T}, xᵖ::T)

Calculate `logprior(b, xᵖ) - logprior(b, b.value)`.
"""
Δlogprior(b::Brightness{T}, xᵖ::T) where {T<:AbstractFloat} = Δlogpdf(b.prior, xᵖ, b.value)

function update!(
    rng::AbstractRNG,
    brightness::Brightness{T},
    tracksᵥ::AbstractArray{T,3},
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    isfixed(brightness) && return brightness
    valueᵖ, logacceptance = mrw_propose(rng, brightness.proposal, brightness.value)
    set_poisson_means!(llarray, detector, tracksᵥ, brightness.value, valueᵖ, psf)
    logacceptance +=
        get_Δloglikelihood!(llarray, detector) / 𝑇 + Δlogprior(brightness, valueᵖ) -
        log(rand(rng))
    accepted = logacceptance > 0
    brightness.counter .+= (accepted, 1)
    accepted && (brightness.value = valueᵖ)
    return brightness
end
