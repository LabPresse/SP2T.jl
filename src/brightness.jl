"""
    Brightness{T<:AbstractFloat, P} <: RandomVariable{T}

A mutable struct representing the brightness random variable. `value::T` is the value, `prior::P` should be a probability distribution, `fixed::Bool` determines whether its `value` gets updated. `proposal::Beta{T}` is a Beta proposal distribution for the mltiplicative random walk (see Pressé, Data Modeling for the Sciences, 2023, p195). `counter::Vector{Int}` is a vector recording the number of proposals and the number of acceptances.
"""
mutable struct Brightness{T<:AbstractFloat,P} <: RandomVariable{T}
    value::T
    prior::P
    fixed::Bool
    proposal::Beta{T}
    counter::Vector{Int}
end

Brightness{T}(;
    guess::Real,
    prior::ContinuousUnivariateDistribution,
    proposalparam::Real,
    fixed::Bool = false,
) where {T<:AbstractFloat} = Brightness(
    convert(T, guess),
    unionalltypeof(prior)(convert.(T, params(prior))...),
    fixed,
    Beta(convert(T, proposalparam), oneunit(T)),
    zeros(Int, 2),
)

logprior(brightness::Brightness{T,P}) where {T,P<:Gamma{T}} =
    (shape(brightness.prior) - 1) * log(brightness.value) -
    brightness.value / scale(brightness.prior)

logprior_norm(brightness::Brightness{T,P}) where {T,P} =
    logpdf(brightness.prior, brightness.value)

function propose(brightness::Brightness{T}) where {T<:AbstractFloat}
    ϵ = rand(brightness.proposal)
    return rand(Bool) ? brightness.value * ϵ : brightness.value / ϵ
end

function update!(
    brightness::Brightness{T},
    tracksᵥ::AbstractArray{T,3},
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    isfixed(brightness) && return brightness
    proposed_value = propose(brightness)
    set_poisson_means!(llarray, detector, tracksᵥ, brightness.value, proposed_value, psf)
    get_Δloglikelihood!(llarray, detector)
    logacceptance =
        get_Δloglikelihood!(llarray, detector) / 𝑇 +
        get_Δlogprior(proposed_value, brightness.value, brightness.prior)
    accepted = logacceptance > log(rand())
    brightness.counter .+= (accepted, 1)
    accepted && (brightness.value = proposed_value)
    return brightness
end
