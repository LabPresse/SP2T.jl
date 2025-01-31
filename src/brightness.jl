Brightness{T}(;
    guess::Real,
    prior::ContinuousUnivariateDistribution,
    proposalparam::Real,
    fixed::Bool = false,
) where {T<:AbstractFloat} = Brightness(
    convert(T, guess),
    unionalltypeof(prior)(convert.(T, params(prior))...),
    Beta(convert(T, proposalparam), oneunit(T)),
    fixed,
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
