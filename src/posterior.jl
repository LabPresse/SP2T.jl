_logπ(ℕ::DNormal{T}, x::AbstractArray{T}) where {T} = sum(vec(@. -(x - ℕ.μ) / (2 * ℕ.σ^2)))

function logprior(tracks::Tracks{T}, nemittersᵥ::Integer, msdᵥ::T) where {T}
    xᵒⁿ = view(tracks.value, :, :, 1:nemittersᵥ)
    Δxᵒⁿ² = view(tracks.displacement₁², :, :, 1:nemittersᵥ)
    diff²!(Δxᵒⁿ², xᵒⁿ)
    return -(log(msdᵥ) * length(Δxᵒⁿ²) + sum(vec(Δxᵒⁿ²)) / msdᵥ) / 2 -
           _logπ(tracks.prior, view(xᵒⁿ, 1, :, :))
end

logprior(msd::MeanSquaredDisplacement{T,P}) where {T,P<:InverseGamma{T}} =
    -(shape(msd.prior) + 1) * log(msd.value) - scale(msd.prior) / msd.value

logprior_norm(msd::MeanSquaredDisplacement{T,P}) where {T,P} = logpdf(msd.prior, msd.value)

logprior(nemitters::NTracks) = nemitters.logprior[nemitters.value+1]

logprior(brightness::Brightness{T,P}) where {T,P<:Gamma{T}} =
    (shape(brightness.prior) - 1) * log(brightness.value) -
    brightness.value / scale(brightness.prior)

logprior_norm(brightness::Brightness{T,P}) where {T,P} =
    logpdf(brightness.prior, brightness.value)

function log𝒫logℒ(
    tracks::Tracks{T},
    nemitters::NTracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    measurements::AbstractArray{<:Union{T,Integer},3},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
) where {T}
    pxcounts!(detector, view(tracks.value, :, :, 1:nemitters.value), brightness.value, psf)
    logℒ1 = logℒ!(detector, measurements)
    log𝒫1 =
        logℒ1 +
        logprior(tracks, nemitters.value, msd.value) +
        logprior(msd) +
        logprior(nemitters)
    return log𝒫1, logℒ1
end