function NTracks{T}(
    value::Integer,
    limit::Integer,
    logonprob::Real,
) where {T<:AbstractFloat}
    logprior = collect((0:limit) .* convert(T, logonprob))
    return NTracks{T,typeof(logprior)}(
        value,
        logprior,
        similar(logprior),
        similar(logprior),
    )
end

function Base.getproperty(n::NTracks, s::Symbol)
    if s == :limit
        return length(getfield(n, :log𝒫)) - 1
    else
        return getfield(n, s)
    end
end

Base.any(ntracks::NTracks) = ntracks.value > 0

logprior(ntracks::NTracks) = ntracks.logprior[ntracks.value+1]

function set_logposterior!(ntracks::NTracks{T}, 𝑇::T) where {T}
    @. ntracks.log𝒫 = ntracks.logprior + ntracks.logℒ / 𝑇
    return ntracks
end

function set_loglikelihood!(
    ntracks::NTracks{T},
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
    psf::PointSpreadFunction{T},
) where {T}
    reset!(llarray, detector, 1)
    @inbounds for m = 1:size(tracksᵥ, 3)
        @views addincident!(
            llarray.means[1],
            tracksᵥ[:, :, m:m],
            brightnessᵥ,
            detector.pxbounds,
            psf,
        )
        ntracks.logℒ[m+1] = get_loglikelihood!(llarray, detector)
    end
    return ntracks
end

function update!(
    ntracks::NTracks{T},
    trackᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    llarray::LogLikelihoodArray{T},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    set_loglikelihood!(ntracks, trackᵥ, brightnessᵥ, llarray, detector, psf)
    set_logposterior!(ntracks, 𝑇)
    ntracks.value = randc(ntracks.log𝒫) - 1
    return ntracks
end