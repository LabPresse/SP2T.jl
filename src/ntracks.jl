mutable struct NTracks{T,V}
    value::Int
    logprior::V
    logℒ::V
    log𝒫::V
end

function NTracks{T}(;
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

function setlog𝒫!(ntracks::NTracks{T}, 𝑇::T) where {T}
    @. ntracks.log𝒫 = ntracks.logprior + ntracks.logℒ / 𝑇
    return ntracks
end

function setlogℒ!(
    ntracks::NTracks,
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    measurements::AbstractArray{<:Union{T,UInt16}},
    detector::PixelDetector{T},
    psf::PointSpreadFunction{T},
) where {T}
    reset!(detector, 1)
    @inbounds for m = 1:size(tracksᵥ, 3)
        addincident!(
            detector.intensity,
            view(tracksᵥ, :, :, m:m),
            brightnessᵥ,
            detector.pxboundsx,
            detector.pxboundsy,
            psf,
        )
        ntracks.logℒ[m+1] = logℒ!(detector, measurements)
    end
    return ntracks
end

function update!(
    ntracks::NTracks{T},
    trackᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    measurements::AbstractArray{<:Union{T,Integer}},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    setlogℒ!(ntracks, trackᵥ, brightnessᵥ, measurements, detector, psf)
    setlog𝒫!(ntracks, 𝑇)
    ntracks.value = randc(ntracks.log𝒫) - 1
    return ntracks
end