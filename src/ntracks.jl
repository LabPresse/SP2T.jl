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

function Base.getproperty(nemitters::NTracks, s::Symbol)
    if s == :limit
        return length(getfield(nemitters, :log𝒫)) - 1
    else
        return getfield(nemitters, s)
    end
end

Base.any(nemitters::NTracks) = nemitters.value > 0

logprior(nemitters::NTracks) = nemitters.logprior[nemitters.value+1]

function setlog𝒫!(nemitters::NTracks{T}, 𝑇::T) where {T}
    @. nemitters.log𝒫 = nemitters.logprior + nemitters.logℒ / 𝑇
    return nemitters
end

function setlogℒ!(
    nemitters::NTracks,
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    measurements::AbstractArray{<:Union{T,UInt16}},
    detector::PixelDetector{T},
    psf::PointSpreadFunction{T},
) where {T}
    initintensity!(detector)
    @inbounds for m = 1:size(tracksᵥ, 3)
        add_pxcounts!(
            detector.intensity₁,
            view(tracksᵥ, :, :, m:m),
            brightnessᵥ,
            detector.pxboundsx,
            detector.pxboundsy,
            psf,
        )


        nemitters.logℒ[m+1] = logℒ!(detector, measurements)
    end
    return nemitters
end

randc(logp::AbstractArray) = argmax(logp .- log.(randexp!(similar(logp))))

function sample!(nemitters::NTracks)
    nemitters.value = randc(nemitters.log𝒫) - 1
    return nemitters
end

function update!(
    nemitters::NTracks{T},
    trackᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    measurements::AbstractArray{<:Union{T,Integer}},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    setlogℒ!(nemitters, trackᵥ, brightnessᵥ, measurements, detector, psf)
    setlog𝒫!(nemitters, 𝑇)
    sample!(nemitters)
    return nemitters
end
