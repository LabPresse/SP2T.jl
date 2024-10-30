mutable struct NEmitters{T,V}
    value::Int
    logprior::V
    logℒ::V
    log𝒫::V
end

function NEmitters{T}(;
    value::Integer,
    limit::Integer,
    logonprob::Real,
) where {T<:AbstractFloat}
    logprior = collect((0:limit) .* convert(T, logonprob))
    return NEmitters{T,typeof(logprior)}(
        value,
        logprior,
        similar(logprior),
        similar(logprior),
    )
end

function Base.getproperty(nemitters::NEmitters, s::Symbol)
    if s == :limit
        return length(getfield(nemitters, :log𝒫)) - 1
    else
        return getfield(nemitters, s)
    end
end

Base.any(nemitters::NEmitters) = nemitters.value > 0

function setlog𝒫!(nemitters::NEmitters{T}, 𝑇::T) where {T}
    @. nemitters.log𝒫 = nemitters.logprior + nemitters.logℒ / 𝑇
    return nemitters
end

function setlogℒ!(
    nemitters::NEmitters,
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

function sample!(nemitters::NEmitters)
    nemitters.value = randc(nemitters.log𝒫) - 1
    return nemitters
end