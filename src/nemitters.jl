function NEmitters{T}(
    value::Integer,
    limit::Integer,
    logonprob::Real,
) where {T<:AbstractFloat}
    logprior = convert(T, logonprob)
    loglikelihood = Vector{T}(undef, limit + 1)
    return NEmitters{T,typeof(loglikelihood)}(
        value,
        logprior,
        loglikelihood,
        similar(loglikelihood),
    )
end

function Base.getproperty(n::NEmitters, s::Symbol)
    if s == :limit
        return length(getfield(n, :log𝒫)) - 1
    else
        return getfield(n, s)
    end
end

Base.any(n::NEmitters) = n.value > 0

logprior(n::NEmitters) = n.logprior * n.value

function set_logposterior!(n::NEmitters{T}, 𝑇::T) where {T}
    n.logposterior .= (0:length(n.loglikelihood)-1) .* n.logprior .+ n.loglikelihood ./ 𝑇
    return n
end

function set_loglikelihood!(
    nemitters::NEmitters{T},
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
    psf::PointSpreadFunction{T},
) where {T}
    reset!(llarray, detector, 1)
    nframes, ~, nparticles = size(tracksᵥ)
    psfcomponents = (
        similar(tracksᵥ, size(detector, 1), 1, nframes),
        similar(tracksᵥ, size(detector, 2), 1, nframes),
    )
    relativepos = (
        similar(tracksᵥ, size(detector, 1) + 1, 1, nframes),
        similar(tracksᵥ, size(detector, 2) + 1, 1, nframes),
    )
    @inbounds for m = 1:nparticles
        addincident!(
            llarray.means[1],
            psfcomponents,
            relativepos,
            view(tracksᵥ, :, :, m:m),
            brightnessᵥ,
            detector.pxbounds,
            psf,
        )
        nemitters.loglikelihood[m+1] = get_loglikelihood!(llarray, detector)
    end
    return nemitters
end

function update!(
    nemitters::NEmitters{T},
    trackᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    llarray::LogLikelihoodArray{T},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    set_loglikelihood!(nemitters, trackᵥ, brightnessᵥ, llarray, detector, psf)
    set_logposterior!(nemitters, 𝑇)
    nemitters.value = randc(nemitters.logposterior) - 1
    return nemitters
end