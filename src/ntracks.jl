function NTracks{T}(
    value::Integer,
    limit::Integer,
    logonprob::Real,
) where {T<:AbstractFloat}
    logprior = convert(T, logonprob)
    loglikelihood = Vector{T}(undef, limit + 1)
    return NTracks{T,typeof(loglikelihood)}(
        value,
        logprior,
        loglikelihood,
        similar(loglikelihood),
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

logprior(ntracks::NTracks) = ntracks.logprior * ntracks.value

function set_logposterior!(ntracks::NTracks{T}, 𝑇::T) where {T}
    ntracks.logposterior .=
        (0:length(ntracks.loglikelihood)-1) .* ntracks.logprior .+
        ntracks.loglikelihood ./ 𝑇
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
        ntracks.loglikelihood[m+1] = get_loglikelihood!(llarray, detector)
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
    ntracks.value = randc(ntracks.logposterior) - 1
    return ntracks
end