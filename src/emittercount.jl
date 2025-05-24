"""
    EmitterCount{T<:AbstractFloat, V<:AbstractVector{T}} <: RandomVariable{T}

A mutable struct that represents the random variable of the number of tracks (the number of emitting particles). The length of `V` should match the weak limit (the total number of particle candidates).
"""
mutable struct EmitterCount{T<:AbstractFloat,V<:AbstractVector{T}} <: RandomVariable{T}
    value::Int
    prior::T
    fixed::Bool
    likelihood::V
    posterior::V
end

EmitterCount{T}(
    guess::Integer,
    limit::Integer,
    logonprob::Real,
    fixed::Bool = false,
) where {T<:AbstractFloat} = EmitterCount(
    guess,
    convert(T, logonprob),
    fixed,
    Vector{T}(undef, limit + 1),
    Vector{T}(undef, limit + 1),
)

function Base.getproperty(n::EmitterCount, s::Symbol)
    if s == :limit
        return length(getfield(n, :log𝒫)) - 1
    elseif s == :logprior
        return n.prior
    elseif s == :loglikelihood
        return n.likelihood
    elseif s == :logposterior
        return n.posterior
    else
        return getfield(n, s)
    end
end

Base.any(n::EmitterCount) = n.value > 0

logprior(n::EmitterCount) = n.logprior * n.value

function set_logposterior!(n::EmitterCount{T}, 𝑇::T) where {T}
    n.logposterior .= (0:length(n.loglikelihood)-1) .* n.logprior .+ n.loglikelihood ./ 𝑇
    return n
end

function set_loglikelihood!(
    count::EmitterCount{T},
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
        count.loglikelihood[m+1] = get_loglikelihood!(llarray, detector)
    end
    return count
end

function update!(
    count::EmitterCount{T},
    trackᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    llarray::LogLikelihoodArray{T},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    set_loglikelihood!(count, trackᵥ, brightnessᵥ, llarray, detector, psf)
    set_logposterior!(count, 𝑇)
    count.value = randc(count.logposterior) - 1
    return count
end