struct Sample{T<:AbstractFloat,A<:AbstractArray{T,3}}
    tracks::A
    diffusivity::T
    brightness::T
    iteration::Int # iteration
    𝑇::T # temperature
    log𝒫::T # log posterior
    logℒ::T # log likelihood
end

Sample(
    tracks::TrackChunk{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    iter::Integer = 0,
    𝑇::Real = 1,
    log𝒫::Real = NaN,
    logℒ::Real = NaN,
) where {T} = Sample(
    collect(tracks.value),
    msd.value,
    brightness.value,
    iter,
    convert(T, 𝑇),
    convert(T, log𝒫),
    convert(T, logℒ),
)

function Base.getproperty(sample::Sample, s::Symbol)
    if s === :nemitters
        return size(sample.tracks, 3)
    else
        return getfield(sample, s)
    end
end

mutable struct Chain{T<:AbstractFloat,VofS<:Vector{<:Sample{T}},A<:Annealing{T}}
    samples::VofS
    sizelimit::Int
    annealing::A
end

Base.propertynames(::Chain) = (
    fieldnames(Chain)...,
    :msds,
    :brightnesses,
    :emittercounts,
    :lasttracks,
    :iterations,
    :loglikelihoods,
    :logposteriors,
    :savestride,
)

function Base.getproperty(c::Chain, s::Symbol)
    if s === :msds
        return [sample.diffusivity for sample in getfield(c, :samples)]
    elseif s === :brightnesses
        return [sample.brightness for sample in getfield(c, :samples)]
    elseif s === :emittercounts
        return [size(sample.tracks, 3) for sample in getfield(c, :samples)]
    elseif s === :lasttracks
        return c.samples[end].tracks
    elseif s === :iterations
        return [sample.iteration for sample in getfield(c, :samples)]
    elseif s === :loglikelihoods
        return [sample.logℒ for sample in getfield(c, :samples)]
    elseif s === :logposteriors
        return [sample.log𝒫 for sample in getfield(c, :samples)]
    elseif s === :savestride
        return savestride(c)
    else
        return getfield(c, s)
    end
end

Base.length(chain::Chain) = length(chain.samples)

isfull(chain::Chain) = length(chain) == chain.sizelimit

savestride(chain::Chain) =
    length(chain) == 1 ? 1 :
    (chain.samples[2].iteration - chain.samples[1].iteration) * 2^isfull(chain)

tosave(chain::Chain, iter::Real) = iter % savestride(chain) == 0

function findmap(chain::Chain; burn_in::Real = 0)
    ~, i = findmax(chain.logposteriors[burn_in+1:end])
    return chain.samples[i+burn_in], i + burn_in
end

function findmle(chain::Chain; burn_in::Real = 0)
    ~, i = findmax(chain.loglikelihoods[burn_in+1:end])
    return chain.samples[i+burn_in], i + burn_in
end

jumpdistances(tracks::AbstractArray{<:Real}) = sqrt.(sum(diff(tracks, dims=1) .^ 2, dims=2))
function jumpdistances(chain::Chain{T}; burn_in::Integer=0) where {T}
    d = Vector{T}()
    for i in burn_in+1:length(chain)
        append!(d, vec(jumpdistances(chain.samples[i].tracks)))
    end
    d
end

function shrink!(chain::Chain)
    deleteat!(chain.samples, 2:2:lastindex(chain.samples))
    return chain
end

temperature(chain::Chain, i::Real) = temperature(chain.annealing, i)

function extend!(
    chain::Chain{T},
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    llarray::LogLikelihoodArray{T},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    iter::Integer,
    𝑇::T,
) where {T}
    loglikelihood = get_loglikelihood!(llarray, tracks, brightness, detector, psf)
    logposterior = get_logposterior(loglikelihood, tracks, msd)
    push!(
        chain.samples,
        Sample(tracks.onchunk, msd, brightness, iter, 𝑇, logposterior, loglikelihood),
    )
    return chain
end

function get_loglikelihood!(
    llarray::LogLikelihoodArray{T},
    tracks::Tracks{T},
    brightness::Brightness{T},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
) where {T}
    seteffvalue!(tracks.onchunk)
    set_poisson_mean!(llarray, detector, tracks.onchunk.effvalue, brightness.value, psf)
    return get_loglikelihood!(llarray, detector)
end

get_logposterior(
    loglikelihood::T,
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
) where {T} =
    loglikelihood +
    logprior(tracks.onchunk, msd.value) +
    logprior(msd) +
    logprior(tracks.nemitters)

function parametricMCMC!(
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    llarray::LogLikelihoodArray{T},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    update_onchunk!(tracks, msd.value, brightness.value, llarray, detector, psf, 𝑇)
    update!(tracks.onchunk, tracks.proposals, msd.value) # swap identities partway through
    seteffvalue!(tracks.onchunk)
    update!(brightness, tracks.onchunk.effvalue, llarray, detector, psf, 𝑇)
    setdisplacement²!(tracks.onchunk)
    update!(msd, tracks.onchunk.displacement², 𝑇)
    return tracks, msd
end

function nonparametricMCMC!(
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    llarray::LogLikelihoodArray{T},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    simulate!(tracks.offchunk, msd.value)
    if any(tracks)
        update_onchunk!(tracks, msd.value, brightness.value, llarray, detector, psf, 𝑇)
        update!(tracks.onchunk, tracks.proposals, msd.value) # swap identities partway through
        onshuffle!(tracks)
    end

    seteffvalue!(tracks)
    update!(
        tracks.nemitters,
        tracks.effvalue[1],
        brightness.value,
        llarray,
        detector,
        psf,
        𝑇,
    )
    reassign!(tracks)

    update!(brightness, tracks.onchunk.effvalue, llarray, detector, psf, 𝑇)

    setdisplacement²!(tracks)
    update!(msd, tracks.displacement²[1], 𝑇)
    return tracks, msd
end

"""
    runMCMC!(; kwargs...)

Continue running a MCMC sampling with the specified keyword arguments. Requires the following keyword arguments:
- `chain`: A `Chain` object representing the MCMC chain.
- `tracks`: A `Tracks` object representing the tracks of all particle candidates.
- `msd`: A `MeanSquaredDisplacement` object representing the mean squared displacement of the particles.
- `brightness`: A `Brightness` object representing the brightness of the particles.
- `detector`: A `Detector` object representing the detector used to capture the particles.
- `psf`: A `PointSpreadFunction` object representing the point spread function of the system.
- `niters`: The number of additional iterations for the MCMC simulation.
- `parametric`: A boolean indicating whether to use parametric or non-parametric MCMC. A fixed emitter count (`tracks.nemitters.fixed`) also selects parametric MCMC.
The chain's size limit and annealing schedule are those stored in `chain`.
"""
function runMCMC!(;
    chain::Chain{T},
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    niters::Integer,
    parametric::Bool,
) where {T}
    prev_niters = chain.samples[end].iteration
    llarray = LogLikelihoodArray{T}(detector.readouts)
    reset!(llarray, detector, 1)
    tracks.nemitters.loglikelihood[1] = get_loglikelihood!(llarray, detector)
    nextsample! =
        parametric || isfixed(tracks.nemitters) ? parametricMCMC! : nonparametricMCMC!
    @showprogress 1 "Computing..." for iter in prev_niters .+ (1:niters)
        𝑇 = temperature(chain, iter)
        nextsample!(tracks, msd, brightness, llarray, detector, psf, 𝑇)
        if tosave(chain, iter)
            isfull(chain) && shrink!(chain)
            extend!(chain, tracks, msd, brightness, llarray, detector, psf, iter, 𝑇)
        end
    end
end

"""
    runMCMC!(chain, tracks, msd, brightness, detector, psf, niters, parametric)

Positional form of `runMCMC!(; kwargs...)`, with the arguments in the same order.
"""
runMCMC!(
    chain::Chain{T},
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    niters::Integer,
    parametric::Bool,
) where {T} = runMCMC!(
    chain = chain,
    tracks = tracks,
    msd = msd,
    brightness = brightness,
    detector = detector,
    psf = psf,
    niters = niters,
    parametric = parametric,
)

"""
    runMCMC(; kwargs...)

Run a MCMC sampling with the specified keyword arguments. Requires the following keyword arguments:
- `tracks`: A `Tracks` object representing the tracks of all particle candidates.
- `msd`: A `MeanSquaredDisplacement` object representing the mean squared displacement of the particles.
- `brightness`: A `Brightness` object representing the brightness of the particles.
- `detector`: A `Detector` object representing the detector used to capture the particles.
- `psf`: A `PointSpreadFunction` object representing the point spread function of the system.
- `niters`: The number of iterations for the MCMC simulation (default: 1000).
- `sizelimit`: The maximum size of the chain (default: 1000).
- `annealing`: An `Annealing` object representing the annealing schedule (default: `ConstantAnnealing{T}(1)`).
- `parametric`: A boolean indicating whether to use parametric or non-parametric MCMC (default: `false`). A fixed emitter count (`tracks.nemitters.fixed`) also selects parametric MCMC.
"""
function runMCMC(;
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    niters::Integer = 1000,
    sizelimit::Integer = 1000,
    annealing::Union{Annealing{T},Nothing} = nothing,
    parametric::Bool = false,
) where {T}
    isnothing(annealing) && (annealing = ConstantAnnealing{T}(1))
    chain = Chain([Sample(tracks.onchunk, msd, brightness)], sizelimit, annealing)
    runMCMC!(
        chain = chain,
        tracks = tracks,
        msd = msd,
        brightness = brightness,
        detector = detector,
        psf = psf,
        niters = niters,
        parametric = parametric,
    )
    return chain
end
