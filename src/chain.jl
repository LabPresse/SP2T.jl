struct Sample{T<:AbstractFloat,A<:AbstractArray{T,3}}
    tracks::A
    diffusivity::T
    brightness::T
    iteration::Int # iteration
    𝑇::T # temperature
    log𝒫::T # standardized-coordinate score; off candidates use offlogprior
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
    iteration::Int # completed transitions, including states not retained after thinning
end

# Saved samples do not identify an unsaved live state. Supply iteration explicitly
# when reconstructing a chain together with a separately preserved sampler state.
function Chain(samples::Vector{<:Sample}, sizelimit::Integer, annealing::Annealing;
    iteration::Integer = isempty(samples) ? 0 : samples[end].iteration,
)
    sizelimit >= 2 || throw(ArgumentError("sizelimit must be at least two"))
    iteration >= 0 || throw(ArgumentError("iteration must be nonnegative"))
    return Chain(samples, Int(sizelimit), annealing, Int(iteration))
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

"""
    findmap(chain::Chain; burn_in::Integer = 0)

Return the sample with the largest finite saved score after the first `burn_in`
stored samples, and its index. The score uses first positions and standardized
increments. With off candidates it includes an expected or independently drawn
off-prior score, not their marginal density or the density of their stored
coordinates. Consequently this is a score-maximizing point estimate, not general
full-state MAP or a summary of the posterior distribution.
"""
function findmap(chain::Chain; burn_in::Integer = 0)
    return _findbest(chain, :log𝒫, burn_in)
end

function findmle(chain::Chain; burn_in::Integer = 0)
    return _findbest(chain, :logℒ, burn_in)
end

function _findbest(chain::Chain, field::Symbol, burn_in::Integer)
    0 <= burn_in < length(chain) || throw(ArgumentError(
        "burn_in must be nonnegative and smaller than the number of stored samples",
    ))
    best_index, best_score = 0, -Inf
    for i in burn_in+1:length(chain)
        score = getproperty(chain.samples[i], field)
        if isfinite(score) && (best_index == 0 || score > best_score)
            best_index, best_score = i, score
        end
    end
    best_index > 0 || throw(ArgumentError("no finite $field scores remain after burn-in"))
    return chain.samples[best_index], best_index
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
    llarray::LogLikelihoodAux{T},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    iter::Integer,
    𝑇::T,
    logposterior_rng::Union{Nothing,AbstractRNG} = nothing,
) where {T}
    loglikelihood = get_loglikelihood!(llarray, tracks, brightness, detector, psf)
    logposterior = get_logposterior(loglikelihood, tracks, msd, brightness, logposterior_rng)
    push!(
        chain.samples,
        Sample(tracks.onchunk, msd, brightness, iter, 𝑇, logposterior, loglikelihood),
    )
    return chain
end

function get_loglikelihood!(
    llarray::LogLikelihoodAux{T},
    tracks::Tracks{T},
    brightness::Brightness{T},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
) where {T}
    seteffvalue!(tracks.onchunk)
    set_poisson_mean!(llarray, detector, tracks.onchunk.effvalue, brightness.value, psf)
    return get_loglikelihood!(llarray, detector)
end

# Reduced standardized-coordinate score, retained for callers that do not supply
# brightness. Off candidates contribute an expected or independently drawn
# log-prior proxy, not the marginalized density or their stored-state log density.
get_logposterior(
    loglikelihood::T,
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
    logposterior_rng::Union{Nothing,AbstractRNG} = nothing,
) where {T} =
    loglikelihood +
    logprior(tracks.onchunk, msd.value) +
    offlogprior(logposterior_rng, tracks, msd.value) +
    logjacobian(tracks, msd.value) +
    logprior(msd) +
    logprior(tracks.nemitters)

# Saved scores include the brightness prior as well. Keep the standardized-
# increment Jacobian and the declared off-candidate proxy from the current model.
get_logposterior(
    loglikelihood::T,
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    logposterior_rng::Union{Nothing,AbstractRNG} = nothing,
) where {T} = get_logposterior(loglikelihood, tracks, msd, logposterior_rng) +
    logprior(brightness; normalization = true)

# GPU kernel launches can consume the host task RNG during deterministic scratch
# initialization. Rebuilding scratch at a continuation boundary must not consume
# proposal draws. This guard does not restore device RNGs or cover sampling/scoring.
function _continuation_likelihood_array(tracks::Tracks{T}, detector::Detector{T}) where {T}
    host_rng = Random.default_rng()
    saved_rng = copy(host_rng)
    try
        llarray = loglikelihoodaux(detector)
        reset!(llarray, detector, 1)
        tracks.nemitters.loglikelihood[1] = get_loglikelihood!(llarray, detector)
        return llarray
    finally
        copy!(host_rng, saved_rng)
    end
end

function parametricMCMC!(
    rng::AbstractRNG,
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    llarray::LogLikelihoodAux{T},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    𝑇::T;
    msdrescale::Bool = false,
    countbrightness::Bool = false, # ignored: the emitter count is fixed
) where {T}
    update_onchunk!(rng, tracks, msd.value, brightness.value, llarray, detector, psf, 𝑇)
    update!(rng, tracks.onchunk, tracks.proposals, msd.value) # swap identities partway through
    seteffvalue!(tracks.onchunk)
    update!(rng, brightness, tracks.onchunk.effvalue, llarray, detector, psf, 𝑇)
    setdisplacement²!(tracks.onchunk)
    update!(rng, msd, tracks.onchunk.displacement², 𝑇)
    msdrescale && update!(rng, msd, tracks, brightness.value, llarray, detector, psf, 𝑇)
    return tracks, msd
end

function nonparametricMCMC!(
    rng::AbstractRNG,
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    llarray::LogLikelihoodAux{T},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    𝑇::T;
    msdrescale::Bool = false,
    countbrightness::Bool = false,
) where {T}
    simulate!(rng, tracks.offchunk, msd.value)
    if any(tracks)
        update_onchunk!(rng, tracks, msd.value, brightness.value, llarray, detector, psf, 𝑇)
        update!(rng, tracks.onchunk, tracks.proposals, msd.value) # swap identities partway through
        onshuffle!(rng, tracks)
    end

    seteffvalue!(tracks)
    update!(
        rng,
        tracks.nemitters,
        tracks.effvalue[1],
        brightness.value,
        llarray,
        detector,
        psf,
        𝑇,
    )
    # rescales the brightness; the count update above is the move at a fixed brightness
    countbrightness && update!(
        rng,
        tracks.nemitters,
        brightness,
        tracks.effvalue[1],
        llarray,
        detector,
        psf,
        𝑇,
    )
    reassign!(tracks)

    update!(rng, brightness, tracks.onchunk.effvalue, llarray, detector, psf, 𝑇)

    # the off tracks are integrated out: drawn from the MSD, they would pull it toward its
    # current value; they are redrawn from the new MSD before the next count update
    setdisplacement²!(tracks.onchunk)
    update!(rng, msd, tracks.onchunk.displacement², 𝑇)
    msdrescale && update!(rng, msd, tracks, brightness.value, llarray, detector, psf, 𝑇)
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
- `rng`: The random number generator (default: `Random.default_rng()`). On GPU, array draws use `CUDA.default_rng()` instead; seed it separately for reproducible GPU runs.
- `logposterior_rng`: Saved scores include the brightness prior and use standardized increments. Off candidates contribute their expected log prior (default: `nothing`) or an independent draw of that log prior using this generator. This is an off-prior score proxy, not a marginalized density or the density of the stored off tracks. Use a generator distinct from `rng` to leave sampling unchanged.
- `msdrescale`: Also run the rescaling MSD move after the MSD's Gibbs draw, which stretches the on tracks with the MSD (default: `false`; `docs/msdrescale.qmd`). It costs one likelihood evaluation per iteration.
- `countbrightness`: Also run the joint count–brightness move after the emitter-count update, which switches one emitter on or off and rescales the brightness by n/n′ (default: `false`; `docs/countbrightness.qmd`). Nonparametric sweep only.
Both moves leave the posterior unchanged; they change only how the chain moves.
The chain's size limit and annealing schedule are those stored in `chain`.
`chain.iteration` counts completed transitions independently of retained samples.
Free brightness must be finite and strictly positive: its multiplicative proposal
cannot leave zero. Fixed zero brightness remains permitted.
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
    rng::AbstractRNG = Random.default_rng(),
    logposterior_rng::Union{Nothing,AbstractRNG} = nothing,
    msdrescale::Bool = false,
    countbrightness::Bool = false,
) where {T}
    niters >= 0 || throw(ArgumentError("niters must be nonnegative"))
    isfixed(brightness) || (isfinite(brightness.value) && brightness.value > zero(T)) ||
        throw(ArgumentError(
            "free brightness must be finite and positive; its multiplicative proposal cannot leave zero",
        ))
    prev_niters = chain.iteration
    llarray = _continuation_likelihood_array(tracks, detector)
    if isempty(chain.samples)
        extend!(chain, tracks, msd, brightness, llarray, detector, psf,
            prev_niters, temperature(chain, prev_niters), logposterior_rng)
    end
    parametric = parametric || isfixed(tracks.nemitters)
    nextsample! = parametric ? parametricMCMC! : nonparametricMCMC!
    @showprogress dt = 1 desc = "Computing..." for iter in prev_niters .+ (1:niters)
        𝑇 = temperature(chain, iter)
        nextsample!(
            rng,
            tracks,
            msd,
            brightness,
            llarray,
            detector,
            psf,
            𝑇;
            msdrescale,
            countbrightness,
        )
        chain.iteration = iter
        if tosave(chain, iter)
            isfull(chain) && shrink!(chain)
            extend!(
                chain,
                tracks,
                msd,
                brightness,
                llarray,
                detector,
                psf,
                iter,
                𝑇,
                logposterior_rng,
            )
        end
    end
    return chain
end

"""
    runMCMC!(chain, tracks, msd, brightness, detector, psf, niters, parametric; rng = Random.default_rng(), logposterior_rng = nothing, msdrescale = false, countbrightness = false)

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
    parametric::Bool;
    rng::AbstractRNG = Random.default_rng(),
    logposterior_rng::Union{Nothing,AbstractRNG} = nothing,
    msdrescale::Bool = false,
    countbrightness::Bool = false,
) where {T} = runMCMC!(
    chain = chain,
    tracks = tracks,
    msd = msd,
    brightness = brightness,
    detector = detector,
    psf = psf,
    niters = niters,
    parametric = parametric,
    rng = rng,
    logposterior_rng = logposterior_rng,
    msdrescale = msdrescale,
    countbrightness = countbrightness,
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
- `rng`: The random number generator (default: `Random.default_rng()`). On GPU, array draws use `CUDA.default_rng()` instead; seed it separately for reproducible GPU runs.
- `logposterior_rng`: Saved scores include the brightness prior and use standardized increments. Off candidates contribute their expected log prior (default: `nothing`) or an independent draw of that log prior using this generator. This is an off-prior score proxy, not a marginalized density or the density of the stored off tracks. Use a generator distinct from `rng` to leave sampling unchanged.
- `msdrescale`: Also run the rescaling MSD move after the MSD's Gibbs draw, which stretches the on tracks with the MSD (default: `false`; `docs/msdrescale.qmd`). It costs one likelihood evaluation per iteration.
- `countbrightness`: Also run the joint count–brightness move after the emitter-count update, which switches one emitter on or off and rescales the brightness by n/n′ (default: `false`; `docs/countbrightness.qmd`). Nonparametric sweep only.
Both moves leave the posterior unchanged; they change only how the chain moves.
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
    rng::AbstractRNG = Random.default_rng(),
    logposterior_rng::Union{Nothing,AbstractRNG} = nothing,
    msdrescale::Bool = false,
    countbrightness::Bool = false,
) where {T}
    isnothing(annealing) && (annealing = ConstantAnnealing{T}(1))
    sample_type = typeof(Sample(tracks.onchunk, msd, brightness))
    chain = Chain(sample_type[], sizelimit, annealing)
    runMCMC!(
        chain = chain,
        tracks = tracks,
        msd = msd,
        brightness = brightness,
        detector = detector,
        psf = psf,
        niters = niters,
        parametric = parametric,
        rng = rng,
        logposterior_rng = logposterior_rng,
        msdrescale = msdrescale,
        countbrightness = countbrightness,
    )
    return chain
end
