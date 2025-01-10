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
    tracks::TrackParts{T},
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

mutable struct Chain{T<:AbstractFloat,VofS<:Vector{<:Sample{T}},A<:AbstractAnnealing{T}}
    samples::VofS
    sizelimit::Int
    annealing::A
end

function Base.getproperty(c::Chain, s::Symbol)
    if s === :msd
        return [sample.diffusivity for sample in getfield(c, :samples)]
    elseif s === :nemitters
        return [size(sample.tracks, 3) for sample in getfield(c, :samples)]
    elseif s === :logposterior
        return [sample.log𝒫 for sample in getfield(c, :samples)]
    elseif s === :loglikelihood
        return [sample.logℒ for sample in getfield(c, :samples)]
    elseif s === :stride
        return length(getfield(c, :samples)) == 1 ? 1 :
               getfield(c, :samples)[2].iteration - getfield(c, :samples)[1].iteration
    else
        return getfield(c, s)
    end
end

Base.length(c::Chain) = length(c.samples)

isfull(chain::Chain) = length(chain.samples) == chain.sizelimit

findmap(chain::Chain; burn_in::Real = 0) = @views findmax(chain.logposterior[burn_in+1:end])

findmle(chain::Chain; burn_in::Real = 0) =
    @views findmax(chain.loglikelihood[burn_in+1:end])

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
    measurements::AbstractArray{<:Union{T,Integer}},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    iter::Integer,
    𝑇::T,
) where {T}
    if iter % chain.stride == 0
        log𝒫, logℒ = log𝒫logℒ(tracks, msd, brightness, measurements, detector, psf)
        push!(chain.samples, Sample(tracks.onpart, msd, brightness, iter, 𝑇, log𝒫, logℒ))
        isfull(chain) && shrink!(chain)
    end
    return chain
end

# saveperiod(chain::Chain) =
#     length(chain.samples) == 1 ? 1 : chain.samples[2].iteration - chain.samples[1].iteration

function log𝒫logℒ(
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    measurements::AbstractArray{<:Union{T,Integer},3},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
) where {T}
    seteffvalue!(tracks.onpart)
    @views pxcounts!(detector, tracks.onpart.effvalue, brightness.value, psf)
    logℒ1 = logℒ!(detector, measurements)
    log𝒫1 =
        logℒ1 +
        logprior(tracks.onpart, msd.value) +
        logprior(msd) +
        logprior(tracks.ntracks)
    return log𝒫1, logℒ1
end

function parametricMCMC!(
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    measurements::AbstractArray{<:Union{T,Integer}},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    update_onpart!(tracks, msd.value, brightness.value, measurements, detector, psf, 𝑇)
    setdisplacement²!(tracks)
    update!(msd, tracks.displacement², 𝑇)
    return tracks, msd
end

function nonparametricMCMC!(
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    measurements::AbstractArray{<:Union{T,Integer}},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    simulate!(tracks.offpart, msd.value)
    if any(tracks)
        update_onpart!(tracks, msd.value, brightness.value, measurements, detector, psf, 𝑇)
        onshuffle!(tracks)
    end

    seteffvalue!(tracks)
    update!(
        tracks.ntracks,
        tracks.effvalue,
        brightness.value,
        measurements,
        detector,
        psf,
        𝑇,
    )
    reassign!(tracks)

    setdisplacement²!(tracks)
    update!(msd, tracks.displacement², 𝑇)
    return tracks, msd
end

function runMCMC!(
    chain::Chain{T},
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    measurements::AbstractArray{<:Union{T,Integer}},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    niters::Integer,
    parametric::Bool,
) where {T}
    prev_niters = chain.samples[end].iteration
    reset!(detector, 1)
    tracks.ntracks.logℒ[1] = logℒ!(detector, measurements)
    nextsample! = parametric ? parametricMCMC! : nonparametricMCMC!
    @showprogress 1 "Computing..." for iter in prev_niters .+ (1:niters)
        𝑇 = temperature(chain, iter)
        nextsample!(tracks, msd, brightness, measurements, detector, psf, 𝑇)
        extend!(chain, tracks, msd, brightness, measurements, detector, psf, iter, 𝑇)
    end
end

function runMCMC(;
    tracks::Tracks{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    measurements::AbstractArray{<:Union{T,Integer}},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    niters::Integer = 1000,
    sizelimit::Integer = 1000,
    annealing::Union{AbstractAnnealing{T},Nothing} = nothing,
    parametric::Bool = false,
) where {T}
    isnothing(annealing) && (annealing = ConstantAnnealing{T}(1))
    chain = Chain([Sample(tracks.onpart, msd, brightness)], sizelimit, annealing)
    runMCMC!(
        chain,
        tracks,
        msd,
        brightness,
        measurements,
        detector,
        psf,
        niters,
        parametric,
    )
    return chain
end
