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
    tracks::AbstractArray{T,3},
    nemitters::Integer,
    msd::T,
    brightness::T,
    iter::Integer,
    𝑇::T,
    log𝒫::T,
    logℒ::T,
) where {T} =
    Sample(collect(view(tracks, :, :, 1:nemitters)), msd, brightness, iter, 𝑇, log𝒫, logℒ)

Sample(
    tracks::Tracks{T},
    nemitters::NEmitters{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    iter::Integer,
    𝑇::T,
    log𝒫::T,
    logℒ::T,
) where {T} =
    Sample(tracks.value, nemitters.value, msd.value, brightness.value, iter, 𝑇, log𝒫, logℒ)

Sample(
    tracksᵥ::AbstractArray{T,3},
    nemittersᵥ::Integer,
    msdᵥ::T,
    brightnessᵥ::T,
) where {T<:AbstractFloat} = Sample(
    tracksᵥ,
    nemittersᵥ,
    msdᵥ,
    brightnessᵥ,
    0,
    oneunit(T),
    convert(T, NaN),
    convert(T, NaN),
)

Sample(
    tracks::Tracks{T},
    nemitters::NEmitters{T},
    brightness::MeanSquaredDisplacement{T},
    𝑏𝑟𝑖𝑔ℎ𝑡𝑛𝑒𝑠𝑠::Brightness{T},
) where {T} = Sample(tracks.value, nemitters.value, brightness.value, 𝑏𝑟𝑖𝑔ℎ𝑡𝑛𝑒𝑠𝑠.value)

function Base.getproperty(sample::Sample, s::Symbol)
    if s === :nemitters
        return size(sample.tracks, 3)
    else
        return getfield(sample, s)
    end
end

# get_B(v::AbstractVector{Sample}) = [size(s.tracks, 2) for s in v]

# get_D(v::AbstractVector{Sample}) = [s.D for s in v]

# get_h(v::AbstractVector{Sample}) = [s.h for s in v]

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
    nemitters::NEmitters{T},
    msd::MeanSquaredDisplacement{T},
    brightness::Brightness{T},
    measurements::AbstractArray{<:Union{T,Integer}},
    detector::Detector{T},
    psf::PointSpreadFunction{T},
    iter::Integer,
    𝑇::T,
) where {T}
    if iter % chain.stride == 0
        log𝒫, logℒ =
            log𝒫logℒ(tracks, nemitters, msd, brightness, measurements, detector, psf)
        push!(
            chain.samples,
            Sample(
                tracks.value,
                nemitters.value,
                msd.value,
                brightness.value,
                iter,
                𝑇,
                log𝒫,
                logℒ,
            ),
        )
        isfull(chain) && shrink!(chain)
    end
    return chain
end

# saveperiod(chain::Chain) =
#     length(chain.samples) == 1 ? 1 : chain.samples[2].iteration - chain.samples[1].iteration