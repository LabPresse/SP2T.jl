abstract type AbstractSample end

mutable struct Sample{T<:AbstractFloat} <: AbstractSample
    tracks::Array{T,3}
    diffusivity::T
    brightness::T
    iteration::Int # iteration
    temperature::T # temperature
    logposterior::T # log posterior
    loglikelihood::T # log likelihood
    Sample(tracks::Array{T,3}, diffusivity::T, brightness::T) where {T<:AbstractFloat} =
        new{T}(tracks, diffusivity, brightness, 0, 1, T(NaN))
    Sample(; tracks::Array{T,3}, diffusivity::T, brightness::T) where {T<:AbstractFloat} =
        new{T}(tracks, diffusivity, brightness, 0, 1, T(NaN))
    Sample{T}(
        tracks::Array{T,3},
        diffusivity::T,
        brightness::T,
        iteration::Int,
        temperature::T,
        logposterior::T,
        loglikelihood::T,
    ) where {T<:AbstractFloat} = new{T}(
        tracks,
        diffusivity,
        brightness,
        iteration,
        temperature,
        logposterior,
        loglikelihood,
    )
end

get_B(s::Sample) = size(s.tracks, 2)

_eltype(s::Sample{T}) where {T} = T

function simulate(;
    param::ExperimentalParameter{FT},
    framecount::Integer,
    emittercount::Integer,
    diffusivity::Real,
    brightness::Real,
    init_pos_prior::Union{Missing,MultivariateDistribution} = missing,
    device::Device = CPU(),
) where {FT}
    diffusivity = convert(FT, diffusivity)
    brightness = convert(FT, brightness)
    if ismissing(init_pos_prior)
        init_pos_prior = default_init_pos_prior(param)
    end
    tracks = Array{FT,3}(undef, 3, emittercount, framecount)
    simulate!(tracks, init_pos_prior, diffusivity, param.period, device)
    return Sample(tracks, diffusivity, brightness)
end

# # FullSample contains auxiliary variables
# mutable struct FullSample{FT<:AbstractFloat} <: AbstractSample
#     b::BitVector
#     x::Array{FT,3}
#     D::FT
#     h::FT
#     F::Matrix{FT}
#     G::Array{FT,3}
#     i::Int # iteration
#     𝑇::FT # temperature
#     ln𝒫::FT # log posterior
#     FullSample(
#         b::BitVector,
#         x::Array{FT,3},
#         D::FT,
#         h::FT,
#         F::Matrix{FT},
#         G::Array{FT,3},
#         i::Int = 0,
#         𝑇::FT = 1.0,
#         ln𝒫::FT = NaN,
#     ) where {FT<:AbstractFloat} = new{FT}(b, x, D, h, F, G, i, 𝑇, ln𝒫)
# end

# get_B(s::FullSample) = count(s.b)

# get_M(s::FullSample) = size(s.x, 2)

# ftypeof(s::FullSample{FT}) where {FT} = FT

# view_x(s::FullSample) = @view s.x[:, 1:get_B(s), :]

# view_𝕩(s::FullSample) = @view s.x[:, get_B(s)+1:end, :]