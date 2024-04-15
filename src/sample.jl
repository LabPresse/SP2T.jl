abstract type AbstractSample end

mutable struct Sample{FT<:AbstractFloat} <: AbstractSample
    tracks::Array{FT,3}
    diffusivity::FT
    brightness::FT
    iteration::Int # iteration
    temperature::FT # temperature
    logposterior::FT # log posterior
    loglikelihood::FT # log likelihood
    Sample(x::Array{FT,3}, D::FT, h::FT) where {FT<:AbstractFloat} =
        new{FT}(x, D, h, 0, 1, FT(NaN))
    Sample(;
        tracks::Array{FT,3},
        diffusion_coefficient::FT,
        emission_rate::FT,
    ) where {FT<:AbstractFloat} =
        new{FT}(tracks, diffusion_coefficient, emission_rate, 0, 1, FT(NaN))
    Sample{FT}(
        x::Array{FT,3},
        D::FT,
        h::FT,
        i::Int,
        𝑇::FT,
        ln𝒫::FT,
        lnℒ::FT,
    ) where {FT<:AbstractFloat} = new{FT}(x, D, h, i, 𝑇, ln𝒫, lnℒ)
end

get_B(s::Sample) = size(s.tracks, 2)

_eltype(s::Sample{FT}) where {FT} = FT

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