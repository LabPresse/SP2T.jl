abstract type AbstractSample end

mutable struct Sample{FT<:AbstractFloat} <: AbstractSample
    x::Array{FT,3}
    D::FT
    h::FT
    i::Int # iteration
    𝑇::FT # temperature
    ln𝒫::FT # log posterior
    lnℒ::FT # log likelihood
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

get_B(s::Sample) = size(s.x, 2)

ftypeof(s::Sample{FT}) where {FT} = FT

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