abstract type AbstractSample end

mutable struct Sample{FT<:AbstractFloat} <: AbstractSample
    x::Array{FT,3}
    D::FT
    h::FT
    F::Matrix{FT}
    i::Int # iteration
    𝕋::FT # temperature
    logℙ::FT # log posterior
    Sample(x::Array{FT,3}, D::FT, h::FT, F::Matrix{FT}) where {FT<:AbstractFloat} =
        new{FT}(x, D, h, F, 0, 1, FT(NaN))
    Sample{FT}(
        x::Array{FT,3},
        D::FT,
        h::FT,
        F::Matrix{FT},
        i::Int,
        𝕋::FT,
        logℙ::FT,
    ) where {FT<:AbstractFloat} = new{FT}(x, D, h, F, i, 𝕋, logℙ)
end

get_B(s::Sample) = size(s.x, 2)

ftypeof(s::Sample{FT}) where {FT} = FT

# FullSample contains auxiliary variables
mutable struct FullSample{FT<:AbstractFloat} <: AbstractSample
    b::BitVector
    x::Array{FT,3}
    D::FT
    h::FT
    F::Matrix{FT}
    i::Int # iteration
    𝕋::FT # temperature
    logℙ::FT # log posterior
    FullSample(FT) = new{FT}()
    FullSample(
        b::BitVector,
        x::Array{FT,3},
        D::FT,
        h::FT,
        F::Matrix{FT},
        i::Int = 0,
        T::FT = 1.0,
        logℙ::FT = NaN,
    ) where {FT<:AbstractFloat} = new{FT}(b, x, D, h, F, i, T, logℙ)
end

get_B(s::FullSample) = count(s.b)

get_M(s::FullSample) = size(s.x, 2)

ftypeof(s::FullSample{FT}) where {FT} = FT

view_x(s::FullSample) = @view s.x[:, 1:get_B(s), :]

view_𝕩(s::FullSample) = @view s.x[:, get_B(s)+1:end, :]