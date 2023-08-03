abstract type AbstractSample end

mutable struct Sample{FT<:AbstractFloat} <: AbstractSample
    x::Array{FT,3}
    D::FT
    h::FT
    F::Matrix{FT}
    i::Int # iteration
    T::FT # temperature
    logℙ::FT # log posterior
    Sample(FT) = new{FT}()
    Sample(
        x::Array{FT,3},
        D::FT,
        h::FT,
        F::Matrix{FT},
    ) where {FT<:AbstractFloat} = new{FT}(x, D, h, F, 0, 1, FT(NaN))
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
    T::FT # temperature
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