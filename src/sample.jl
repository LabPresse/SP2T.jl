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
        i::Int = 0,
        T::FT = 1.0,
        logℙ::FT = NaN,
    ) where {FT<:AbstractFloat} = new{FT}(x, D, h, F, i, T, logℙ)
end

get_B(s::Sample) = size(s.x, 2)

get_type(s::Sample{FT}) where {FT} = FT

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

get_type(s::FullSample{FT}) where {FT} = FT

Sample(s::FullSample) = Sample(s.x[:, s.b, :], s.D, s.h, s.F, s.i, s.T, s.logℙ)

# Sample(s::FullSample) = Sample(s.x[:, 1:get_B(s), :], s.D, s.h, s.F, s.i, s.T, s.logℙ)

function FullSample(s::Sample{FT}, M::Integer) where {FT<:AbstractFloat}
    (~, B, N) = size(s.x, 3)
    M < B && error("Weak limit is too small!")
    b = zeros(Bool, M)
    b[1:B] = true
    x = Array{FT,3}(undef, 3, M, N)
    x[:, 1:B, :] = s.x
    return FullSample(b, x, s.D, s.h, s.F, iszero(s.i) ? 1 : s.i, s.T, s.logℙ)
    #TODO initialize T and logℙ better
end
