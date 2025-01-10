abstract type Detector{T} end
abstract type PixelDetector{T} <: Detector{T} end

# abstract type StreamDetector{T} <: Detector{T} end

abstract type PointSpreadFunction{T} end
abstract type GaussianPSF{T} <: PointSpreadFunction{T} end

abstract type RandomVariable{T} end

abstract type SP2TDistribution{T} end

abstract type AbstractAnnealing{T} end

abstract type AbstractTrackParts{T} end

mutable struct NTracks{T,V}
    value::Int
    logprior::V
    logℒ::V
    log𝒫::V
end

struct TrackParts{T,A<:AbstractArray{T},P<:SP2TDistribution{T}} <: AbstractTrackParts{T}
    value::A
    presence::A
    displacement²::A
    effvalue::A
    prior::P
end

struct MHTrackParts{T,A<:AbstractArray{T},V<:AbstractVector{T}} <: AbstractTrackParts{T}
    value::A
    presence::A
    displacement²::A
    effvalue::A
    ΣΔdisplacement²::V
    perturbsize::V
    logacceptance::V
    acceptance::V
    counter::Matrix{Int}
end

mutable struct Tracks{
    T,
    A<:AbstractArray{T},
    NT<:NTracks,
    TR<:TrackParts{T},
    MH<:MHTrackParts{T},
}
    fullvalue::A
    fullpresence::A
    fulldisplacement²::A
    fulleffvalue::A
    ntracks::NT
    onpart::TR
    offpart::TR
    proposals::MH
end

function Base.getproperty(tracks::Tracks, s::Symbol)
    if s === :value
        return selectdim(getfield(tracks, :fullvalue), 4, 1)
    elseif s === :presence
        return selectdim(getfield(tracks, :fullpresence), 4, 1)
    elseif s === :displacement²
        return selectdim(getfield(tracks, :fulldisplacement²), 4, 1)
    elseif s === :effvalue
        return selectdim(getfield(tracks, :fulleffvalue), 4, 1)
    else
        return getfield(tracks, s)
    end
end

unionalltypeof(::Gamma) = Gamma
unionalltypeof(::InverseGamma) = InverseGamma
unionalltypeof(::Matrix) = Matrix
unionalltypeof(::Vector) = Vector
unionalltypeof(::Array) = Array

anneal(logℒ::T, 𝑇::T) where {T} = logℒ / 𝑇
anneal!(logℒ::AbstractVector{T}, 𝑇::T) where {T} = logℒ ./= 𝑇

randc(logp::AbstractArray) = argmax(logp .- log.(randexp!(similar(logp))))

logrand!(x::AbstractArray) = x .= log.(rand!(x))
neglogrand!(x::AbstractArray) = x .= .-logrand!(x)

function _rand(prior::P, Δparams::NTuple{2,T}) where {T,P}
    postparams = params(prior) .+ Δparams
    return rand(P(postparams...))
end

function _randn!(x::AbstractArray{T}, σ::Union{T,AbstractMatrix{T}}) where {T}
    randn!(x)
    x .*= σ
end

function _randn!(x::AbstractArray{T,3}, σ::T, σ₀::AbstractVecOrMat{T}) where {T}
    _randn!(x, σ)
    @views @. x[1, :, :] *= σ₀ / σ
end

"""
    diff²!(Δx²::AbstractArray{T,3}, x::AbstractArray{T,3}, prev::AbstractArray{T,3}) where {T}

Compute the squared differences between successive slices along the first dimension of two 3D arrays, `prev` and `x`, and store the results in the provided 3D array `Δx²`.

# Arguments
- `Δx²::AbstractArray{T,3}`: A pre-allocated 3D array to store the computed squared differences. The size of `Δx²` must be `(size(y, 1) - 1, size(y, 2), size(y, 3))`.
- `x::AbstractArray{T,3}`: A 3D array representing the "current" set of values.
- `prev::AbstractArray{T,3}`: A 3D array representing the "previous" set of values. Must have the same dimensions as `x`.

"""

diff²!(Δx²::AbstractArray{T,3}, x::AbstractArray{T,3}, prev::AbstractArray{T,3}) where {T} =
    @views Δx² .= (x[2:end, :, :] .- prev[1:end-1, :, :]) .^ 2

diff²!(Δx²::AbstractArray{T,3}, x::AbstractArray{T,3}) where {T} = diff²!(Δx², x, x)

logaccept!(
    ans::AbstractVector,
    logratio::AbstractVector;
    start::Integer = 1,
    step::Integer = 1,
) = ans[start:step:end] .= (@view logratio[start:step:end]) .> 0