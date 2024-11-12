abstract type Detector{T} end

# abstract type StreamDetector{T} <: Detector{T} end

abstract type PointSpreadFunction{T} end

abstract type RandomVariable{T} end

unionalltypeof(::Gamma) = Gamma
unionalltypeof(::InverseGamma) = InverseGamma
unionalltypeof(::Matrix) = Matrix
unionalltypeof(::Vector) = Vector
unionalltypeof(::Array) = Array

anneal(logℒ::T, 𝑇::T) where {T} = logℒ / 𝑇
anneal!(logℒ::AbstractVector{T}, 𝑇::T) where {T} = logℒ ./= 𝑇

randc(logp::AbstractArray) = argmax(logp .- log.(randexp!(similar(logp))))

logrand!(x::AbstractArray) = x .= log.(rand!(x))

neglogrand!(x::AbstractArray) = x .= .-log.(rand!(x))

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