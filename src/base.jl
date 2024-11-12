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

function _randn!(x::AbstractArray{T}, σ::Union{T,AbstractMatrix{T}}) where {T}
    randn!(x)
    x .*= σ
end

function _randn!(x::AbstractArray{T,3}, σ::T, σ₀::AbstractVecOrMat{T}) where {T}
    _randn!(x, σ)
    @views @. x[1, :, :] *= σ₀ / σ
end