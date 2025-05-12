"""
    Detector{T}

An abstract type representing a generic detector. This serves as a base type for defining specific detector implementations. The type parameter `T` can be used to specify the type of data.
"""
abstract type Detector{T} end

"""
    PixelDetector{T} <: Detector{T}

An abstract type representing a pixel-based detector. This serves as a 
base type for defining specific pixel detector implementations. The type parameter `T` can be used to specify the type of data.
"""
abstract type PixelDetector{T} <: Detector{T} end

# abstract type StreamDetector{T} <: Detector{T} end

"""
    AbstractEMCCD{T} <: PixelDetector{T}

An abstract type representing an EMCCD detector. This serves as a 
base type for defining specific EMCCD implementations. The type parameter `T` can be used to specify the type of data.
"""
abstract type AbstractEMCCD{T} <: PixelDetector{T} end

"""
    PointSpreadFunction{T}

An abstract type representing a generic point spread function (PSF). This serves as a base type for defining specific PSF implementations. The type parameter `T` can be used to specify the type of data.
"""
abstract type PointSpreadFunction{T} end

"""
    GaussianPSF{T} <: PointSpreadFunction{T}

An abstract type representing a generic Gaussian PSF (Gaussain in each 𝑧-plane). This serves as a base type for defining specific Gaussian PSF implementations. The type parameter `T` can be used to specify the type of data.
"""
abstract type GaussianPSF{T} <: PointSpreadFunction{T} end

# abstract type RandomVariable{T} end
# An abstract type for random variables. Not used right now. My current design: each RV <: RandomVariable should contain value and prior.

# abstract type SP2TDistribution{T} end

"""
    AbstractAnnealing{T}

An abstract type representing a generic annealing schedule. This serves as a base type for defining specific annealing implementations. The type parameter `T` can be used to specify the type of data.
"""
abstract type AbstractAnnealing{T} end

"""
    AbstractTrackParts{T}

An abstract type representing a generic track part. The type parameter `T` can be used to specify the type of data.
"""
abstract type AbstractTrackParts{T} end

"""
    LogLikelihoodAux{T}

An abstract type representing a generic log-likelihood auxiliary variable. The type parameter `T` can be used to specify the type of data.
"""
abstract type LogLikelihoodAux{T} end

"""
    struct LogLikelihoodArray{T<:AbstractFloat, A<:AbstractArray{T,3}, V<:AbstractVector{T}}

A structure representing a log-likelihood array with specific type constraints. This structure is designed to encapsulate data and auxiliary information related to log-likelihood computations. The `pixel::A`, an array storing the per-pixel log-likelihoods. Its shape should match that of the detector readout, encoding likelihood information for each pixel across frames. `frame::V`, A vector containing the per-frame log-likelihoods. `means::NTuple{2,A}`, A tuple containing two arrays of per-pixel expected photon counts. The first array holds values computed from the current parameter set, while the second contains values based on proposed parameters.
"""
struct LogLikelihoodArray{T<:AbstractFloat,A<:AbstractArray{T,3},V<:AbstractVector{T}} <:
       LogLikelihoodAux{T}
    pixel::A
    frame::V
    means::NTuple{2,A}
end

function LogLikelihoodArray{T}(frames::AbstractArray{<:Real,3}) where {T<:AbstractFloat}
    nframes = size(frames, 3)
    return LogLikelihoodArray(
        similar(frames, T),
        similar(frames, T, nframes),
        (similar(frames, T), similar(frames, T)),
    )
end

framesum!(r::AbstractVector{T}, A::AbstractArray{T,3}, b::AbstractMatrix{T}) where {T} =
    mul!(r, transpose(reshape(A, length(b), :)), vec(b))

mutable struct NTracks{T<:AbstractFloat,V<:AbstractVector{T}}
    value::Int
    logprior::T
    loglikelihood::V
    logposterior::V
end

struct TrackParts{T<:AbstractFloat,A<:AbstractArray{T},P} <: AbstractTrackParts{T}
    value::A
    presence::A
    displacement²::A
    effvalue::A
    prior::P
end

struct MHTrackParts{T<:AbstractFloat,A<:AbstractArray{T},V<:AbstractVector{T}} <:
       AbstractTrackParts{T}
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
    T<:AbstractFloat,
    A<:AbstractArray{T,3},
    NT<:NTracks{T},
    TR<:TrackParts{T},
    MH<:MHTrackParts{T},
}
    values::NTuple{2,A}
    presences::NTuple{2,A}
    displacement²s::NTuple{2,A}
    effvalues::NTuple{2,A}
    ntracks::NT
    onpart::TR
    offpart::TR
    proposals::MH
end

mutable struct MeanSquaredDisplacement{T<:AbstractFloat,P}
    value::T
    prior::P
    fixed::Bool
end

mutable struct Brightness{T<:AbstractFloat,P}
    value::T
    prior::P
    proposal::Beta{T}
    fixed::Bool
    counter::Vector{Int}
end

isfixed(brightness::Brightness) = brightness.fixed
isfixed(msd::MeanSquaredDisplacement) = msd.fixed

unionalltypeof(::Gamma) = Gamma
unionalltypeof(::InverseGamma) = InverseGamma

anneal(logℒ::T, 𝑇::T) where {T} = logℒ / 𝑇
anneal!(logℒ::AbstractArray{T}, 𝑇::T) where {T} = logℒ ./= 𝑇

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

function elconvert(TargetType::Type{T}, A::AbstractArray) where {T}
    if eltype(A) === TargetType
        return A
    else
        B = similar(A, TargetType)
        copyto!(B, A)
        return B
    end
end

dimsmatch(A::AbstractArray, B::AbstractArray; dims::Union{Integer,AbstractUnitRange}) =
    size(A)[dims] == size(B)[dims]

get_Δlogprior(xᵖ::T, xᵒ::T, distr::Gamma{T}) where {T<:AbstractFloat} =
    (shape(distr) - 1) * log(xᵖ / xᵒ) - (xᵖ - xᵒ) / scale(distr)
