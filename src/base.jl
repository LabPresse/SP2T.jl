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
    PointSpreadFunction{T}

An abstract type representing a generic point spread function (PSF). This serves as a base type for defining specific PSF implementations. The type parameter `T` can be used to specify the type of data.
"""
abstract type PointSpreadFunction{T} end

"""
    RandomVariable{T}

An abstract type representing a generic random variable. It is used mostly for prototyping. Based on the initial design, a random variable should have at least three fields, `value::T`, `prior::P`, and `fixed::Bool`.
"""
abstract type RandomVariable{T} end

isfixed(x::RandomVariable) = x.fixed

"""
    LogLikelihoodAux{T}

An abstract type representing a generic log-likelihood auxiliary variable. The type parameter `T` can be used to specify the type of data.
"""
abstract type LogLikelihoodAux{T} end

"""
    LogLikelihoodArray{T<:AbstractFloat, A<:AbstractArray{T,3}, V<:AbstractVector{T}}

This structure is designed to encapsulate auxiliary variables related to log-likelihood computations. The `pixel::A`, an array storing the per-pixel log-likelihoods. Its shape should match that of the detector readout, encoding likelihood information for each pixel across frames. `frame::V`, A vector containing the per-frame log-likelihoods. `means::NTuple{2,A}`, A tuple containing two arrays of per-pixel expected photon counts. The first array holds values computed from the current parameter set, while the second contains values based on proposed parameters.
"""
struct LogLikelihoodArray{T<:AbstractFloat,A<:AbstractArray{T,3},V<:AbstractVector{T}} <:
       LogLikelihoodAux{T}
    pixel::A
    frame::V
    means::NTuple{2,A}
end

"""
    LogLikelihoodArray{T}(frames::AbstractArray) where {T<:AbstractFloat}

Constructor for the LogLikelihoodArray type.
"""
LogLikelihoodArray{T}(frames::AbstractArray) where {T<:AbstractFloat} = LogLikelihoodArray(
    similar(frames, T),
    similar(frames, T, size(frames, 3)),
    (similar(frames, T), similar(frames, T)),
)

"""
    framesum!(r::AbstractVector{T}, A::AbstractArray{T,3}, b::AbstractMatrix{T}) where {T}

A more efficient way to implememnt `r .= sum(A .* b, dims=3)`.
"""
framesum!(r::AbstractVector{T}, A::AbstractArray{T,3}, b::AbstractMatrix{T}) where {T} =
    mul!(r, transpose(reshape(A, length(b), :)), vec(b))

unionalltypeof(::Gamma) = Gamma
unionalltypeof(::InverseGamma) = InverseGamma

"""
    randc(logp::AbstractVector{<:Real})

Sample from a categoriacal distribution given the log-probabilities `logp` using the Gumbel trick.
"""
randc(logp::AbstractVector{<:Real}) = argmax(logp .- log.(randexp!(similar(logp))))

logrand!(x::AbstractArray) = x .= log.(rand!(x))
neglogrand!(x::AbstractArray) = x .= .-logrand!(x)

function _rand(prior::P, Δparams::NTuple{2,T}) where {T,P}
    postparams = params(prior) .+ Δparams
    return rand(P(postparams...))
end

_randn!(x::AbstractVecOrMat{T}, μ::AbstractVector{T}, σ::AbstractVector{T}) where {T} =
    x .= muladd.(randn!(x), σ, μ)

_randn!(x::AbstractArray{T}, σ::Union{T,AbstractMatrix{T}}) where {T} = randn!(x) .*= σ

function _randn!(x::AbstractArray{T,3}, σ::T, σ₀::AbstractVecOrMat{T}) where {T}
    _randn!(x, σ)
    @views @. x[1, :, :] *= σ₀ / σ
end

"""
    diff²!(Δx²::AbstractArray{T,3}, x::AbstractArray{T,3}, y::AbstractArray{T,3}) where {T}

Compute the squared differences between successive slices along the first dimension of two 3D arrays, `y` and `x`, and store the results in the provided 3D array `Δx²`.
"""
diff²!(Δx²::AbstractArray{T,3}, x::AbstractArray{T,3}, y::AbstractArray{T,3}) where {T} =
    @views Δx² .= (x[2:end, :, :] .- y[1:end-1, :, :]) .^ 2

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

Δlogpdf(P::Gamma{T}, xᵖ::T, xᵒ::T) where {T<:AbstractFloat} =
    (shape(P) - 1) * log(xᵖ / xᵒ) - (xᵖ - xᵒ) / scale(P)

"""
    mrw_propose(P, xᵒ::T)

Return a proposed value generated a multiplicative random walk. `P` is used to sample the multiplicative factor and `xᵒ` is the previous value. This function also  returns the contribution to the overall acceptance ratio (in the log-space) of the proposal.
"""
function mrw_propose(P::Beta{T}, xᵒ::T) where {T<:AbstractFloat}
    xᵖ = rand(Bool) ? xᵒ * rand(P) : xᵒ / rand(P)
    return xᵖ, log(xᵖ) - log(xᵒ)
end
function mrw_propose(P::Union{BetaPrime{T},Gamma{T}}, xᵒ::T) where {T<:AbstractFloat}
    ϵ = rand(P)
    xᵖ = ϵ * xᵒ
    return xᵖ, log(xᵒ) + logpdf(P, 1 / ϵ) - log(xᵖ) - logpdf(P, ϵ) #* Needs test
end

function _resize3(
    x::AbstractArray{T,3},
    newsz::Integer;
    fill = Union{Nothing,T} = nothing,
) where {T}
    y = similar(x, size(x, 1), size(x, 2), newsz)
    copyto!(y, x)
    !isnothing(fill) && fill!(view(y, :, :, size(x, 3)+1:newsz), fill)
    return y
end
# get_Δlogprior(xᵖ::T, xᵒ::T, distr::Gamma{T}) where {T<:AbstractFloat} =
#     (shape(distr) - 1) * log(xᵖ / xᵒ) - (xᵖ - xᵒ) / scale(distr)
