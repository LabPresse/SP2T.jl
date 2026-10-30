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

"""
    PointSpreadFunction{T}

An abstract type representing a generic point spread function (PSF). This serves as a base type for defining specific PSF implementations. The type parameter `T` can be used to specify the type of data.
"""
abstract type PointSpreadFunction{T} end

"""
    RandomVariable{T}

An abstract type representing a generic random variable. A random variable should have at least three fields, `value::T`, `prior::P`, and `fixed::Bool`.
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

Store in `r` the `b`-weighted sum over pixels of each frame of `A`; a more efficient way to implement `r .= vec(sum(A .* b, dims = (1, 2)))`.
"""
framesum!(r::AbstractVector{T}, A::AbstractArray{T,3}, b::AbstractMatrix{T}) where {T} =
    mul!(r, transpose(reshape(A, length(b), :)), vec(b))

"""
    devicerng(rng::AbstractRNG, x::AbstractArray)

The random number generator that fills `x`: `rng` for arrays in CPU memory. A CPU generator cannot fill device memory, so `SP2TCUDAExt` returns `CUDA.default_rng()` for CUDA arrays; seed that one separately for reproducible GPU runs.
"""
devicerng(rng::AbstractRNG, ::AbstractArray) = rng
devicerng(rng::AbstractRNG, x::SubArray) = devicerng(rng, parent(x))

"""
    randc(rng::AbstractRNG, logp::AbstractVector{<:Real})

Sample from a categorical distribution given the log-probabilities `logp` using the Gumbel trick.
"""
randc(rng::AbstractRNG, logp::AbstractVector{<:Real}) =
    argmax(logp .- log.(randexp!(rng, similar(logp))))

logrand!(rng::AbstractRNG, x::AbstractArray) = x .= log.(rand!(devicerng(rng, x), x))
neglogrand!(rng::AbstractRNG, x::AbstractArray) = x .= .-logrand!(rng, x)

"""
    _rand(rng::AbstractRNG, prior::P, Δparams::NTuple{2,T})
Generate a random value based on the given `prior` distribution and a tuple of parameter differences `Δparams`.
"""
function _rand(rng::AbstractRNG, prior::P, Δparams::NTuple{2,T}) where {T,P}
    postparams = params(prior) .+ Δparams
    return rand(rng, P(postparams...))
end

_randn!(
    rng::AbstractRNG,
    x::AbstractVecOrMat{T},
    μ::AbstractVector{T},
    σ::AbstractVector{T},
) where {T} = x .= muladd.(randn!(devicerng(rng, x), x), σ, μ)

_randn!(rng::AbstractRNG, x::AbstractArray{T}, σ::Union{T,AbstractMatrix{T}}) where {T} =
    randn!(devicerng(rng, x), x) .*= σ

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
    mrw_propose(rng::AbstractRNG, P, xᵒ::T)

Return a proposed value generated by a multiplicative random walk. `P` is used to sample the multiplicative factor and `xᵒ` is the previous value. This function also returns the contribution to the overall acceptance ratio (in the log-space) of the proposal.
"""
function mrw_propose(rng::AbstractRNG, P::Beta{T}, xᵒ::T) where {T<:AbstractFloat}
    xᵖ = rand(rng, Bool) ? xᵒ * rand(rng, P) : xᵒ / rand(rng, P)
    return xᵖ, log(xᵖ) - log(xᵒ)
end

"""
    arw_propose!(rng::AbstractRNG, xᵖ::AbstractArray{T}, xᵒ::AbstractArray{T}, scaling::AbstractArray{T})

Overwrite values in `xᵖ` with a additive random walk from `xᵒ`.`scaling` is a scaling factor for the proposal. 
"""
arw_propose!(
    rng::AbstractRNG,
    xᵖ::AbstractArray{T},
    xᵒ::AbstractArray{T},
    scaling::AbstractArray{T},
) where {T<:AbstractFloat} = xᵖ .= muladd.(randn!(devicerng(rng, xᵖ), xᵖ), scaling, xᵒ)

"""
    _resize3(x::AbstractArray{T,3}, newsz::Integer; fill::Union{Nothing,T} = nothing)

Internal function to resize a 3D array in the 3rd dimension.
"""
function _resize3(
    x::AbstractArray{T,3},
    newsz::Integer;
    fill::Union{Nothing,T} = nothing,
) where {T}
    y = similar(x, size(x, 1), size(x, 2), newsz)
    copyto!(y, x)
    !isnothing(fill) && fill!(view(y, :, :, size(x, 3)+1:newsz), fill)
    return y
end

"""
    boolcopyto!(dest::AbstractArray{T,3}, src::AbstractArray{T,3}, i::AbstractVector{T})

Copy values from `src` to `dest` at indices where `i` is 1.
"""
function boolcopyto!(
    dest::AbstractArray{T,3},
    src::AbstractArray{T,3},
    i::AbstractVector{T},
) where {T}
    @. dest .+= i .* (src .- dest)
    return dest
end
