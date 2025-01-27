struct SPAD{
    T<:AbstractFloat,
    V<:AbstractVector{T},
    M<:AbstractMatrix{T},
    R<:AbstractArray{UInt16,3},
} <: PixelDetector{T}
    batchsize::UInt16
    period::T
    pxsize::T
    pxbounds::NTuple{2,V}
    darkcounts::M
    filter::M
    readouts::R
end

function SPAD{T}(;
    period::Real,
    pixel_size::Real,
    darkcounts::AbstractMatrix{<:Real},
    cutoffs::Tuple{<:Real,<:Real},
    readouts::AbstractArray{UInt16,3},
    batchsize::Integer = one(UInt16),
) where {T<:AbstractFloat}
    period = convert(T, period)
    pixel_size = convert(T, pixel_size)
    darkcounts = elconvert(T, darkcounts)
    filter = similar(darkcounts)
    filter .= cutoffs[1] .< darkcounts .< cutoffs[2]
    width, height = size(darkcounts)
    pxboundsx = similar(darkcounts, width + 1)
    pxboundsx .= 0:pixel_size:width*pixel_size
    pxboundsy = similar(darkcounts, height + 1)
    pxboundsy .= 0:pixel_size:height*pixel_size
    return SPAD{T,typeof(pxboundsx),typeof(darkcounts),typeof(readouts)}(
        batchsize,
        period,
        pixel_size,
        (pxboundsx, pxboundsy),
        darkcounts,
        filter,
        readouts,
    )
end

function Base.getproperty(detector::SPAD, s::Symbol)
    if s === :framecenter
        return mean(detector.pxbounds[1]), mean(detector.pxbounds[2])
    else
        return getfield(detector, s)
    end
end

Base.size(detector::SPAD) = size(detector.darkcounts)

function reset!(
    loglikelihood::LogLikelihoodArray{T},
    detector::PixelDetector{T},
    i::Integer,
) where {T}
    loglikelihood.means[i] .= detector.darkcounts
    return loglikelihood
end

function set_pixel_loglikelihood!(
    loglikelihood::LogLikelihoodArray{T},
    detector::SPAD{T},
) where {T}
    set_binomial_loglikelihood!(
        loglikelihood.pixel,
        detector.readouts,
        loglikelihood.means[1],
        detector.batchsize,
    )
    return loglikelihood
end

function set_pixel_Δloglikelihood!(
    loglikelihood::LogLikelihoodArray{T},
    detector::SPAD{T},
) where {T}
    set_binomial_Δloglikelihood!(
        loglikelihood.pixel,
        detector.readouts,
        loglikelihood.means[1],
        loglikelihood.means[2],
        detector.batchsize,
    )
    return loglikelihood
end

function sum_pixel_loglikelihood!(
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
) where {T}
    framesum!(llarray.frame, llarray.pixel, detector.filter)
    return llarray
end

function get_loglikelihood!(
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
) where {T}
    set_pixel_loglikelihood!(llarray, detector)
    sum_pixel_loglikelihood!(llarray, detector)
    return sum(llarray.frame)
end

function set_Δloglikelihood!(
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
) where {T}
    set_pixel_Δloglikelihood!(llarray, detector)
    return sum_pixel_loglikelihood!(llarray, detector)
end

function getpsfcomponents(
    tracksᵥ::AbstractArray{T,3},
    bounds::NTuple{2,<:AbstractVector{T}},
    psf::CircularGaussian{T},
) where {T<:AbstractFloat}
    @views begin
        psfx = _erf(tracksᵥ[:, 1:1, :], bounds[1], psf.σ)
        psfy = _erf(tracksᵥ[:, 2:2, :], bounds[2], psf.σ)
    end
    return psfx, psfy
end

function getpsfcomponents(
    tracksᵥ::AbstractArray{T,3},
    bounds::NTuple{2,<:AbstractVector{T}},
    psf::CircularGaussianLorentzian{T},
) where {T<:AbstractFloat}
    @views begin
        σ = lateral_std(tracksᵥ[:, 3:3, :], psf)
        psfx = _erf(tracksᵥ[:, 1:1, :], bounds[1], σ)
        psfy = _erf(tracksᵥ[:, 2:2, :], bounds[2], σ)
    end
    return psfx, psfy
end

function addincident!(
    intensity::AbstractArray{T,3},
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    bounds::NTuple{2,<:AbstractVector{T}},
    psf::GaussianPSF{T},
) where {T<:AbstractFloat}
    psfx, psfy = getpsfcomponents(tracksᵥ, bounds, psf)
    return batched_mul!(intensity, psfx, batched_transpose(psfy), brightnessᵥ / psf.A, 1)
end

function set_poisson_mean!(
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    psf::PointSpreadFunction{T},
    i::Integer = 1,
) where {T}
    reset!(llarray, detector, i)
    addincident!(llarray.means[i], tracksᵥ, brightnessᵥ, detector.pxbounds, psf)
    return detector
end

function set_poisson_mean!(
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
    tracksᵥ₁::AbstractArray{T,3},
    tracksᵥ₂::AbstractArray{T,3},
    brightnessᵥ::T,
    psf::PointSpreadFunction{T},
) where {T}
    set_poisson_mean!(llarray, detector, tracksᵥ₁, brightnessᵥ, psf, 1)
    set_poisson_mean!(llarray, detector, tracksᵥ₂, brightnessᵥ, psf, 2)
    return detector
end

# function pxcounts!(
#     detector::PixelDetector{T},
#     tracksᵥ::AbstractArray{T,3},
#     brightnessᵥ₁::AbstractArray{T,3},
#     brightnessᵥ₂::AbstractArray{T,3},
#     psf::PointSpreadFunction{T},
# ) where {T}
#     #TODO optimize!
#     pxcounts!(detector, tracksᵥ, brightnessᵥ₁, psf, 1)
#     pxcounts!(detector, tracksᵥ, brightnessᵥ₂, psf, 2)
#     return detector
# end

function getincident(
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    darkcounts::AbstractMatrix{T},
    bounds::NTuple{2,<:AbstractVector{T}},
    psf::PointSpreadFunction{T},
) where {T}
    incident = repeat(darkcounts, 1, 1, size(tracksᵥ, 1))
    return addincident!(incident, tracksᵥ, brightnessᵥ, bounds, psf)
end

simframes!(W::AbstractArray{UInt16,3}, U::AbstractArray{<:Real,3}) =
    @. W = $rand!($similar(U)) < -expm1(-U)

simframes!(W::AbstractArray{UInt16,3}, U::AbstractArray{<:Real,3}, B::Integer) =
    @. W = rand(Binomial(B, -expm1(-U)))

function simframes(U::AbstractArray{<:Real,3}, B::Integer = 1)
    W = similar(U, UInt16)
    if B == 1
        simframes!(W, U)
    else
        simframes!(W, U, B)
    end
end

struct EMCCD{T} <: Detector{T} end