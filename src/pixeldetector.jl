function check_pixel_dimsmatch(darkcounts::AbstractArray, readouts::AbstractArray)
    dimsmatch(darkcounts, readouts, dims = 1:2) ||
        throw(DimensionMismatch("size of darkcounts dose not match size of readouts"))
end

"""
    _init_pixel_detector_params(TargetType::Type{T}, period::Real, pixel_size::Real, darkcounts::AbstractMatrix{<:Real}, cutoffs::Tuple{<:Real,<:Real}) where {T}

Initialize the parameters for a pixel detector such that they share the same data type `T`.
"""
function _init_pixel_detector_params(
    TargetType::Type{T},
    pixel_size::Real,
    darkcounts::AbstractMatrix{<:Real},
    cutoffs::Tuple{<:Real,<:Real},
) where {T}
    darkcounts = elconvert(TargetType, darkcounts)
    darkcounts[isinf.(darkcounts)] .= floatmax(eltype(darkcounts))
    darkcounts[iszero.(darkcounts)] .= floatmin(eltype(darkcounts))
    filter = similar(darkcounts) .= cutoffs[1] .< darkcounts .< cutoffs[2]
    width, height = size(darkcounts)
    pxboundsx = similar(darkcounts, width + 1) .= 0:pixel_size:width*pixel_size
    pxboundsy = similar(darkcounts, height + 1) .= 0:pixel_size:height*pixel_size
    return (pxboundsx, pxboundsy), darkcounts, filter
end

function Base.getproperty(detector::PixelDetector, s::Symbol)
    if s === :framecenter
        return mean(detector.pxbounds[1]), mean(detector.pxbounds[2])
    else
        return getfield(detector, s)
    end
end

Base.size(detector::PixelDetector) = size(detector.darkcounts)
Base.size(detector::PixelDetector, d::Integer) = size(detector.darkcounts, d)

function reset!(
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
    i::Integer,
) where {T}
    llarray.means[i] .= detector.darkcounts
    return llarray
end

function sum_pixel_loglikelihood!(
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
) where {T}
    framesum!(llarray.frame, llarray.pixel, detector.filter)
    return llarray
end

function set_frame_Δloglikelihood!(
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
) where {T}
    set_pixel_Δloglikelihood!(llarray, detector)
    return sum_pixel_loglikelihood!(llarray, detector)
end

function get_loglikelihood!(
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
) where {T}
    set_pixel_loglikelihood!(llarray, detector)
    sum_pixel_loglikelihood!(llarray, detector)
    return sum(llarray.frame)
end

function get_Δloglikelihood!(
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
) where {T}
    set_frame_Δloglikelihood!(llarray, detector)
    return sum(llarray.frame)
end

integratepsf!(
    psfcomponents::NTuple{2,<:AbstractArray{T,3}},
    relativepos::NTuple{2,<:AbstractArray{T,3}},
    tracksᵥ::AbstractArray{T,3},
    bounds::NTuple{2,<:AbstractVector{T}},
    psf::CircularGaussian{T},
) where {T<:AbstractFloat} =
    integratepsf!(psfcomponents, relativepos, tracksᵥ, bounds, psf.σ)

function integratepsf!(
    psfcomponents::NTuple{2,<:AbstractArray{T,3}},
    relativepos::NTuple{2,<:AbstractArray{T,3}},
    tracksᵥ::AbstractArray{T,3},
    bounds::NTuple{2,<:AbstractVector{T}},
    psf::CircularGaussianLorentzian{T},
) where {T<:AbstractFloat}
    @views σ = lateral_std(tracksᵥ[:, 3:3, :], psf)
    return integratepsf!(psfcomponents, relativepos, tracksᵥ, bounds, σ)
end

function integratepsf!(
    psfcomponents::NTuple{2,<:AbstractArray{T,3}},
    relativepos::NTuple{2,<:AbstractArray{T,3}},
    tracksᵥ::AbstractArray{T,3},
    bounds::NTuple{2,<:AbstractVector{T}},
    σ::Union{AbstractArray{T},T},
) where {T<:AbstractFloat}
    @views begin
        _erf!(psfcomponents[1], relativepos[1], tracksᵥ[:, 1:1, :], bounds[1], σ)
        _erf!(psfcomponents[2], relativepos[2], tracksᵥ[:, 2:2, :], bounds[2], σ)
    end
    return psfcomponents
end

function addincident!(
    intensity::AbstractArray{T,3},
    psfcomponents::NTuple{2,<:AbstractArray{T,3}},
    relativepos::NTuple{2,<:AbstractArray{T,3}},
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    bounds::NTuple{2,<:AbstractVector{T}},
    psf::GaussianPSF{T},
) where {T<:AbstractFloat}
    integratepsf!(psfcomponents, relativepos, tracksᵥ, bounds, psf)
    return batched_mul!(
        intensity,
        psfcomponents[1],
        batched_transpose(psfcomponents[2]),
        brightnessᵥ / psf.A,
        1,
    )
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
    nframes, ~, nemitters = size(tracksᵥ)
    psfcomponents = (
        similar(tracksᵥ, size(detector, 1), nemitters, nframes),
        similar(tracksᵥ, size(detector, 2), nemitters, nframes),
    )
    relativepos = (
        similar(tracksᵥ, size(detector, 1) + 1, nemitters, nframes),
        similar(tracksᵥ, size(detector, 2) + 1, nemitters, nframes),
    )
    addincident!(
        llarray.means[i],
        psfcomponents,
        relativepos,
        tracksᵥ,
        brightnessᵥ,
        detector.pxbounds,
        psf,
    )
    return llarray
end

function set_poisson_means!(
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
    tracksᵥ₁::AbstractArray{T,3},
    tracksᵥ₂::AbstractArray{T,3},
    brightnessᵥ::T,
    psf::PointSpreadFunction{T},
) where {T}
    set_poisson_mean!(llarray, detector, tracksᵥ₁, brightnessᵥ, psf, 1)
    set_poisson_mean!(llarray, detector, tracksᵥ₂, brightnessᵥ, psf, 2)
    return llarray
end

function set_poisson_means!(
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ₁::T,
    brightnessᵥ₂::T,
    psf::PointSpreadFunction{T},
) where {T}
    set_poisson_mean!(llarray, detector, tracksᵥ, brightnessᵥ₁, psf, 1)
    llarray.means[2] .=
        (llarray.means[1] .- detector.darkcounts) .* (brightnessᵥ₂ / brightnessᵥ₁)
    return llarray
end

function getincident(
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    darkcounts::AbstractMatrix{T},
    bounds::NTuple{2,<:AbstractVector{T}},
    psf::PointSpreadFunction{T},
) where {T}
    nframes, ~, nemitters = size(tracksᵥ)
    psfcomponents = (
        similar(tracksᵥ, size(darkcounts, 1), nemitters, nframes),
        similar(tracksᵥ, size(darkcounts, 2), nemitters, nframes),
    )
    relativepos = (
        similar(tracksᵥ, length(bounds[1]), nemitters, nframes),
        similar(tracksᵥ, length(bounds[2]), nemitters, nframes),
    )
    incident = repeat(darkcounts, 1, 1, nframes)
    return addincident!(
        incident,
        psfcomponents,
        relativepos,
        tracksᵥ,
        brightnessᵥ,
        bounds,
        psf,
    )
end