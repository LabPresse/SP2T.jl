function Base.getproperty(detector::PixelDetector, s::Symbol)
    if s === :framecenter
        return mean(detector.pxbounds[1]), mean(detector.pxbounds[2])
    else
        return getfield(detector, s)
    end
end

Base.size(detector::PixelDetector) = size(detector.darkcounts)

function reset!(
    loglikelihood::LogLikelihoodArray{T},
    detector::PixelDetector{T},
    i::Integer,
) where {T}
    loglikelihood.means[i] .= detector.darkcounts
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