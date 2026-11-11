"""
    AbstractEMCCD{T} <: PixelDetector{T}

An abstract type representing an EMCCD detector. This serves as a 
base type for defining specific EMCCD implementations. The type parameter `T` can be used to specify the type of data.
"""
abstract type AbstractEMCCD{T} <: PixelDetector{T} end

struct EMCCD{
    T<:AbstractFloat,
    V<:AbstractVector{T},
    M<:AbstractMatrix{T},
    R<:AbstractArray{UInt16,3},
} <: AbstractEMCCD{T}
    period::T
    pxsize::T
    pxbounds::NTuple{2,V}
    darkcounts::M
    filter::M
    readouts::R
    offset::T
    gain::T
    variance::T
end

function EMCCD{T}(;
    period::Real,
    pixel_size::Real,
    darkcounts::AbstractMatrix{<:Real},
    cutoffs::Tuple{<:Real,<:Real},
    readouts::AbstractArray{UInt16,3},
    offset::Real,
    gain::Real,
    variance::Real,
) where {T<:AbstractFloat}
    check_pixel_dimsmatch(darkcounts, readouts)
    pxbounds, darkcounts, filter =
        _init_pixel_detector_params(T, pixel_size, darkcounts, cutoffs)
    return EMCCD{T,typeof(pxbounds[1]),typeof(darkcounts),typeof(readouts)}(
        period,
        pixel_size,
        pxbounds,
        darkcounts,
        filter,
        readouts,
        offset,
        gain,
        variance,
    )
end

function set_pixel_loglikelihood!(
    llarray::LogLikelihoodArray{T},
    detector::EMCCD{T},
) where {T}
    llarray.pixel .=
        (-log(detector.variance) / 2) .-
        (detector.readouts .- detector.gain .* llarray.means[1] .- detector.offset) .^ 2 ./
        (2 * detector.variance)
    return llarray
end

# Factor the residual-square difference before device arithmetic. This preserves
# exact identity increments despite device fused multiply-add contraction.
function _emccd_Δloglikelihood(
    readout,
    oldmean::T,
    newmean::T,
    gain::T,
    offset::T,
    variance::T,
) where {T}
    if oldmean == newmean && isfinite(oldmean) && isfinite(gain) &&
       isfinite(offset) && isfinite(variance) && variance > zero(T)
        return zero(T)
    end
    return gain * (newmean - oldmean) *
           (2 * (T(readout) - offset) - gain * (newmean + oldmean)) / (2 * variance)
end

function set_pixel_Δloglikelihood!(
    llarray::LogLikelihoodArray{T},
    detector::EMCCD{T},
) where {T}
    @. llarray.pixel = _emccd_Δloglikelihood(
        detector.readouts,
        llarray.means[1],
        llarray.means[2],
        detector.gain,
        detector.offset,
        detector.variance,
    )
    return llarray
end

function simulate_readouts!(
    rng::AbstractRNG,
    detector::EMCCD{T},
    means::AbstractArray{T,3},
) where {T}
    floatreadouts = similar(detector.readouts, T)
    randn!(devicerng(rng, floatreadouts), floatreadouts)
    floatreadouts .*= sqrt(detector.variance)
    @. floatreadouts += detector.gain * means + detector.offset
    detector.readouts .= round.(UInt16, clamp.(floatreadouts, 0, typemax(UInt16)))
    return detector
end

struct EMCCDGamma{
    T<:AbstractFloat,
    V<:AbstractVector{T},
    M<:AbstractMatrix{T},
    R<:AbstractArray{UInt16,3},
} <: AbstractEMCCD{T}
    period::T
    pxsize::T
    pxbounds::NTuple{2,V}
    darkcounts::M
    filter::M
    readouts::R
    gain::T
    noise_excess_factor::T
end

function EMCCDGamma{T}(;
    period::Real,
    pixel_size::Real,
    darkcounts::AbstractMatrix{<:Real},
    cutoffs::Tuple{<:Real,<:Real},
    readouts::AbstractArray{UInt16,3},
    gain::Real,
    noise_excess_factor::Real = 2,
) where {T<:AbstractFloat}
    check_pixel_dimsmatch(darkcounts, readouts)
    any(iszero, readouts) &&
        throw(ArgumentError("EMCCD Gamma does not support zero readouts"))
    pxbounds, darkcounts, filter =
        _init_pixel_detector_params(T, pixel_size, darkcounts, cutoffs)
    return EMCCDGamma{T,typeof(pxbounds[1]),typeof(darkcounts),typeof(readouts)}(
        period,
        pixel_size,
        pxbounds,
        darkcounts,
        filter,
        readouts,
        gain,
        noise_excess_factor,
    )
end

function _gamma_pixel(weight::T, readout, mean::T, excess::T, logscale::T) where {T}
    # Exclude before calling loggamma: a filtered negative calibration can have
    # a shape outside the real log-gamma domain.
    iszero(weight) && return zero(T)
    k = mean / excess
    return (k - one(T)) * log(T(readout)) - loggamma(k) - k * logscale
end

function _gamma_pixel_delta(weight::T, readout, old::T, new::T, excess::T, logscale::T) where {T}
    iszero(weight) && return zero(T)
    ko, kn = old / excess, new / excess
    return (kn - ko) * (log(T(readout)) - logscale) - (loggamma(kn) - loggamma(ko))
end

function set_pixel_loglikelihood!(llarray::LogLikelihoodArray{T}, detector::EMCCDGamma{T}) where {T}
    logscale = log(detector.noise_excess_factor * detector.gain)
    llarray.pixel .= _gamma_pixel.(detector.filter, detector.readouts, llarray.means[1],
                                  detector.noise_excess_factor, logscale)
    return llarray
end

function set_pixel_Δloglikelihood!(
    Δllarray::LogLikelihoodArray{T},
    detector::EMCCDGamma{T},
) where {T}
    logscale = log(detector.noise_excess_factor * detector.gain)
    Δllarray.pixel .= _gamma_pixel_delta.(detector.filter, detector.readouts,
        Δllarray.means[1], Δllarray.means[2], detector.noise_excess_factor, logscale)
    return Δllarray
end

function simulate_readouts!(
    rng::AbstractRNG,
    detector::EMCCDGamma{T},
    means::AbstractArray{T,3},
) where {T}
    floatreadouts =
        rand.(
            Ref(rng),
            Gamma.(
                means ./ detector.noise_excess_factor,
                detector.noise_excess_factor * detector.gain,
            )
        )
    detector.readouts .= round.(UInt16, clamp.(floatreadouts, 0, typemax(UInt16)))
    return detector
end
