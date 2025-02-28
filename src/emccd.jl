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
    loglikelihood::LogLikelihoodArray{T},
    detector::EMCCD{T},
) where {T}
    loglikelihood.pixel .=
        (-log(detector.variance) / 2) .-
        (detector.readouts .- detector.gain .* loglikelihood.means[1] .- detector.offset) .^
        2 ./ (2 * detector.variance)
    return loglikelihood
end

function set_pixel_Δloglikelihood!(
    Δloglikelihood::LogLikelihoodArray{T},
    detector::EMCCD{T},
) where {T}
    @. Δloglikelihood.pixel =
        (
            (
                detector.readouts - detector.gain * Δloglikelihood.means[1] -
                detector.offset
            )^2 -
            (
                detector.readouts - detector.gain * Δloglikelihood.means[2] -
                detector.offset
            )^2
        ) / (2 * detector.variance)
    return Δloglikelihood
end

function simulate_readouts!(detector::EMCCD{T}, means::AbstractArray{T,3}) where {T}
    floatreadouts = randn!(similar(detector.readouts, T))
    floatreadouts .*= sqrt(detector.variance)
    @. floatreadouts += detector.gain * means + detector.offset
    detector.readouts .= round.(UInt16, floatreadouts)
    return detector
end

struct EMCCDGamma{
    T<:AbstractFloat,
    V<:AbstractVector{T},
    M<:AbstractMatrix{T},
    R<:AbstractArray{UInt16,3},
} <: PixelDetector{T}
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

function set_pixel_loglikelihood!(
    llarray::LogLikelihoodArray{T},
    detector::EMCCDGamma{T},
) where {T}
    llarray.means[2] .= llarray.means[1] ./ detector.noise_excess_factor
    llarray.pixel .=
        (llarray.means[2] .- 1) .* log.(detector.readouts) .- loggamma.(llarray.means[2]) .-
        llarray.means[2] .* log(detector.noise_excess_factor * detector.gain)
    return llarray
end

function set_pixel_Δloglikelihood!(
    Δllarray::LogLikelihoodArray{T},
    detector::EMCCDGamma{T},
) where {T}
    Δllarray.means[1] ./= detector.noise_excess_factor
    Δllarray.means[2] ./= detector.noise_excess_factor
    @. Δllarray.pixel =
        (Δllarray.means[2] .- Δllarray.means[1]) .*
        (log.(detector.readouts) .- log(detector.noise_excess_factor * detector.gain)) .-
        loggamma.(Δllarray.means[2]) .+ loggamma.(Δllarray.means[1])
    return Δllarray
end

function simulate_readouts!(detector::EMCCDGamma{T}, means::AbstractArray{T,3}) where {T}
    floatreadouts =
        rand.(
            Gamma.(
                means ./ detector.noise_excess_factor,
                detector.noise_excess_factor * detector.gain,
            )
        )
    detector.readouts .= round.(UInt16, floatreadouts)
    return detector
end