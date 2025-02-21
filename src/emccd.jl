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

set_emccd_loglikelihood!(
    l::AbstractArray{T,3},
    x::AbstractArray{UInt16,3},
    c::AbstractArray{T,3},
    g::T,
    μ::T,
    v::T,
) where {T} = l .= (-log(v) / 2) .- (x .- g .* c .- μ) .^ 2 ./ (2 * v)

set_emccd_Δloglikelihood!(
    Δ::AbstractArray{T,3},
    x::AbstractArray{UInt16,3},
    c1::AbstractArray{T,3},
    c2::AbstractArray{T,3},
    g::Union{T,AbstractMatrix{T}},
    μ::Union{T,AbstractMatrix{T}},
    v::Union{T,AbstractMatrix{T}},
) where {T} = @. Δ = ((x - g * c1 - μ)^2 - (x - g * c2 - μ)^2) / (2 * v)

function set_pixel_loglikelihood!(
    loglikelihood::LogLikelihoodArray{T},
    detector::EMCCD{T},
) where {T}
    set_emccd_loglikelihood!(
        loglikelihood.pixel,
        detector.readouts,
        loglikelihood.means[1],
        detector.gain,
        detector.offset,
        detector.variance,
    )
    return loglikelihood
end

function set_pixel_Δloglikelihood!(
    loglikelihood::LogLikelihoodArray{T},
    detector::EMCCD{T},
) where {T}
    set_emccd_Δloglikelihood!(
        loglikelihood.pixel,
        detector.readouts,
        loglikelihood.means[1],
        loglikelihood.means[2],
        detector.gain,
        detector.offset,
        detector.variance,
    )
    return loglikelihood
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

set_emccdγ_loglikelihood!(
    l::AbstractArray{T,3},
    x::AbstractArray{UInt16,3},
    c::AbstractArray{T,3},
    f::T,
) where {T} = l .= (c ./ f .- 1) .* log.(x)

set_emccdγ_Δloglikelihood!(
    Δ::AbstractArray{T,3},
    x::AbstractArray{UInt16,3},
    c1::AbstractArray{T,3},
    c2::AbstractArray{T,3},
    f::T,
) where {T} = @. Δ = (c2 .- c1) ./ f .* log.(x)

function set_pixel_loglikelihood!(
    loglikelihood::LogLikelihoodArray{T},
    detector::EMCCDGamma{T},
) where {T}
    set_emccdγ_loglikelihood!(
        loglikelihood.pixel,
        detector.readouts,
        loglikelihood.means[1],
        detector.noise_excess_factor,
    )
    return loglikelihood
end

function set_pixel_Δloglikelihood!(
    loglikelihood::LogLikelihoodArray{T},
    detector::EMCCDGamma{T},
) where {T}
    set_emccdγ_Δloglikelihood!(
        loglikelihood.pixel,
        detector.readouts,
        loglikelihood.means[1],
        loglikelihood.means[2],
        detector.noise_excess_factor,
    )
    return loglikelihood
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