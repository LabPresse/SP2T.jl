struct SPAD{
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
    batchsize::UInt16
end

function SPAD{T}(;
    period::Real,
    pixel_size::Real,
    darkcounts::AbstractMatrix{<:Real},
    cutoffs::Tuple{<:Real,<:Real},
    readouts::AbstractArray{UInt16,3},
    batchsize::Integer = one(UInt16),
) where {T<:AbstractFloat}
    check_pixel_dimsmatch(darkcounts, readouts)
    pxbounds, darkcounts, filter =
        _init_pixel_detector_params(T, pixel_size, darkcounts, cutoffs)
    return SPAD{T,typeof(pxbounds[1]),typeof(darkcounts),typeof(readouts)}(
        period,
        pixel_size,
        pxbounds,
        darkcounts,
        filter,
        readouts,
        batchsize,
    )
end

set_spad_loglikelihood!(
    l::AbstractArray{T,3},
    k::AbstractArray{UInt16,3},
    c::AbstractArray{T,3},
    n::UInt16,
) where {T} = @. l = k * logexpm1(c) - n * c

set_spad_Δloglikelihood!(
    Δ::AbstractArray{T,3},
    k::AbstractArray{UInt16,3},
    c1::AbstractArray{T,3},
    c2::AbstractArray{T,3},
    n::UInt16,
) where {T} = @. Δ = k * (logexpm1(c2) - logexpm1(c1)) - n * (c2 - c1)

function set_pixel_loglikelihood!(
    llarray::LogLikelihoodArray{T},
    detector::SPAD{T},
) where {T}
    set_spad_loglikelihood!(
        llarray.pixel,
        detector.readouts,
        llarray.means[1],
        detector.batchsize,
    )
    return llarray
end

function set_pixel_Δloglikelihood!(
    loglikelihood::LogLikelihoodArray{T},
    detector::SPAD{T},
) where {T}
    set_spad_Δloglikelihood!(
        loglikelihood.pixel,
        detector.readouts,
        loglikelihood.means[1],
        loglikelihood.means[2],
        detector.batchsize,
    )
    return loglikelihood
end

function simulate_readouts!(detector::SPAD{T}, means::AbstractArray{T,3}) where {T}
    if detector.batchsize == 1
        @. detector.readouts = $rand!($similar(means)) < -expm1(-means)
    else
        @. detector.readouts = rand(Binomial(detector.batchsize, -expm1(-means)))
    end
    return detector
end