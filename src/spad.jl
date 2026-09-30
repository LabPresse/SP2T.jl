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

# Keep the hit probability accurate near zero and at saturation.
_spad_log_hit(mean::T) where {T} =
    mean <= log(T(2)) ? log(-expm1(-mean)) : log1p(-exp(-mean))

function _spad_loglikelihood(readout, batchsize, mean::T) where {T}
    hits = iszero(readout) ? zero(T) : readout * _spad_log_hit(mean)
    misses = readout == batchsize ? zero(T) : (batchsize - readout) * mean
    return hits - misses
end

function _spad_Δloglikelihood(readout, batchsize, oldmean::T, newmean::T) where {T}
    hits = iszero(readout) ? zero(T) :
           readout * (_spad_log_hit(newmean) - _spad_log_hit(oldmean))
    misses = readout == batchsize ? zero(T) :
             (batchsize - readout) * (newmean - oldmean)
    return hits - misses
end

function set_pixel_loglikelihood!(
    llarray::LogLikelihoodArray{T},
    detector::SPAD{T},
) where {T}
    @. llarray.pixel = _spad_loglikelihood(
        detector.readouts,
        detector.batchsize,
        llarray.means[1],
    )
    return llarray
end

function set_pixel_Δloglikelihood!(
    Δllarray::LogLikelihoodArray{T},
    detector::SPAD{T},
) where {T}
    @. Δllarray.pixel = _spad_Δloglikelihood(
        detector.readouts,
        detector.batchsize,
        Δllarray.means[1],
        Δllarray.means[2],
    )
    return Δllarray
end

function simulate_readouts!(detector::SPAD{T}, means::AbstractArray{T,3}) where {T}
    if detector.batchsize == 1
        @. detector.readouts = $rand!($similar(means)) < -expm1(-means)
    else
        @. detector.readouts = rand(Binomial(detector.batchsize, -expm1(-means)))
    end
    return detector
end
