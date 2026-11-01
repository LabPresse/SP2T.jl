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

# Most SPAD readouts are 0, so `logexpm1` is evaluated only where it contributes: a real
# branch (not `ifelse`) is 5× faster on CPU and up to 3.5× on Metal.
clickterm(r::Integer, λ::T) where {T} = iszero(r) ? zero(T) : r * logexpm1(λ)
Δclickterm(r::Integer, λ₁::T, λ₂::T) where {T} =
    iszero(r) ? zero(T) : r * (logexpm1(λ₂) - logexpm1(λ₁))

function set_pixel_loglikelihood!(
    llarray::LogLikelihoodArray{T},
    detector::SPAD{T},
) where {T}
    @. llarray.pixel =
        clickterm(detector.readouts, llarray.means[1]) -
        detector.batchsize * llarray.means[1]
    return llarray
end

function set_pixel_Δloglikelihood!(
    Δllarray::LogLikelihoodArray{T},
    detector::SPAD{T},
) where {T}
    @. Δllarray.pixel =
        Δclickterm(detector.readouts, Δllarray.means[1], Δllarray.means[2]) -
        detector.batchsize * (Δllarray.means[2] - Δllarray.means[1])
    return Δllarray
end

function simulate_readouts!(
    rng::AbstractRNG,
    detector::SPAD{T},
    means::AbstractArray{T,3},
) where {T}
    if detector.batchsize == 1
        @. detector.readouts = $rand!($devicerng(rng, means), $similar(means)) < -expm1(-means)
    else
        detector.readouts .= rand.(Ref(rng), Binomial.(detector.batchsize, -expm1.(-means)))
    end
    return detector
end