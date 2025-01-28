struct EMCCD{
    T<:AbstractFloat,
    V<:AbstractVector{T},
    M<:AbstractMatrix{T},
    A<:AbstractArray{T,3},
} <: PixelDetector{T}
    period::T
    pxsize::T
    pxbounds::NTuple{2,V}
    darkcounts::M
    filter::M
    readouts::A
    offset::T
    gain::T
    variance::T
end

function EMCCD{T}(;
    period::Real,
    pixel_size::Real,
    darkcounts::AbstractMatrix{<:Real},
    cutoffs::Tuple{<:Real,<:Real},
    readouts::AbstractArray{<:Real,3},
    offset::Real,
    gain::Real,
    variance::Real,
) where {T<:AbstractFloat}
    period = convert(T, period)
    pixel_size = convert(T, pixel_size)
    darkcounts = elconvert(T, darkcounts)
    readouts = elconvert(T, readouts)
    filter = similar(darkcounts)
    filter .= cutoffs[1] .< darkcounts .< cutoffs[2]
    width, height = size(darkcounts)
    pxboundsx = similar(darkcounts, width + 1)
    pxboundsx .= 0:pixel_size:width*pixel_size
    pxboundsy = similar(darkcounts, height + 1)
    pxboundsy .= 0:pixel_size:height*pixel_size
    return EMCCD{T,typeof(pxboundsx),typeof(darkcounts),typeof(readouts)}(
        period,
        pixel_size,
        (pxboundsx, pxboundsy),
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
    x::AbstractArray{T,3},
    c::AbstractArray{T,3},
    g::Union{T,AbstractMatrix{T}},
    μ::Union{T,AbstractMatrix{T}},
    v::Union{T,AbstractMatrix{T}},
) where {T} = @. l = -log(v) / 2 - (x - g * c - μ)^2 / (2 * v)

set_emccd_Δloglikelihood!(
    Δ::AbstractArray{T,3},
    x::AbstractArray{T,3},
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
    randn!(detector.readouts)
    detector.readouts .*= sqrt(detector.variance)
    @. detector.readouts += detector.gain * means + detector.offset
    return detector
end