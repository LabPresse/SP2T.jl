abstract type PixelDetector{T} <: Detector{T} end
struct SPAD{
    T<:AbstractFloat,
    V<:AbstractVector{T},
    M<:AbstractMatrix{T},
    A<:AbstractArray{T,3},
    A2<:AbstractArray{T,4},
    MB<:AbstractMatrix{Bool},
} <: PixelDetector{T}
    batchsize::UInt16
    period::T
    pxsize::T
    pxboundsx::V
    pxboundsy::V
    darkcounts::M
    filter::MB
    fullintensity::A2
    pxlogℒ::A
    framelogℒ::V
end

function SPAD{T}(
    period::Real,
    pxsize::Real,
    darkcounts::AbstractMatrix{<:Real},
    cutoffs::Tuple{<:Real,<:Real},
    nframes::Integer,
    batchsize::Integer = one(UInt16),
) where {T<:AbstractFloat}
    period = convert(T, period)
    pxsize = convert(T, pxsize)
    darkcounts = convert(unionalltypeof(darkcounts){T}, darkcounts)
    filter = similar(darkcounts, Bool)
    filter .= cutoffs[1] .< darkcounts .< cutoffs[2]
    width, height = size(darkcounts)
    pxboundsx = similar(darkcounts, width + 1)
    pxboundsx .= 0:pxsize:width*pxsize
    pxboundsy = similar(darkcounts, height + 1)
    pxboundsy .= 0:pxsize:height*pxsize
    fullintensity = repeat(darkcounts, 1, 1, nframes, 2)
    pxlogℒ = fill!(similar(fullintensity, width, height, nframes), NaN)
    framelogℒ = fill!(similar(fullintensity, nframes), NaN)
    return SPAD(
        batchsize,
        period,
        pxsize,
        pxboundsx,
        pxboundsy,
        darkcounts,
        filter,
        fullintensity,
        pxlogℒ,
        framelogℒ,
    )
end

function Base.getproperty(detector::SPAD, s::Symbol)
    if s === :framecenter
        return mean(detector.pxboundsx), mean(detector.pxboundsy)
    elseif s === :intensity
        return selectdim(getfield(detector, :fullintensity), 4, 1)
    else
        return getfield(detector, s)
    end
end

Base.size(detector::SPAD) = size(detector.darkcounts)

function reset!(detector::PixelDetector, i::Integer)
    detector.fullintensity[:, :, :, i] .= detector.darkcounts
    return detector
end

pxlogℒ!(
    logℒ::AbstractArray{T,3},
    measurements::AbstractArray{UInt16,3},
    intensity::AbstractArray{T,3},
    batchsize::UInt16,
) where {T} = @. logℒ = measurements * logexpm1(intensity) - batchsize * intensity

Δpxlogℒ!(
    Δlogℒ::AbstractArray{T,3},
    measurements::AbstractArray{UInt16,3},
    fullintensity::AbstractArray{T,4},
    batchsize::UInt16,
) where {T} = @views @. Δlogℒ =
    measurements *
    (logexpm1(fullintensity[:, :, :, 2]) - logexpm1(fullintensity[:, :, :, 1])) -
    batchsize * (fullintensity[:, :, :, 2] - fullintensity[:, :, :, 1])

framesum!(
    framelogℒ::AbstractVector{T},
    pxlogℒ::AbstractArray{T,3},
    filter::AbstractMatrix{Bool},
) where {T} = mul!(framelogℒ, transpose(reshape(pxlogℒ, length(filter), :)), vec(filter))

function logℒ!(
    detector::PixelDetector{T},
    measurements::AbstractArray{<:Union{T,Integer},3},
) where {T}
    pxlogℒ!(detector.pxlogℒ, measurements, detector.intensity, detector.batchsize)
    framesum!(detector.framelogℒ, detector.pxlogℒ, detector.filter)
    return sum(detector.framelogℒ)
end

function Δlogℒ!(
    detector::PixelDetector{T},
    measurements::AbstractArray{<:Union{T,Integer},3},
) where {T}
    Δpxlogℒ!(detector.pxlogℒ, measurements, detector.fullintensity, detector.batchsize)
    return framesum!(detector.framelogℒ, detector.pxlogℒ, detector.filter)
end

function addincident!(
    intensity::AbstractArray{T,3},
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    xᵖ::AbstractVector{T},
    yᵖ::AbstractVector{T},
    psf::CircularGaussian{T},
    β = 1,
) where {T<:AbstractFloat}
    @views begin
        𝐗 = _erf(tracksᵥ[:, 1:1, :], xᵖ, psf.σ)
        𝐘 = _erf(tracksᵥ[:, 2:2, :], yᵖ, psf.σ)
    end
    return batched_mul!(intensity, 𝐗, batched_transpose(𝐘), brightnessᵥ / psf.A, β)
end

function addincident!(
    intensity::AbstractArray{T,3},
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    xᵖ::AbstractVector{T},
    yᵖ::AbstractVector{T},
    psf::CircularGaussianLorentzian{T},
    β = 1,
) where {T<:AbstractFloat}
    @views begin
        σ = lateral_std(tracksᵥ[:, 3:3, :], psf)
        𝐗 = _erf(tracksᵥ[:, 1:1, :], xᵖ, σ)
        𝐘 = _erf(tracksᵥ[:, 2:2, :], yᵖ, σ)
    end
    return batched_mul!(intensity, 𝐗, batched_transpose(𝐘), brightnessᵥ / psf.A, β)
end

function pxcounts!(
    detector::PixelDetector{T},
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    psf::PointSpreadFunction{T},
    i::Integer = 1,
) where {T}
    reset!(detector, i)
    addincident!(
        selectdim(detector.fullintensity, 4, i),
        tracksᵥ,
        brightnessᵥ,
        detector.pxboundsx,
        detector.pxboundsy,
        psf,
    )
    return detector
end

function pxcounts!(
    detector::PixelDetector{T},
    tracksᵥ₁::AbstractArray{T,3},
    tracksᵥ₂::AbstractArray{T,3},
    brightnessᵥ::T,
    psf::PointSpreadFunction{T},
) where {T}
    pxcounts!(detector, tracksᵥ₁, brightnessᵥ, psf, 1)
    pxcounts!(detector, tracksᵥ₂, brightnessᵥ, psf, 2)
    return detector
end

function pxcounts!(
    detector::PixelDetector{T},
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ₁::T,
    brightnessᵥ₂::T,
    psf::PointSpreadFunction{T},
) where {T}
    #TODO optimize!
    pxcounts!(detector, tracksᵥ, brightnessᵥ₁, psf, 1)
    pxcounts!(detector, tracksᵥ, brightnessᵥ₂, psf, 2)
    return detector
end

function getincident(
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    darkcounts::AbstractMatrix{T},
    xᵖ::AbstractVector{T},
    yᵖ::AbstractVector{T},
    psf::PointSpreadFunction{T},
) where {T}
    𝐔 = repeat(darkcounts, 1, 1, size(tracksᵥ, 1))
    return addincident!(𝐔, tracksᵥ, brightnessᵥ, xᵖ, yᵖ, psf)
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