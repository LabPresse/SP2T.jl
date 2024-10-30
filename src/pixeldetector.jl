# struct PixelDetectorAuxiliary{
#     T<:AbstractFloat,
#     A<:AbstractArray{T,3},
#     V<:AbstractVector{T},
# } <: AuxiliaryVariables{T}
#     intensity₁::A
#     intensity₂::A
#     pxlogℒ::A # scratch array
#     framelogℒ::V # scratch vector
# end

# function PixelDetectorAuxiliary(nframes::Integer, detector::Detector{T}) where {T}
#     intensity₁ = repeat(detector.darkcounts, 1, 1, nframes)
#     intensity₂ = copy(intensity₁)
#     pxlogℒ = fill!(similar(intensity₁), NaN)
#     framelogℒ = fill!(similar(intensity₁, nframes), NaN)
#     return PixelDetectorAuxiliary(intensity₁, intensity₂, pxlogℒ, framelogℒ)
# end

# detectoraux(measurements::AbstractArray, detector::PixelDetector{T}) where {T} =
#     PixelDetectorAuxiliary(size(measurements, 3), detector)

struct SPAD{
    T<:AbstractFloat,
    V<:AbstractVector{T},
    M<:AbstractMatrix{T},
    A<:AbstractArray{T,3},
    MB<:AbstractMatrix{Bool},
    # AV<:PixelDetectorAuxiliary{T},
} <: PixelDetector{T}
    batchsize::UInt16
    period::T
    pxsize::T
    pxboundsx::V
    pxboundsy::V
    darkcounts::M
    filter::MB
    intensity₁::A
    intensity₂::A
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
    intensity₁ = repeat(darkcounts, 1, 1, nframes)
    intensity₂ = copy(intensity₁)
    pxlogℒ = fill!(similar(intensity₁), NaN)
    framelogℒ = fill!(similar(intensity₁, nframes), NaN)
    return SPAD(
        batchsize,
        period,
        pxsize,
        pxboundsx,
        pxboundsy,
        darkcounts,
        filter,
        intensity₁,
        intensity₂,
        pxlogℒ,
        framelogℒ,
    )
end

function initintensity!(detector::PixelDetector, both::Bool = false)
    detector.intensity₁ .= detector.darkcounts
    both && (detector.intensity₂ .= detector.darkcounts)
    return detector
end

pxlogℒ!(
    logℒ::AbstractArray{T,3},
    measurements::AbstractArray{UInt16,3},
    intensity::AbstractArray{T,3},
    batchsize::UInt16,
) where {T} = @. logℒ = measurements * logexpm1(intensity) - batchsize * intensity

pxlogℒ!(detector::SPAD, measurements::AbstractArray{UInt16,3}) =
    pxlogℒ!(detector.pxlogℒ, measurements, detector.intensity₁, detector.batchsize)

Δpxlogℒ!(
    Δlogℒ::AbstractArray{T,3},
    measurements::AbstractArray{UInt16,3},
    intensity₁::AbstractArray{T,3},
    intensity₂::AbstractArray{T,3},
    batchsize::UInt16,
) where {T} = @. Δlogℒ =
    measurements * (logexpm1(intensity₂) - logexpm1(intensity₁)) -
    batchsize * (intensity₂ - intensity₁)

Δpxlogℒ!(detector::SPAD, measurements::AbstractArray{UInt16,3}) = Δpxlogℒ!(
    detector.pxlogℒ,
    measurements,
    detector.intensity₁,
    detector.intensity₂,
    detector.batchsize,
)

framesum!(
    framelogℒ::AbstractVector{T},
    pxlogℒ::AbstractArray{T,3},
    filter::AbstractMatrix{Bool},
) where {T} = mul!(framelogℒ, transpose(reshape(pxlogℒ, length(filter), :)), vec(filter))

framesum!(detector::PixelDetector) =
    framesum!(detector.framelogℒ, detector.pxlogℒ, detector.filter)

function logℒ!(
    detector::PixelDetector{T},
    measurements::AbstractArray{<:Union{T,Integer},3},
) where {T}
    pxlogℒ!(detector, measurements)
    framesum!(detector)
    return sum(detector.framelogℒ)
end

function Δlogℒ!(
    detector::PixelDetector{T},
    measurements::AbstractArray{<:Union{T,Integer},3},
) where {T}
    Δpxlogℒ!(detector, measurements)
    return framesum!(detector)
end

function add_pxcounts!(
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

# function pxcounts!(
#     intensity::AbstractArray{T,3},
#     tracks::AbstractArray{T,3},
#     brightness::T,
#     darkcounts::AbstractMatrix{T},
#     xbnds::AbstractVector{T},
#     ybnds::AbstractVector{T},
#     psf::PointSpreadFunction{T},
# ) where {T}
#     intensity .= darkcounts
#     return add_pxcounts!(intensity, tracks, brightness, xbnds, ybnds, psf)
# end

# pxcounts!(
#     intensity::AbstractArray{T,3},
#     tracks::AbstractArray{T,3},
#     brightness::T,
#     detector::PixelBasedDetector{T},
#     psf::PointSpreadFunction{T},
# ) where {T} = pxcounts!(
#     intensity,
#     tracks,
#     brightness,
#     detector.darkcounts,
#     detector.pxboundsx,
#     detector.pxboundsy,
#     psf,
# )

function pxcounts!(
    detector::PixelDetector{T},
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    psf::PointSpreadFunction{T},
) where {T}
    initintensity!(detector)
    add_pxcounts!(
        detector.intensity₁,
        tracksᵥ,
        brightnessᵥ,
        detector.pxboundsx,
        detector.pxboundsy,
        psf,
    )
end

function pxcounts!(
    detector::PixelDetector{T},
    tracks₁::AbstractArray{T,3},
    tracks₂::AbstractArray{T,3},
    brightnessᵥ::T,
    psf::PointSpreadFunction{T},
) where {T}
    initintensity!(detector, true)
    add_pxcounts!(
        detector.intensity₁,
        tracks₁,
        brightnessᵥ,
        detector.pxboundsx,
        detector.pxboundsy,
        psf,
    )
    add_pxcounts!(
        detector.intensity₂,
        tracks₂,
        brightnessᵥ,
        detector.pxboundsx,
        detector.pxboundsy,
        psf,
    )
end

function pxcounts(
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    darkcounts::AbstractMatrix{T},
    xᵖ::AbstractVector{T},
    yᵖ::AbstractVector{T},
    psf::PointSpreadFunction{T},
) where {T}
    𝐔 = repeat(darkcounts, 1, 1, size(tracksᵥ, 1))
    return add_pxcounts!(𝐔, tracksᵥ, brightnessᵥ, xᵖ, yᵖ, psf)
end

# pxcounts(
#     tracks::AbstractArray{T,3},
#     brightness::T,
#     detector::PixelDetector{T},
#     psf::PointSpreadFunction{T},
# ) where {T} = pxcounts(
#     tracks,
#     brightness,
#     detector.darkcounts,
#     detector.pxboundsx,
#     detector.pxboundsy,
#     psf,
# )

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