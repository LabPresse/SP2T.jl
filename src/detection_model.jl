abstract type AbstractPSF{T} end

struct CircularGaussianLorentzian{T} <: AbstractPSF{T}
    σ₀::T # [length] std of PSF along xy (image plane)
    z₀::T # [length] std of PSF along z (optical axis)
end

function CircularGaussianLorentzian{FT}(;
    NA::Real,
    nᵣ::Real,
    λ::Real,
) where {FT<:AbstractFloat}
    a = λ / pi / nᵣ
    b = _b(NA, nᵣ)
    z₀ = a * b
    σ₀ = sqrt(a * z₀) / 2
    return CircularGaussianLorentzian{FT}(σ₀, z₀)
end

function _b(NA, nᵣ)
    α = semiangle(NA, nᵣ)
    cos12α = sqrt(cos(α))
    cos32α, cos72α = cos12α^3, cos12α^7
    return ((7 * (1 - cos32α)) / (4 - 7 * cos32α + 3 * cos72α))
end

semiangle(NA, nᵣ) = asin(NA / nᵣ)

_σ(z, PSF::CircularGaussianLorentzian) = @. √2 * PSF.σ₀ * √(1 + (z / PSF.z₀)^2)

function _erf(x, bnds, σ)
    𝐗 = @. (bnds - x) / (√2 * σ)
    return @views erf.(𝐗[1:end-1, :, :], 𝐗[2:end, :, :]) ./ 2
end

struct ExperimentalParameters{Ts,Tv,Tm,Tp}
    period::Ts
    pxboundsx::Tv
    pxboundsy::Tv
    darkcounts::Tm
    PSF::Tp
end

function ExperimentalParameters(
    T::DataType,
    period::Real,
    pxsize::Real,
    darkcounts::Matrix{<:Real},
    NA::Real,
    refractiveindex::Real,
    wavelength::Real,
)
    period, pxsize, NA, refractiveindex, wavelength =
        convert.(T, (period, pxsize, NA, refractiveindex, wavelength))
    @show typeof(darkcounts)
    return ExperimentalParameters(
        period,
        range(0, step = pxsize, length = size(darkcounts, 1) + 1),
        range(0, step = pxsize, length = size(darkcounts, 2) + 1),
        convert(Matrix{T}, darkcounts),
        CircularGaussianLorentzian{T}(NA = NA, nᵣ = refractiveindex, λ = wavelength),
    )
end

function ExperimentalParameters(
    T::DataType,
    period::Real,
    pxsize::Real,
    darkcounts::Matrix{<:Real},
    σ₀::Real,
    z₀::Real,
)
    period, pxsize, σ₀, z₀ = convert.(T, (period, pxsize, σ₀, z₀))
    return ExperimentalParameters(
        period,
        range(0, step = pxsize, length = size(darkcounts, 1) + 1),
        range(0, step = pxsize, length = size(darkcounts, 2) + 1),
        convert(Matrix{T}, darkcounts),
        CircularGaussianLorentzian(σ₀, z₀),
    )
end

framecenter(params::ExperimentalParameters) = [
    (params.pxboundsx[1] + params.pxboundsx[end]) / 2,
    (params.pxboundsy[1] + params.pxboundsy[end]) / 2,
    0,
]

# _eltype(::ExperimentalParameters{S,T1,T2}) where {S,T1,T2} = S

pxsize(params::ExperimentalParameters) = params.pxboundsx[2] - params.pxboundsx[1]

to_cpu(params::ExperimentalParameters) = ExperimentalParameters(
    params.period,
    Array(params.pxboundsx),
    Array(params.pxboundsy),
    Array(params.darkcounts),
    params.PSF,
)
# function add_px_intensity!(
#     𝐔::AbstractArray{T,3},
#     x::AbstractArray{T,3},
#     h::T,
#     params::ExperimentalParameter,
#     β::Integer = 1,
# ) where {T}
#     σ = getσ(view(x, 3:3, :, :), params.PSF)
#     𝐗 = geterf(view(x, 1:1, :, :), params.pxboundsx, σ)
#     𝐘 = geterf(view(x, 2:2, :, :), params.pxboundsy, σ)
#     return batched_mul!(𝐔, 𝐗, batched_transpose(𝐘), h, β)
# end

function add_pxcounts!(
    𝐔::AbstractArray{T,3},
    x::AbstractArray{T,3},
    h::T,
    xᵖ::AbstractVector{T},
    yᵖ::AbstractVector{T},
    PSF::AbstractPSF{T},
    β = 1,
) where {T<:AbstractFloat}
    σ = _σ(view(x, 3:3, :, :), PSF)
    𝐗 = _erf(view(x, 1:1, :, :), xᵖ, σ)
    𝐘 = _erf(view(x, 2:2, :, :), yᵖ, σ)
    return batched_mul!(𝐔, 𝐗, batched_transpose(𝐘), h, β)
end

add_pxcounts!(𝐔, x, h, params::ExperimentalParameters) =
    add_pxcounts!(𝐔, x, h, params.pxboundsx, params.pxboundsy, params.PSF)

function get_pxPSF(
    x::AbstractArray{T,3},
    xᵖ::AbstractVector{T},
    yᵖ::AbstractVector{T},
    PSF::AbstractPSF{T},
) where {T}
    𝐔 = similar(x, length(xᵖ) - 1, length(yᵖ) - 1, size(x, 3))
    return add_pxcounts!(𝐔, x, oneunit(T), xᵖ, yᵖ, PSF, 0)
end

get_pxPSF(x, params::ExperimentalParameters) =
    get_pxPSF(x, params.pxboundsx, params.pxboundsy, params.PSF)

# function get_px_intensity!(
#     𝐔::AbstractArray{FT,3},
#     x::AbstractArray{FT,3},
#     h::FT,
#     params::ExperimentalParameter{FT},
# ) where {FT<:AbstractFloat}
#     𝐔 .= params.darkcounts
#     return add_px_intensity!(𝐔, x, h, params)
# end

# function get_px_intensity!(
#     𝐔::AbstractArray{T,3},
#     x::AbstractArray{T,3},
#     expparams::ExperimentalParameter,
#     brightness::Brightness,
# ) where {T<:AbstractFloat}
#     𝐔 .= expparams.darkcounts
#     return add_px_intensity!(
#         𝐔,
#         x,
#         expparams.pxboundsx,
#         expparams.pxboundsy,
#         expparams.PSF,
#         brightness.h,
#     )
# end

function pxcounts!(
    U::AbstractArray,
    x::AbstractArray,
    h,
    F::AbstractMatrix,
    xbnds::AbstractVector,
    ybnds::AbstractVector,
    PSF::AbstractPSF,
)
    U .= F
    return add_pxcounts!(U, x, h, xbnds, ybnds, PSF)
end

pxcounts!(U, x, h, params::ExperimentalParameters) =
    pxcounts!(U, x, h, params.darkcounts, params.pxboundsx, params.pxboundsy, params.PSF)

function pxcounts(
    x::AbstractArray{T,3},
    h::T,
    𝐅::AbstractMatrix{T},
    xᵖ::AbstractVector{T},
    yᵖ::AbstractVector{T},
    PSF::AbstractPSF{T},
) where {T<:AbstractFloat}
    𝐔 = repeat(𝐅, 1, 1, size(x, 3))
    return add_pxcounts!(𝐔, x, h, xᵖ, yᵖ, PSF)
end

pxcounts(x, h, params::ExperimentalParameters) =
    pxcounts(x, h, params.darkcounts, params.pxboundsx, params.pxboundsy, params.PSF)

simframes(𝐔) = rand(eltype(𝐔), size(𝐔)) .< -expm1.(-𝐔)