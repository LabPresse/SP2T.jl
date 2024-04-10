abstract type AbstractPSF{T} end

struct CircularGaussianLorentzian{FT<:AbstractFloat} <: AbstractPSF{FT}
    z₀::FT # [length] std of PSF along z (optical axis)
    σ₀::FT # [length] std of PSF along xy (image plane)
    σ₀_sqrt2::FT # σ₀√2
end

function CircularGaussianLorentzian{FT}(;
    NA::Real,
    nᵣ::Real,
    λ::Real,
) where {FT<:AbstractFloat}
    a = λ / pi / nᵣ
    b = getratio(NA, nᵣ)
    z₀ = a * b
    σ₀ = sqrt(a * z₀) / 2
    return CircularGaussianLorentzian{FT}(z₀, σ₀, sqrt(2) * σ₀)
end

function getratio(NA::Real, nᵣ::Real)
    α = getsemiangle(NA, nᵣ)
    cos12α = sqrt(cos(α))
    cos32α, cos72α = cos12α^3, cos12α^7
    return ((7 * (1 - cos32α)) / (4 - 7 * cos32α + 3 * cos72α))
end

getsemiangle(NA::Real, nᵣ::Real) = asin(NA / nᵣ)

get_σ_sqrt2(
    z::AbstractArray{FT,3},
    PSF::CircularGaussianLorentzian{FT},
) where {FT<:AbstractFloat} = @. PSF.σ₀_sqrt2 * √(1 + (z / PSF.z₀)^2)

function geterf(
    x::AbstractArray{FT},
    xᵖ::AbstractArray{FT},
    σ_sqrt2::AbstractArray{FT},
) where {FT<:AbstractFloat}
    𝐗 = (xᵖ .- x) ./ σ_sqrt2
    return @views erf.(𝐗[1:end-1, :, :], 𝐗[2:end, :, :]) ./ 2
end

function add_px_intensity!(
    𝐔::AbstractArray{FT,3},
    x::AbstractArray{FT,3},
    xᵖ::AbstractVector{FT},
    yᵖ::AbstractVector{FT},
    PSF::AbstractPSF{FT},
    hτ::FT,
    β::Integer = 1,
) where {FT<:AbstractFloat}
    σ_sqrt2 = get_σ_sqrt2(view(x, 3:3, :, :), PSF)
    𝐗 = geterf(view(x, 1:1, :, :), xᵖ, σ_sqrt2)
    𝐘 = geterf(view(x, 2:2, :, :), yᵖ, σ_sqrt2)
    return batched_mul!(𝐔, 𝐗, batched_transpose(𝐘), hτ, β)
end

function get_px_PSF(
    x::AbstractArray{FT,3},
    xᵖ::AbstractVector{FT},
    yᵖ::AbstractVector{FT},
    PSF::AbstractPSF{FT},
) where {FT<:AbstractFloat}
    𝐔 = similar(x, length(xᵖ) - 1, length(yᵖ) - 1, size(x, 3))
    add_px_intensity!(𝐔, x, xᵖ, yᵖ, PSF, one(FT), 0)
    return 𝐔
end

function get_px_intensity(
    x::AbstractArray{FT,3},
    xᵖ::AbstractVector{FT},
    yᵖ::AbstractVector{FT},
    hτ::FT,
    𝐅::AbstractMatrix{FT},
    PSF::AbstractPSF{FT},
) where {FT<:AbstractFloat}
    𝐔 = repeat(𝐅, 1, 1, size(x, 3))
    add_px_intensity!(𝐔, x, xᵖ, yᵖ, PSF, hτ)
    return 𝐔
end

_getframes(𝐔::AbstractArray{<:Real}) = rand(eltype(𝐔), size(𝐔)) .< -expm1.(-𝐔)