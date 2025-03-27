struct CircularGaussian{T} <: GaussianPSF{T}
    A::T # maximum intensity possible in one pixel
    σ::T # [length] std of PSF along xy (image plane)
    CircularGaussian{T}(σ::Real, pxsize::Real) where {T<:AbstractFloat} =
        new{T}(gaussianpeak(σ, pxsize), σ)
end

function CircularGaussian{T}(;
    numerical_aperture::Real,
    refractive_index::Real,
    emission_wavelength::Real,
    pixels_size::Real,
) where {T<:AbstractFloat}
    σ, ~ = getσ₀z₀(numerical_aperture, refractive_index, emission_wavelength)
    return CircularGaussian{T}(σ, pixels_size)
end

struct CircularGaussianLorentzian{T} <: GaussianPSF{T}
    A::T # maximum intensity possible in one pixel
    σ₀::T # [length] std of PSF along xy (image plane)
    z₀::T # [length] std of PSF along z (optical axis)
    CircularGaussianLorentzian{T}(
        σ₀::Real,
        z₀::Real,
        pxsize::Real,
    ) where {T<:AbstractFloat} = new{T}(gaussianpeak(σ₀, pxsize), σ₀, z₀)
end

function CircularGaussianLorentzian{T}(;
    numerical_aperture::Real,
    refractive_index::Real,
    emission_wavelength::Real,
    pixels_size::Real,
) where {T<:AbstractFloat}
    σ₀, z₀ = getσ₀z₀(numerical_aperture, refractive_index, emission_wavelength)
    return CircularGaussianLorentzian{T}(σ₀, z₀, pixels_size)
end

function getσ₀z₀(na::Real, nᵣ::Real, λ::Real)
    α = semiangle(na, nᵣ)
    cos12α = √cos(α)
    cos32α, cos72α = cos12α^3, cos12α^7
    b = (7 * (1 - cos32α)) / (4 - 7 * cos32α + 3 * cos72α)
    a = λ / pi / nᵣ
    z₀ = a * b
    σ₀ = sqrt(a * z₀) / 2
    return σ₀, z₀
end

semiangle(na::T, nᵣ::T) where {T<:AbstractFloat} = asin(na / nᵣ)

function gaussianpeak(σ::Real, pxsize::Real)
    x = pxsize / 2 / (√2 * σ)
    return erf(-x, x)^2 / 4
end

lateral_std!(
    σ::AbstractArray{T,N},
    z::AbstractArray{T,N},
    PSF::CircularGaussianLorentzian{T},
) where {T,N} = @. σ = PSF.σ₀ * √(oneunit(T) + (z / PSF.z₀)^2)

function lateral_std(
    z::AbstractArray{T,3},
    PSF::CircularGaussianLorentzian{T},
) where {T<:AbstractFloat}
    z′ = PermutedDimsArray(z, (2, 3, 1))
    return lateral_std!(similar(z′), z′, PSF)
end

function _erf(
    x::AbstractArray{T,3},
    bnds::AbstractVector{T},
    σ::Union{AbstractArray{T},T},
) where {T}
    psfcomponents = similar(x, length(bnds) - 1, 1, size(x, 1))
    X = similar(x, length(bnds), 1, size(x, 1))
    return _erf!(psfcomponents, X,x, bnds, σ)
end

function _erf!(
    psfx::AbstractArray{T,3},
    𝐗::AbstractArray{T,3},
    x::AbstractArray{T,3},
    bnds::AbstractVector{T},
    σ::T,
) where {T}
    𝐗 .= (bnds .- PermutedDimsArray(x, (2, 3, 1))) ./ (√convert(T, 2) * σ)
    @views psfx .= erf.(𝐗[1:end-1, :, :], 𝐗[2:end, :, :]) ./ 2
    return psfx
end

function _erf!(
    psfx::AbstractArray{T,3},
    𝐗::AbstractArray{T,3},
    x::AbstractArray{T,3},
    bnds::AbstractVector{T},
    σ::AbstractArray{T},
) where {T}
    𝐗 .= (bnds .- PermutedDimsArray(x, (2, 3, 1))) ./ (√convert(T, 2) .* σ)
    @views psfx .= erf.(𝐗[1:end-1, :, :], 𝐗[2:end, :, :]) ./ 2
    return psfx
end