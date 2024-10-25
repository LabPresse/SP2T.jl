abstract type PointSpreadFunction{T} end

struct CircularGaussianLorentzian{T} <: PointSpreadFunction{T}
    A::T # maximum intensity possible in one pixel
    σ₀::T # [length] std of PSF along xy (image plane)
    z₀::T # [length] std of PSF along z (optical axis)
end

function CircularGaussianLorentzian{T}(
    na::Real,
    nᵣ::Real,
    λ::Real,
    pxsize::Real,
) where {T<:AbstractFloat}
    σ₀, z₀ = getσ₀z₀(na, nᵣ, λ)
    A = peakintensityCGL(σ₀, pxsize)
    return CircularGaussianLorentzian{T}(A, σ₀, z₀)
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

function peakintensityCGL(σ₀::Real, pxsize::Real)
    x = pxsize / 2 / (√2 * σ₀)
    return erf(-x, x)^2 / 4
end

lateral_std!(
    σ::AbstractArray{T,N},
    z::AbstractArray{T,N},
    PSF::CircularGaussianLorentzian{T},
) where {T,N} = @. σ = PSF.σ₀ * √(oneunit(T) + (z / PSF.z₀)^2)

function lateral_std(
    z::AbstractArray{T},
    PSF::CircularGaussianLorentzian{T},
) where {T<:AbstractFloat}
    z′ = PermutedDimsArray(z, (2, 3, 1))
    return lateral_std!(similar(z′), z′, PSF)
end

function _erf(x::AbstractArray{T}, bnds::AbstractVector{T}, σ::AbstractArray{T}) where {T}
    𝐗 = @. (bnds - $PermutedDimsArray(x, (2, 3, 1))) / (√convert(T, 2) * σ)
    return @views erf.(𝐗[1:end-1, :, :], 𝐗[2:end, :, :]) ./ 2
end