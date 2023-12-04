abstract type AbstractPSF{T} end

struct CircularGaussianLorenzian{FT<:AbstractFloat} <: AbstractPSF{FT}
    z_ref::FT # [length] std of psf along z  (optical axis)
    σ_ref::FT # [length] std of psf along xy (image plane)
    σ_ref_sqrt2::FT # [length] std of psf along xy (image plane)
    function CircularGaussianLorenzian{FT}(
        NA::Real,
        nᵣ::Real,
        λ::Real,
    ) where {FT<:AbstractFloat}
        cos12α = sqrt(cos(asin(NA / nᵣ)))
        cos32α = cos12α^3
        cos72α = cos12α^7
        a = λ / pi / nᵣ
        b = ((7 * (1 - cos32α)) / (4 - 7 * cos32α + 3 * cos72α))
        z_ref = a * b
        σ_ref = sqrt(a * z_ref) / 2
        return new{FT}(z_ref, σ_ref, sqrt(2) * σ_ref)
    end
    # CircularGaussianLorenzian(
    #     z_ref::FT,
    #     σ_ref::FT,
    #     σ_ref_sqrt2::FT,
    # ) where {FT<:AbstractFloat} = CircularGaussianLorenzian{FT}(z_ref, σ_ref, σ_ref_sqrt2)
end

get_σ_sqrt2(
    z::AbstractArray{FT,3},
    PSF::CircularGaussianLorenzian{FT},
) where {FT<:AbstractFloat} = @. PSF.σ_ref_sqrt2 * √(1 + (z / PSF.z_ref)^2)

function get_erf(
    x::AbstractArray{FT},
    xᵖ::AbstractArray{FT},
    σ::AbstractArray{FT},
) where {FT<:AbstractFloat}
    𝐗 = (xᵖ .- x) ./ σ
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
    𝐗 = get_erf(view(x, 1:1, :, :), xᵖ, σ_sqrt2)
    𝐘 = get_erf(view(x, 2:2, :, :), yᵖ, σ_sqrt2)
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

# function G2u!(
#     u::AbstractArray{FT,3},
#     G::AbstractArray{FT,3},
#     hτ::FT,
#     F::AbstractMatrix{FT},
# ) where {FT<:AbstractFloat}
#     @. u = F + hτ * G
#     return u
# end

# G2u(G::AbstractArray{FT,3}, hτ::FT, F::AbstractMatrix{FT}) where {FT<:AbstractFloat} =
#     F .+ hτ .* G

# function simulate!(
#     w::AbstractArray{Bool,3},
#     G::AbstractArray{FT,3},
#     hτ::FT,
#     F::AbstractMatrix{FT},
# ) where {FT<:AbstractFloat}
#     u = G2u(G, hτ, F)
#     w .= rand(eltype(u), size(u)) .< -expm1.(-u)
#     return w
# end

# function simulate_w(
#     G::AbstractArray{FT,3},
#     hτ::FT,
#     F::AbstractMatrix{FT},
# ) where {FT<:AbstractFloat}
#     𝐔 = G2u(G, hτ, F)
#     return rand(eltype(𝐔), size(𝐔)) .< -expm1.(-𝐔)
# end

intensity2frame(𝐔::AbstractArray) = rand(eltype(𝐔), size(𝐔)) .< -expm1.(-𝐔)