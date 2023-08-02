abstract type AbstractPSF end

struct CircularGaussianLorenzian{FT<:AbstractFloat} <: AbstractPSF
    z_ref::FT # [length] std of psf along z  (optical axis)
    σ_ref::FT # [length] std of psf along xy (image plane)
    σ_ref_sqrt2::FT # [length] std of psf along xy (image plane)
    function CircularGaussianLorenzian{FT}(NA::FT, nᵣ::FT, λ::FT) where {FT<:AbstractFloat}
        cos12α = sqrt(cos(asin(NA / nᵣ)))
        cos32α = cos12α^3
        cos72α = cos12α^7
        a = λ / pi / nᵣ
        b = ((7 * (1 - cos32α)) / (4 - 7 * cos32α + 3 * cos72α))
        z_ref = a * b
        σ_ref = sqrt(a * z_ref) / 2
        new{FT}(z_ref, σ_ref, sqrt(2) * σ_ref)
    end
    CircularGaussianLorenzian(
        z_ref::FT,
        σ_ref::FT,
        σ_ref_sqrt2::FT,
    ) where {FT<:AbstractFloat} = CircularGaussianLorenzian{FT}(z_ref, σ_ref, σ_ref_sqrt2)
end