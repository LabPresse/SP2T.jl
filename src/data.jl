abstract type Device end

struct CPU <: Device end

struct GPU <: Device end

mutable struct ExperimentalParameter{FT<:AbstractFloat}
    units::Tuple{String,String}
    length::Int
    period::FT
    fourτ::FT
    pxboundsx::AbstractVector{FT}
    pxboundsy::AbstractVector{FT}
    pxnumx::Int
    pxnumy::Int
    pxsize::FT
    pxarea::FT
    darkcounts::AbstractMatrix{FT}
    NA::FT
    nᵣ::FT
    λ::FT
    PSF::AbstractPSF{FT}
end

ExperimentalParameter(
    FT::DataType;
    units::Tuple{String,String} = ("μm", "s"),
    length::Integer,
    period::Real,
    pxsize::Real,
    darkcounts::AbstractMatrix{<:Real},
    NA::Real,
    nᵣ::Real,
    λ::Real,
) = ExperimentalParameter{FT}(
    units,
    length,
    period,
    4 * period,
    range(0, step = pxsize, length = size(darkcounts, 1) + 1),
    range(0, step = pxsize, length = size(darkcounts, 2) + 1),
    size(darkcounts, 1),
    size(darkcounts, 2),
    pxsize,
    pxsize^2,
    darkcounts,
    NA,
    nᵣ,
    λ,
    CircularGaussianLorentzian{FT}(NA, nᵣ, λ),
)

ExperimentalParameter(
    FT::DataType;
    units::Tuple{String,String} = ("μm", "s"),
    length::Integer,
    period::Real,
    pxsize::Real,
    darkcounts::AbstractMatrix{<:Real},
    z₀::Real,
    σ₀::Real,
) = ExperimentalParameter{FT}(
    units,
    length,
    period,
    4 * period,
    range(0, step = pxsize, length = size(darkcounts, 1) + 1),
    range(0, step = pxsize, length = size(darkcounts, 2) + 1),
    size(darkcounts, 1),
    size(darkcounts, 2),
    pxsize,
    pxsize^2,
    darkcounts,
    Inf,
    Inf,
    Inf,
    CircularGaussianLorentzian{FT}(z₀, σ₀, sqrt(2) * σ₀),
)

ftypeof(p::ExperimentalParameter{FT}) where {FT} = FT

mutable struct Video{FT<:AbstractFloat}
    data::AbstractArray{Bool,3}
    param::ExperimentalParameter{FT}
end

ftypeof(p::Video{FT}) where {FT} = FT
