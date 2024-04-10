abstract type Device end

struct CPU <: Device end

struct GPU <: Device end

mutable struct ExperimentalParameter{FT<:AbstractFloat}
    length::Int
    period::FT
    pxboundsx::AbstractVector{FT}
    pxboundsy::AbstractVector{FT}
    pxnumx::Int
    pxnumy::Int
    pxsize::FT
    darkcounts::AbstractMatrix{FT}
    PSF::AbstractPSF{FT}
end

ExperimentalParameter(
    FT::DataType;
    length::Integer,
    period::Real,
    pxsize::Real,
    darkcounts::AbstractMatrix{<:Real},
    NA::Real,
    nᵣ::Real,
    λ::Real,
) = ExperimentalParameter{FT}(
    length,
    period,
    range(0, step = pxsize, length = size(darkcounts, 1) + 1),
    range(0, step = pxsize, length = size(darkcounts, 2) + 1),
    size(darkcounts, 1),
    size(darkcounts, 2),
    pxsize,
    darkcounts,
    CircularGaussianLorentzian{FT}(NA = NA, nᵣ = nᵣ, λ = λ),
)

ExperimentalParameter(
    FT::DataType;
    length::Integer,
    period::Real,
    pxsize::Real,
    darkcounts::AbstractMatrix{<:Real},
    z₀::Real,
    σ₀::Real,
) = ExperimentalParameter{FT}(
    length,
    period,
    range(0, step = pxsize, length = size(darkcounts, 1) + 1),
    range(0, step = pxsize, length = size(darkcounts, 2) + 1),
    size(darkcounts, 1),
    size(darkcounts, 2),
    pxsize,
    darkcounts,
    CircularGaussianLorentzian{FT}(z₀, σ₀),
)

ftypeof(p::ExperimentalParameter{FT}) where {FT} = FT

struct Video{FT<:AbstractFloat}
    frames::AbstractArray{Bool,3}
    param::ExperimentalParameter{FT}
    metadata::Dict{String,Any}
end

ftypeof(p::Video{FT}) where {FT} = FT
