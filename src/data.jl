abstract type Device end

struct CPU <: Device end

struct GPU <: Device end

mutable struct ExperimentalParameter{FT<:AbstractFloat}
    period::FT
    pxboundsx::AbstractVector{FT}
    pxboundsy::AbstractVector{FT}
    darkcounts::AbstractMatrix{FT}
    PSF::AbstractPSF{FT}
end

getpxsize(param::ExperimentalParameter) = param.pxboundsx[2] - param.pxboundsx[1]

ExperimentalParameter(
    FT::DataType;
    period::Real,
    pxsize::Real,
    darkcounts::AbstractMatrix{<:Real},
    NA::Real,
    nᵣ::Real,
    λ::Real,
) = ExperimentalParameter{FT}(
    period,
    range(0, step = pxsize, length = size(darkcounts, 1) + 1),
    range(0, step = pxsize, length = size(darkcounts, 2) + 1),
    darkcounts,
    CircularGaussianLorentzian{FT}(NA = NA, nᵣ = nᵣ, λ = λ),
)

ExperimentalParameter(
    FT::DataType;
    period::Real,
    pxsize::Real,
    darkcounts::AbstractMatrix{<:Real},
    z₀::Real,
    σ₀::Real,
) = ExperimentalParameter{FT}(
    period,
    range(0, step = pxsize, length = size(darkcounts, 1) + 1),
    range(0, step = pxsize, length = size(darkcounts, 2) + 1),
    darkcounts,
    CircularGaussianLorentzian{FT}(z₀, σ₀),
)

ftypeof(p::ExperimentalParameter{FT}) where {FT} = FT

struct Video{FT<:AbstractFloat}
    frames::AbstractArray{Bool,3}
    param::ExperimentalParameter{FT}
    metadata::Dict{String,Any}
end

ftypeof(v::Video{FT}) where {FT} = FT

_length(v::Video) = size(v.frames, 3)
