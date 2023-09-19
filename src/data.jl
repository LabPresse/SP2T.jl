abstract type Device end

struct CPU <: Device end

struct GPU <: Device end

mutable struct ExperimentalParameter{FT<:AbstractFloat}
    units::Tuple{String,String}
    length::Int
    period::FT
    exposure::FT
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
    ExperimentalParameter(
        FT::DataType;
        units::Tuple{String,String} = ("μm", "s"),
        length::Integer,
        period::Real,
        exposure::Real,
        pxsize::Real,
        darkcounts::AbstractMatrix{<:Real},
        NA::Real,
        nᵣ::Real,
        λ::Real,
        offsetx::Real = 0,
        offsety::Real = 0,
    ) = new{FT}(
        units,
        length,
        period,
        exposure,
        range(offsetx, step = pxsize, length = size(darkcounts, 1) + 1),
        range(offsety, step = pxsize, length = size(darkcounts, 2) + 1),
        size(darkcounts, 1),
        size(darkcounts, 2),
        pxsize,
        pxsize^2,
        darkcounts,
        NA,
        nᵣ,
        λ,
        CircularGaussianLorenzian{FT}(NA, nᵣ, λ),
    )
end

ftypeof(p::ExperimentalParameter{FT}) where {FT} = FT

mutable struct Video{FT<:AbstractFloat}
    data::AbstractArray{Bool,3}
    param::ExperimentalParameter{FT}
end

ftypeof(p::Video{FT}) where {FT} = FT
