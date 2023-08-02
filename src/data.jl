struct ExperimentalParameter{FT<:AbstractFloat}
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
    pxareatimesexposure::FT
    NA::FT
    nᵣ::FT
    λ::FT
    PSF::AbstractPSF
    ExperimentalParameter(
        FT::DataType;
        units::Tuple{String,String} = ("μm", "s"),
        length::Integer,
        period::Real,
        exposure::Real,
        pxnumx::Integer,
        pxnumy::Integer,
        pxsize::Real,
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
        collect(range(offsetx, step = pxsize, length = pxnumx + 1)),
        collect(range(offsety, step = pxsize, length = pxnumy + 1)),
        pxnumx,
        pxnumy,
        pxsize,
        pxsize^2,
        pxsize^2 * exposure,
        NA,
        nᵣ,
        λ,
        CircularGaussianLorenzian(NA, nᵣ, λ),
    )
end

convert(FT::AbstractFloat, params::ExperimentalParameter) = ExperimentalParameter{FT}(
    params.units,
    params.length,
    params.period,
    params.exposure,
    params.pxboundsx,
    params.pxboundsy,
    params.pxnumx,
    params.pxnumy,
    params.pxsize,
    params.pxarea,
    params.pxareatimesexposure,
    params.NA,
    params.nᵣ,
    params.λ,
    params.PSF,
)

struct Video
    data::BitArray{3}
    param::ExperimentalParameter
    Video(p::ExperimentalParameter) = new(zeros(Bool, p.pxnumx, p.pxnumy, p.length), p)
end