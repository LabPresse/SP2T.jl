mutable struct ExperimentalParameter{
    Ts<:AbstractFloat,
    Tv<:AbstractVector{Ts},
    Tm<:AbstractMatrix{Ts},
}
    period::Ts
    pxboundsx::Tv
    pxboundsy::Tv
    darkcounts::Tm
    PSF::AbstractPSF{Ts}
end

function ExperimentalParameter(
    T::DataType;
    period::Real,
    pxsize::Real,
    darkcounts::AbstractMatrix{<:Real},
    numericalaperture::Real,
    refractiveindex::Real,
    wavelength::Real,
)
    period, pxsize, numericalaperture, refractiveindex, wavelength =
        convert.(T, (period, pxsize, numericalaperture, refractiveindex, wavelength))
    return ExperimentalParameter(
        period,
        range(0, step = pxsize, length = size(darkcounts, 1) + 1),
        range(0, step = pxsize, length = size(darkcounts, 2) + 1),
        convert(Matrix{T}, darkcounts),
        CircularGaussianLorentzian{T}(
            NA = numericalaperture,
            nᵣ = refractiveindex,
            λ = wavelength,
        ),
    )
end

function ExperimentalParameter(
    T::DataType,
    σ₀::Real,
    z₀::Real;
    period::Real,
    pxsize::Real,
    darkcounts::AbstractMatrix{<:Real},
)
    period, pxsize, σ₀, z₀ = convert.(T, (period, pxsize, σ₀, z₀))
    return ExperimentalParameter(
        period,
        range(0, step = pxsize, length = size(darkcounts, 1) + 1),
        range(0, step = pxsize, length = size(darkcounts, 2) + 1),
        convert(Matrix{T}, darkcounts),
        CircularGaussianLorentzian(σ₀, z₀),
    )
end

_eltype(::ExperimentalParameter{S,T1,T2}) where {S,T1,T2} = S

getpxsize(param::ExperimentalParameter) = param.pxboundsx[2] - param.pxboundsx[1]

struct Video
    frames::AbstractArray{Bool,3}
    param::ExperimentalParameter
    metadata::Dict{String,Any}
end

# _eltype(::Video{T}) where {T} = T

_length(v::Video) = size(v.frames, 3)
