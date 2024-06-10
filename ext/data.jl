function SP2T.Data(
    T::DataType,
    frames::CuArray{<:Integer,3},
    period::Real,
    pxsize::Real,
    darkcounts::CuMatrix{<:Real},
    NA::Real,
    refractiveindex::Real,
    wavelength::Real,
)
    period, pxsize, NA, refractiveindex, wavelength =
        convert.(T, (period, pxsize, NA, refractiveindex, wavelength))
    return Data(
        frames,
        period,
        CuArray(range(0, step = pxsize, length = size(darkcounts, 1) + 1)),
        CuArray(range(0, step = pxsize, length = size(darkcounts, 2) + 1)),
        convert(CuMatrix{T}, darkcounts),
        CircularGaussianLorentzian{T}(NA = NA, nᵣ = refractiveindex, λ = wavelength),
    )
end

function SP2T.Data(
    T::DataType,
    frames::CuArray{<:Integer,3},
    period::Real,
    pxsize::Real,
    darkcounts::CuMatrix{<:Real},
    σ₀::Real,
    z₀::Real,
)
    period, pxsize, σ₀, z₀ = convert.(T, (period, pxsize, σ₀, z₀))
    return Data(
        frames,
        period,
        CuArray(range(0, step = pxsize, length = size(darkcounts, 1) + 1)),
        CuArray(range(0, step = pxsize, length = size(darkcounts, 2) + 1)),
        convert(CuMatrix{T}, darkcounts),
        CircularGaussianLorentzian(σ₀, z₀),
    )
end