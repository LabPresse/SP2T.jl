function SP2T.Data(
    T::DataType,
    frames::CuArray{UInt16,3},
    period::Real,
    pxsize::Real,
    darkcounts::CuMatrix{<:Real},
    maskthreshold::Real,
    NA::Real,
    refractiveindex::Real,
    wavelength::Real,
)
    period, pxsize, NA, refractiveindex, wavelength =
        convert.(T, (period, pxsize, NA, refractiveindex, wavelength))
    mask = CuMatrix{Bool}(undef, size(darkcounts))
    mask .= darkcounts .< maskthreshold
    return Data(
        frames,
        1,
        period,
        CuArray(range(0, step = pxsize, length = size(darkcounts, 1) + 1)),
        CuArray(range(0, step = pxsize, length = size(darkcounts, 2) + 1)),
        convert(CuMatrix{T}, darkcounts),
        mask,
        CircularGaussianLorentzian{T}(NA = NA, nᵣ = refractiveindex, λ = wavelength),
    )
end

function SP2T.Data(
    T::DataType,
    frames::CuArray{UInt16,3},
    period::Real,
    pxsize::Real,
    darkcounts::CuMatrix{<:Real},
    maskthreshold::Real,
    σ₀::Real,
    z₀::Real,
)
    period, pxsize, σ₀, z₀ = convert.(T, (period, pxsize, σ₀, z₀))
    mask = CuMatrix{Bool}(undef, size(darkcounts))
    mask .= darkcounts .< maskthreshold
    return Data(
        frames,
        1,
        period,
        CuArray(range(0, step = pxsize, length = size(darkcounts, 1) + 1)),
        CuArray(range(0, step = pxsize, length = size(darkcounts, 2) + 1)),
        convert(CuMatrix{T}, darkcounts),
        mask,
        CircularGaussianLorentzian(σ₀, z₀),
    )
end
