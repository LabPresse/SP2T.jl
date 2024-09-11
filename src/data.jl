abstract type AbstractPSF{T} end

struct CircularGaussianLorentzian{T} <: AbstractPSF{T}
    σ₀::T # [length] std of PSF along xy (image plane)
    z₀::T # [length] std of PSF along z (optical axis)
end

function CircularGaussianLorentzian{T}(;
    NA::Real,
    refractiveindex::Real,
    wavelength::Real,
) where {T<:AbstractFloat}
    a = wavelength / pi / refractiveindex
    b = _b(NA, refractiveindex)
    z₀ = a * b
    σ₀ = sqrt(a * z₀) / 2
    return CircularGaussianLorentzian{T}(σ₀, z₀)
end

function _b(NA::T, nᵣ::T) where {T<:AbstractFloat}
    α = semiangle(NA, nᵣ)
    cos12α = sqrt(cos(α))
    cos32α, cos72α = cos12α^3, cos12α^7
    return ((7 * (1 - cos32α)) / (4 - 7 * cos32α + 3 * cos72α))
end

semiangle(NA::T, nᵣ::T) where {T<:AbstractFloat} = asin(NA / nᵣ)

_σ!(
    σ::AbstractArray{T,N},
    z::AbstractArray{T,N},
    PSF::CircularGaussianLorentzian{T},
) where {T,N} = @. σ = PSF.σ₀ * √(oneunit(T) + (z / PSF.z₀)^2)

function _σ(
    z::AbstractArray{T},
    PSF::CircularGaussianLorentzian{T},
) where {T<:AbstractFloat}
    z′ = PermutedDimsArray(z, (2, 3, 1))
    return _σ!(similar(z′), z′, PSF)
end

function _erf(x::AbstractArray{T}, bnds::AbstractVector{T}, σ::AbstractArray{T}) where {T}
    𝐗 = @. (bnds - $PermutedDimsArray(x, (2, 3, 1))) / (√convert(T, 2) * σ)
    return @views erf.(𝐗[1:end-1, :, :], 𝐗[2:end, :, :]) ./ 2
end

function maxPSF(PSF::CircularGaussianLorentzian, pxsize::Real)
    x = pxsize / 2 / (√2 * PSF.σ₀)
    erf(-x, x)^2 / 4
end

struct Data{
    F<:AbstractArray{UInt16,3},
    T<:AbstractFloat,
    VofT<:AbstractVector{T},
    MofT<:AbstractMatrix{T},
    MofB<:AbstractMatrix{Bool},
    PSFofT<:AbstractPSF{T},
}
    frames::F
    batchsize::Int
    period::T
    pxboundsx::VofT
    pxboundsy::VofT
    darkcounts::MofT
    filter::MofB
    PSF::PSFofT
end

function Data(
    T::DataType,
    frames::AbstractArray{UInt16,3},
    period::Real,
    pxsize::Real,
    darkcounts::AbstractMatrix{<:Real},
    cutoffs::Tuple{<:Real,<:Real},
    NA::Real,
    refractiveindex::Real,
    wavelength::Real,
)
    period, pxsize = convert.(T, (period, pxsize))
    PSF = CircularGaussianLorentzian{T}(
        NA = NA,
        refractiveindex = refractiveindex,
        wavelength = wavelength,
    )
    darkcounts = T.(darkcounts)
    mask = similar(darkcounts, Bool)
    mask .= cutoffs[1] .< darkcounts .< cutoffs[2]

    pxboundsx = similar(darkcounts, size(darkcounts, 1) + 1)
    pxboundsx .= 0:pxsize:size(darkcounts, 1)*pxsize
    pxboundsy = similar(darkcounts, size(darkcounts, 2) + 1)
    pxboundsy .= 0:pxsize:size(darkcounts, 2)*pxsize

    return Data(frames, 1, period, pxboundsy, pxboundsy, darkcounts, mask, PSF)
end

function Data(
    T::DataType,
    frames::AbstractArray{UInt16,3},
    period::Real,
    pxsize::Real,
    darkcounts::AbstractMatrix{<:Real},
    cutoffs::Tuple{<:Real,<:Real},
    σ₀::Real,
    z₀::Real,
)
    period, pxsize = convert.(T, (period, pxsize))
    PSF = CircularGaussianLorentzian{T}(σ₀, z₀)
    darkcounts = T.(darkcounts)
    mask = similar(darkcounts, Bool)
    mask .= cutoffs[1] .< darkcounts .< cutoffs[2]

    pxboundsx = similar(darkcounts, size(darkcounts, 1) + 1)
    pxboundsx .= 0:pxsize:size(darkcounts, 1)*pxsize
    pxboundsy = similar(darkcounts, size(darkcounts, 2) + 1)
    pxboundsy .= 0:pxsize:size(darkcounts, 2)*pxsize

    return Data(frames, 1, period, pxboundsx, pxboundsy, darkcounts, mask, PSF)
end

framecenter(data::Data) = [
    (data.pxboundsx[1] + data.pxboundsx[end]) / 2,
    (data.pxboundsy[1] + data.pxboundsy[end]) / 2,
    0,
]

function add_pxcounts!(
    𝐔::AbstractArray{T,3},
    x::AbstractArray{T,3},
    h::T,
    xᵖ::AbstractVector{T},
    yᵖ::AbstractVector{T},
    PSF::AbstractPSF{T},
    β = 1,
) where {T<:AbstractFloat}
    @views begin
        σ = _σ(x[:, 3:3, :], PSF)
        𝐗 = _erf(x[:, 1:1, :], xᵖ, σ)
        𝐘 = _erf(x[:, 2:2, :], yᵖ, σ)
    end
    return batched_mul!(𝐔, 𝐗, batched_transpose(𝐘), h, β)
end

add_pxcounts!(U::AbstractArray{T,3}, x::AbstractArray{T,3}, h::T, data::Data) where {T} =
    add_pxcounts!(U, x, h, data.pxboundsx, data.pxboundsy, data.PSF)

function get_pxPSF(
    x::AbstractArray{T,3},
    xᵖ::AbstractVector{T},
    yᵖ::AbstractVector{T},
    PSF::AbstractPSF{T},
) where {T}
    𝐔 = similar(x, length(xᵖ) - 1, length(yᵖ) - 1, size(x, 1))
    return add_pxcounts!(𝐔, x, oneunit(T), xᵖ, yᵖ, PSF, 0)
end

get_pxPSF(x::AbstractArray, data::Data) =
    get_pxPSF(x, data.pxboundsx, data.pxboundsy, data.PSF)

function pxcounts!(
    U::AbstractArray{T,3},
    x::AbstractArray{T,3},
    h::T,
    F::AbstractMatrix{T},
    xbnds::AbstractVector{T},
    ybnds::AbstractVector{T},
    PSF::AbstractPSF{T},
) where {T}
    U .= F
    return add_pxcounts!(U, x, h, xbnds, ybnds, PSF)
end

pxcounts!(U::AbstractArray{T,3}, x::AbstractArray{T,3}, h::T, data::Data) where {T} =
    pxcounts!(U, x, h, data.darkcounts, data.pxboundsx, data.pxboundsy, data.PSF)

function pxcounts(
    x::AbstractArray{T,3},
    h::T,
    DC::AbstractMatrix{T},
    xᵖ::AbstractVector{T},
    yᵖ::AbstractVector{T},
    PSF::AbstractPSF{T},
) where {T}
    𝐔 = repeat(DC, 1, 1, size(x, 1))
    return add_pxcounts!(𝐔, x, h, xᵖ, yᵖ, PSF)
end

pxcounts(x::AbstractArray{T,3}, h::T, data::Data) where {T} =
    pxcounts(x, h, data.darkcounts, data.pxboundsx, data.pxboundsy, data.PSF)

simframes!(W::AbstractArray{UInt16,3}, U::AbstractArray{<:Real,3}) =
    @. W = $rand!($similar(U)) < -expm1(-U)

simframes!(W::AbstractArray{UInt16,3}, U::AbstractArray{<:Real,3}, B::Integer) =
    @. W = rand(Binomial(B, -expm1(-U)))

function simframes(U::AbstractArray{<:Real,3}, B::Integer = 1)
    W = similar(U, UInt16)
    if B == 1
        simframes!(W, U)
    else
        simframes!(W, U, B)
    end
end