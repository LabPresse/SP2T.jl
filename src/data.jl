struct Data{
    T<:AbstractFloat,
    F<:AbstractArray{UInt16,3},
    VofT<:AbstractVector{T},
    MofT<:AbstractMatrix{T},
    MofB<:AbstractMatrix{Bool},
    PSFofT<:PointSpreadFunction{T},
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

function Data{T}(
    frames::AbstractArray{UInt16,3},
    period::Real,
    pxsize::Real,
    darkcounts::AbstractMatrix{<:Real},
    cutoffs::Tuple{<:Real,<:Real},
    NA::Real,
    refractiveindex::Real,
    wavelength::Real,
) where {T<:AbstractFloat}
    period, pxsize = convert.(T, (period, pxsize))
    PSF = CircularGaussianLorentzian{T}(NA, refractiveindex, wavelength, pxsize)
    darkcounts = T.(darkcounts)
    mask = similar(darkcounts, Bool)
    mask .= cutoffs[1] .< darkcounts .< cutoffs[2]

    pxboundsx = similar(darkcounts, size(darkcounts, 1) + 1)
    pxboundsx .= 0:pxsize:size(darkcounts, 1)*pxsize
    pxboundsy = similar(darkcounts, size(darkcounts, 2) + 1)
    pxboundsy .= 0:pxsize:size(darkcounts, 2)*pxsize

    return Data(frames, 1, period, pxboundsy, pxboundsy, darkcounts, mask, PSF)
end

# function Data{T}(
#     frames::AbstractArray{UInt16,3},
#     period::Real,
#     pxsize::Real,
#     darkcounts::AbstractMatrix{<:Real},
#     cutoffs::Tuple{<:Real,<:Real},
#     σ₀::Real,
#     z₀::Real,
# ) where {T<:AbstractFloat}
#     period, pxsize = convert.(T, (period, pxsize))
#     PSF = CircularGaussianLorentzian{T}(σ₀, z₀)
#     darkcounts = T.(darkcounts)
#     mask = similar(darkcounts, Bool)
#     mask .= cutoffs[1] .< darkcounts .< cutoffs[2]

#     pxboundsx = similar(darkcounts, size(darkcounts, 1) + 1)
#     pxboundsx .= 0:pxsize:size(darkcounts, 1)*pxsize
#     pxboundsy = similar(darkcounts, size(darkcounts, 2) + 1)
#     pxboundsy .= 0:pxsize:size(darkcounts, 2)*pxsize

#     return Data(frames, 1, period, pxboundsx, pxboundsy, darkcounts, mask, PSF)
# end

function Base.getproperty(data::Data, s::Symbol)
    if s == :nframes
        return size(getfield(data, :frames), 3)
    elseif s == :framecenter
        return mean(data.pxboundsx), mean(data.pxboundsy)
    else
        return getfield(data, s)
    end
end

function framecenter(data::Data)
    center = similar(data.pxboundsx, 3)
    copyto!(center, [mean(data.pxboundsx), mean(data.pxboundsy), 0])
end

function add_pxcounts!(
    𝐔::AbstractArray{T,3},
    x::AbstractArray{T,3},
    h::T,
    xᵖ::AbstractVector{T},
    yᵖ::AbstractVector{T},
    PSF::CircularGaussianLorentzian{T},
    β = 1,
) where {T<:AbstractFloat}
    @views begin
        σ = lateral_std(x[:, 3:3, :], PSF)
        𝐗 = _erf(x[:, 1:1, :], xᵖ, σ)
        𝐘 = _erf(x[:, 2:2, :], yᵖ, σ)
    end
    return batched_mul!(𝐔, 𝐗, batched_transpose(𝐘), h / PSF.A, β)
end

# add_pxcounts!(U::AbstractArray{T,3}, x::AbstractArray{T,3}, h::T, data::Data) where {T} =
#     add_pxcounts!(U, x, h, data.pxboundsx, data.pxboundsy, data.PSF)

function get_pxPSF(
    x::AbstractArray{T,3},
    xᵖ::AbstractVector{T},
    yᵖ::AbstractVector{T},
    PSF::PointSpreadFunction{T},
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
    PSF::PointSpreadFunction{T},
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
    PSF::PointSpreadFunction{T},
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