# struct Data{
#     T<:AbstractFloat,
#     F<:AbstractArray{UInt16,3},
#     PSF<:PointSpreadFunction{T},
#     D<:Detector{T},
# }
#     frames::F
#     detector::D
#     psf::PSF
# end

# Data{T}(
#     measurements::AbstractArray{UInt16},
#     detector::Detector{T},
#     NA::Real,
#     refractiveindex::Real,
#     wavelength::Real,
# ) where {T<:AbstractFloat} = Data(
#     measurements,
#     detector,
#     CircularGaussianLorentzian{T}(NA, refractiveindex, wavelength, detector.pxsize),
# )

# function Data{T}(
#     measurements::AbstractArray{UInt16,3},
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

# function Base.getproperty(data::Data, s::Symbol)
#     if s == :nframes
#         return size(getfield(data, :frames), 3)
#     elseif s == :framecenter
#         return framecenter(getfield(data, :detector))
#     else
#         return getfield(data, s)
#     end
# end

# function framecenter(data::Data)
#     center = similar(data.pxboundsx, 3)
#     copyto!(center, [mean(data.pxboundsx), mean(data.pxboundsy), 0])
# end



# add_pxcounts!(U::AbstractArray{T,3}, x::AbstractArray{T,3}, h::T, data::Data) where {T} =
#     add_pxcounts!(U, x, h, data.pxboundsx, data.pxboundsy, data.psf)

# function get_pxPSF(
#     x::AbstractArray{T,3},
#     xᵖ::AbstractVector{T},
#     yᵖ::AbstractVector{T},
#     PSF::PointSpreadFunction{T},
# ) where {T}
#     𝐔 = similar(x, length(xᵖ) - 1, length(yᵖ) - 1, size(x, 1))
#     return add_pxcounts!(𝐔, x, oneunit(T), xᵖ, yᵖ, PSF, 0)
# end

# get_pxPSF(x::AbstractArray{T}, data::Data{T}) where {T} =
#     get_pxPSF(x, data.pxboundsx, data.pxboundsy, data.psf)

