function get_lnℒ!(
    lnℒ::AbstractArray{FT,3},
    𝐖::AbstractArray{Bool,3},
    𝐔::AbstractArray{FT,3},
) where {FT<:AbstractFloat}
    @. lnℒ = 𝐖 * logexpm1(𝐔) - 𝐔
    return sum(lnℒ)
end

get_lnℒ(𝐖::AbstractArray{Bool,3}, 𝐔::AbstractArray{FT,3}, ::CPU) where {FT<:AbstractFloat} =
    sum(logexpm1.(𝐔[𝐖])) - sum(𝐔)

get_lnℒ(𝐖::AbstractArray{Bool,3}, 𝐔::AbstractArray{FT,3}, ::GPU) where {FT<:AbstractFloat} =
    get_lnℒ!(similar(𝐔), 𝐖, 𝐔)

function get_frame_Δlnℒ(
    𝐖::AbstractArray{Bool,N},
    𝐔ᵒ::AbstractArray{FT,N},
    𝐔ᵖ::AbstractArray{FT,N},
    ::CPU,
) where {FT<:AbstractFloat,N}
    ln𝓇 = similar(𝐔ᵖ, 1, 1, size(𝐔ᵖ, 3))
    Δlnℒ = 𝐔ᵒ .- 𝐔ᵖ
    @. Δlnℒ[𝐖] += logexpm1(𝐔ᵖ[𝐖]) - logexpm1(𝐔ᵒ[𝐖])
    sum!(ln𝓇, Δlnℒ)
    return vec(ln𝓇)
end

function get_frame_Δlnℒ(
    𝐖::AbstractArray{Bool,N},
    𝐔ᵒ::AbstractArray{FT,N},
    𝐔ᵖ::AbstractArray{FT,N},
    ::GPU,
) where {FT<:AbstractFloat,N}
    ln𝓇 = similar(𝐔ᵖ, 1, 1, size(𝐔ᵖ, 3))
    Δlnℒ = @. 𝐖 * (logexpm1(𝐔ᵖ) - logexpm1(𝐔ᵒ)) - (𝐔ᵖ - 𝐔ᵒ)
    return sum!(ln𝓇, Δlnℒ)
end
