function SP2T.get_lnℒ!(
    lnℒ::AbstractArray{T,3},
    𝐖::AbstractArray{Bool,3},
    𝐔::AbstractArray{T,3},
) where {T}
    @. lnℒ = 𝐖 * logexpm1(𝐔) - 𝐔
    return sum(lnℒ)
end

SP2T.get_lnℒ(𝐖::CuArray, 𝐔::CuArray) = get_lnℒ!(similar(𝐔), 𝐖, 𝐔)

function SP2T.get_frame_Δlnℒ(frames::CuArray, 𝐔ᵒ::CuArray, 𝐔ᵖ::CuArray)
    ln𝓇 = similar(𝐔ᵖ, 1, 1, size(𝐔ᵖ, 3))
    Δlnℒ = @. frames * (logexpm1(𝐔ᵖ) - logexpm1(𝐔ᵒ)) - (𝐔ᵖ - 𝐔ᵒ)
    return sum!(ln𝓇, Δlnℒ)
end