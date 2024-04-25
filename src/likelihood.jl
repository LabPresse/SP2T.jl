get_lnℒ(𝐖, 𝐔, ::CPU) = sum(logexpm1.(𝐔[𝐖])) - sum(𝐔)

function get_frame_Δlnℒ(frames, 𝐔ᵒ, 𝐔ᵖ, ::CPU)
    ln𝓇 = similar(𝐔ᵖ, 1, 1, size(𝐔ᵖ, 3))
    Δlnℒ = 𝐔ᵒ .- 𝐔ᵖ
    @. Δlnℒ[frames] += logexpm1(𝐔ᵖ[frames]) - logexpm1(𝐔ᵒ[frames])
    sum!(ln𝓇, Δlnℒ)
    return vec(ln𝓇)
end
