# function logℒ!(temp, frames, U)
#     @. temp = frames * logexpm1(U) - U
#     return sum(temp)
# end

# logℒ(𝐖, 𝐔) = sum(logexpm1.(𝐔[𝐖])) - sum(𝐔)

_logℒ(𝐖, 𝐔, temp) = sum(logexpm1.(𝐔[𝐖])) - sum(𝐔)

function unsafe_frame_Δlogℒ!(logacceptance, W, U, Uᵖ)
    Δlnℒ = U .- Uᵖ
    @. Δlnℒ[W] += logexpm1(Uᵖ[W]) - logexpm1(U[W])
    return sum!(logacceptance, Δlnℒ, init = false)
end

function frame_Δlogℒ!(ΔlogL, W, 𝐔, 𝐔ᵖ, ΔU, T = 1)
    ΔU .= 𝐔 .- 𝐔ᵖ
    @. ΔU[W] += logexpm1(𝐔ᵖ[W]) - logexpm1(𝐔[W])
    sum!(ΔlogL, ΔU)
    return ΔlogL ./= T
end

function frame_Δlogℒ(frames, 𝐔, 𝐔ᵖ)
    ln𝓇 = similar(𝐔, 1, 1, size(𝐔, 3))
    temp = similar(𝐔)
    return frame_Δlogℒ!(ln𝓇, frames, 𝐔, 𝐔ᵖ, temp)
end
