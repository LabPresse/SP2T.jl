function SP2T._logℒ(W::CuArray{<:Integer}, U::CuArray, ΔU::CuArray)
    @. ΔU = W * logexpm1(U) - U
    return sum(ΔU)
end

# SP2T.logℒ(W::CuArray, 𝐔::CuArray) = SP2T.logℒ!(similar(𝐔), W, 𝐔)

function SP2T.frame_Δlogℒ!(ΔlogL::CuArray, W, U, Uᵖ, temp, T = 1)
    @. temp = W * (logexpm1(Uᵖ) - logexpm1(U)) - (Uᵖ - U)
    sum!(ΔlogL, temp)
    return ΔlogL ./= T
end