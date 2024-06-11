function SP2T._logℒ(W::CuArray{UInt16}, U::CuArray{T}, ΔU::CuArray{T}) where {T}
    @. ΔU = W * logexpm1(U) - U
    return sum(ΔU)
end

# SP2T.logℒ(W::CuArray, 𝐔::CuArray) = SP2T.logℒ!(similar(𝐔), W, 𝐔)

function SP2T.Δlogℒ!(
    ΔlogL::CuArray{T},
    W::CuArray{UInt16},
    U::CuArray{T},
    V::CuArray{T},
    ΔU::CuArray{T},
    𝑇::Union{T,Int} = 1,
) where {T}
    @. ΔU = W * (logexpm1(V) - logexpm1(U)) - (V - U)
    sum!(ΔlogL, ΔU)
    return ΔlogL ./= 𝑇
end