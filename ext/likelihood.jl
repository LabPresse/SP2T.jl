# function SP2T._logℒ(W::CuArray{UInt16,N}, U::CuArray{T,N}, ΔU::CuArray{T,N}) where {T,N}
#     @. ΔU = W * logexpm1(U) - U
#     sum(ΔU)
# end

function SP2T._logℒ(
    W::CuArray{<:Integer,N},
    U::CuArray{T,N},
    ΔU::CuArray{T,N},
    𝟙::CuArray{T,N},
) where {T,N}
    @. ΔU = W * logexpm1(U) - U
    SP2T._sum(ΔU, 𝟙)
end

# SP2T.logℒ(W::CuArray, 𝐔::CuArray) = SP2T.logℒ!(similar(𝐔), W, 𝐔)

SP2T._sum(x::CuArray{T,N}, 𝟙::CuArray{T,N}) where {T,N} = x ⋅ 𝟙

function SP2T.Δlogℒ!(
    ΔlogL::CuArray{T,N},
    W::CuArray{UInt16,N},
    U::CuArray{T,N},
    V::CuArray{T,N},
    ΔU::CuArray{T,N},
    𝑇::Union{T,Int} = 1,
) where {T,N}
    @. ΔU = W * (logexpm1(V) - logexpm1(U)) - (V - U)
    sum!(ΔlogL, ΔU)
    ΔlogL ./= 𝑇
end

function SP2T.Δlogℒ!(
    ΔlogL::CuArray{T,N},
    W::CuArray{UInt16,N},
    U::CuArray{T,N},
    V::CuArray{T,N},
    ΔU::CuArray{T,N},
    𝟙::CuArray{T,N},
    𝑇::Union{T,Int} = 1,
) where {T,N}
    @. ΔU = W * (logexpm1(V) - logexpm1(U)) - (V - U)
    SP2T._sum!(ΔlogL, ΔU, 𝟙)
    ΔlogL ./= 𝑇
end

function SP2T._sum!(o::CuArray{T,3}, x::CuArray{T,3}, 𝟙::CuArray{T,3}) where {T}
    x′ = reshape(x, 1, :, size(x, 3))
    𝟙′ = reshape(𝟙, :, 1, size(𝟙, 3))
    batched_mul!(o, x′, 𝟙′)
end