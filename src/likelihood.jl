# function logℒ!(temp, frames, U)
#     @. temp = frames * logexpm1(U) - U
#     return sum(temp)
# end

# logℒ(𝐖, 𝐔) = sum(logexpm1.(𝐔[𝐖])) - sum(𝐔)
#! must be changed
_logℒ(
    W::AbstractArray{Bool,N},
    U::AbstractArray{T,N},
    ΔU::AbstractArray{T,N},
    𝟙::AbstractArray{T,N},
) where {T,N} = sum(logexpm1.(U[W])) - sum(U)

function _logℒ(
    W::AbstractArray{UInt16,N},
    U::AbstractArray{T,N},
    ΔU::AbstractArray{T,N},
    𝟙::AbstractArray{T,N},
) where {T,N}
    @. ΔU = W * logexpm1(U) - U
    _sum(ΔU, 𝟙)
end

_sum(x::AbstractArray{T,N}, 𝟙::AbstractArray{T,N}) where {T,N} = sum(x)

function unsafe_Δlogℒ!(
    logratio::AbstractArray{T},
    W::AbstractArray{UInt16},
    U::AbstractArray{T},
    V::AbstractArray{T},
) where {T}
    Δlnℒ = U .- V
    @. Δlnℒ[W] += logexpm1(V[W]) - logexpm1(U[W])
    sum!(logratio, Δlnℒ, init = false)
end

function Δlogℒ!(
    ΔlogL::AbstractArray{T,N},
    W::AbstractArray{Bool,N},
    U::AbstractArray{T,N},
    V::AbstractArray{T,N},
    ΔU::AbstractArray{T,N},
    𝑇::Union{T,Int} = 1,
) where {T,N}
    ΔU .= U .- V
    @. ΔU[W] += logexpm1(V[W]) - logexpm1(U[W])
    sum!(ΔlogL, ΔU)
    return ΔlogL ./= 𝑇
end

function Δlogℒ!(
    ΔlogL::AbstractArray{T,N},
    W::AbstractArray{UInt16,N},
    U::AbstractArray{T,N},
    V::AbstractArray{T,N},
    ΔU::AbstractArray{T,N},
    𝑇::Union{T,Int} = 1,
) where {T,N}
    @. ΔU = W * (logexpm1(V) - logexpm1(U)) - (V - U)
    sum!(ΔlogL, ΔU)
    ΔlogL ./= 𝑇
end

function Δlogℒ!(
    ΔlogL::AbstractArray{T,3},
    W::AbstractArray{UInt16,3},
    U::AbstractArray{T,3},
    V::AbstractArray{T,3},
    ΔU::AbstractArray{T,3},
    𝟙::AbstractArray{T,3},
    𝑇::Union{T,Int} = 1,
) where {T}
    @. ΔU = W * (logexpm1(V) - logexpm1(U)) - (V - U)
    _sum!(ΔlogL, ΔU, 𝟙)
    ΔlogL ./= 𝑇
end

_sum!(o::AbstractArray{T,3}, x::AbstractArray{T,3}, 𝟙::AbstractArray{T,3}) where {T} =
    sum!(o, x)

# function Δlogℒ(
#     W::AbstractArray{UInt16,3},
#     U::AbstractArray{T,3},
#     V::AbstractArray{T,3},
#     𝑇::Union{T,Int} = 1,
# ) where {T}
#     ln𝓇 = similar(U, 1, 1, size(U, 3))
#     ΔU = similar(U)
#     Δlogℒ!(ln𝓇, W, U, V, ΔU, 𝑇)
# end
