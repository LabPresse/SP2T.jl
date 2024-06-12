# function logℒ!(temp, frames, U)
#     @. temp = frames * logexpm1(U) - U
#     return sum(temp)
# end

# logℒ(𝐖, 𝐔) = sum(logexpm1.(𝐔[𝐖])) - sum(𝐔)
#! must be changed
# _logℒ(W::AbstractArray{UInt16}, U::AbstractArray{T}, temp::AbstractArray{T}) where {T} =
#     sum(logexpm1.(U[W])) - sum(U)

function _logℒ(
    W::AbstractArray{UInt16},
    U::AbstractArray{T},
    ΔU::AbstractArray{T},
) where {T}
    @. ΔU = W * logexpm1(U) - U
    return sum(ΔU)
end

function unsafe_Δlogℒ!(
    logratio::AbstractArray{T},
    W::AbstractArray{UInt16},
    U::AbstractArray{T},
    V::AbstractArray{T},
) where {T}
    Δlnℒ = U .- V
    @. Δlnℒ[W] += logexpm1(V[W]) - logexpm1(U[W])
    return sum!(logratio, Δlnℒ, init = false)
end

# function Δlogℒ!(
#     ΔlogL::AbstractArray{T},
#     W::AbstractArray{UInt16},
#     U::AbstractArray{T},
#     V::AbstractArray{T},
#     ΔU::AbstractArray{T},
#     𝑇::Union{T,Int} = 1,
# ) where {T}
#     ΔU .= U .- V
#     @. ΔU[W] += logexpm1(V[W]) - logexpm1(U[W])
#     sum!(ΔlogL, ΔU)
#     return ΔlogL ./= 𝑇
# end

function Δlogℒ!(
    ΔlogL::AbstractArray{T},
    W::AbstractArray{UInt16},
    U::AbstractArray{T},
    V::AbstractArray{T},
    ΔU::AbstractArray{T},
    𝑇::Union{T,Int} = 1,
) where {T}
    @. ΔU = W * (logexpm1(V) - logexpm1(U)) - (V - U)
    sum!(ΔlogL, ΔU)
    return ΔlogL ./= 𝑇
end

function Δlogℒ(
    W::AbstractArray{UInt16},
    U::AbstractArray{T},
    V::AbstractArray{T},
    𝑇::Union{T,Int} = 1,
) where {T}
    ln𝓇 = similar(U, 1, 1, size(U, 3))
    ΔU = similar(U)
    return Δlogℒ!(ln𝓇, W, U, V, ΔU, 𝑇)
end
