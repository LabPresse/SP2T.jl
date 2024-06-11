mutable struct NEmitters{Tv}
    value::Int
    logπ::Tv
    logℒ::Tv
    log𝒫::Tv
end

function NEmitters(; value, maxcount, onprob)
    logprior = collect((0:maxcount) .* log1p(-onprob))
    return NEmitters(value, logprior, similar(logprior), similar(logprior))
end

maxcount(M::NEmitters) = length(M.log𝒫)

anyactive(M::NEmitters) = M.value > 0

function setlog𝒫!(M::NEmitters, 𝑇::Real)
    @. M.log𝒫 = M.logπ + M.logℒ / 𝑇
    return M
end

# function setlogℒ!(
#     M::NEmitters,
#     V::AbstractArray{T,3},
#     U::AbstractArray{T,3},
#     𝐖::AbstractArray{UInt16,3},
#     x::AbstractArray{T,3},
#     h::T,
#     F::AbstractMatrix{T},
#     xbnds::AbstractVector{T},
#     ybnds::AbstractVector{T},
#     PSF::AbstractPSF{T},
#     ΔU::AbstractArray{T,3},
# ) where {T}
#     V .= F
#     M.logℒ[1] = _logℒ(𝐖, V, ΔU)
#     @inbounds for m = 1:size(x, 2)
#         if m != M.value
#             add_pxcounts!(V, view(x, :, m:m, :), h, xbnds, ybnds, PSF)
#         else
#             copyto!(V, U)
#         end
#         M.logℒ[m+1] = _logℒ(𝐖, V, ΔU)
#     end
#     return M
# end

function setlogℒ!(
    M::NEmitters,
    V::AbstractArray{T,3},
    U::AbstractArray{T,3},
    x::AbstractArray{T,3},
    h::T,
    data::Data,
    ΔU::AbstractArray{T,3},
) where {T}
    V .= data.darkcounts
    M.logℒ[1] = _logℒ(data.frames, V, ΔU)
    @show M.logℒ[1]
    @inbounds for m = 1:size(x, 2)
        if m != M.value
            add_pxcounts!(V, view(x, :, m:m, :), h, data)
        else
            copyto!(V, U)
        end
        M.logℒ[m+1] = _logℒ(data.frames, V, ΔU)
    end
    return M
end

randc(logp) = argmax(logp .- log.(randexp!(similar(logp))))

function sample!(M::NEmitters)
    M.value = randc(M.log𝒫) - 1
    return M
end

function shuffletracks!(x::AbstractArray{T,3}, y::AbstractArray{T,3}, M::Integer) where {T}
    @views begin
        copyto!(y[:, 1:M, :], x[:, randperm(M), :])
        copyto!(x[:, 1:M, :], y[:, 1:M, :])
    end
    return x
end

shuffletracks!(x::AbstractArray{T,3}, y::AbstractArray{T,3}, M::NEmitters) where {T} =
    shuffletracks!(x, y, M.value)