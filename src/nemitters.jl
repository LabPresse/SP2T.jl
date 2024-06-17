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

maxcount(M::NEmitters) = length(M.log𝒫) - 1

anyactive(M::NEmitters) = M.value > 0

function setlog𝒫!(M::NEmitters, 𝑇::Real)
    @. M.log𝒫 = M.logπ + M.logℒ / 𝑇
    return M
end

function setlogℒ!(
    M::NEmitters,
    U::AbstractArray{T,3},
    x::AbstractArray{T,3},
    h::T,
    data::Data,
    ΔU::AbstractArray{T,3},
    𝟙::AbstractArray{T,3},
) where {T}
    U .= data.darkcounts
    @inbounds for m = 1:size(x, 2)
        add_pxcounts!(U, view(x, :, m:m, :), h, data)
        M.logℒ[m+1] = _logℒ(data.frames, U, ΔU, 𝟙)
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