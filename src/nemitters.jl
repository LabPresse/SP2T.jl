mutable struct NEmitters{Tv}
    value::Int
    logπ::Tv
    logℒ::Tv
    log𝒫::Tv
end

function NEmitters(; value::Integer, maxcount::Integer, logonprob::Real)
    logprior = collect((0:maxcount) .* logonprob)
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
    Sᵤ::AbstractArray{T,3},
) where {T}
    U .= data.darkcounts
    @inbounds for m = 1:size(x, 3)
        add_pxcounts!(U, view(x, :, :, m:m), h, data)
        M.logℒ[m+1] = logℒ(data, U, Sᵤ)
    end
    return M
end

randc(logp::AbstractArray) = argmax(logp .- log.(randexp!(similar(logp))))

function sample!(M::NEmitters)
    M.value = randc(M.log𝒫) - 1
    return M
end

function _permute!(
    x::AbstractArray{T,3},
    p::AbstractVector{<:Integer},
    y::AbstractArray{T,3},
) where {T}
    @views copyto!(y, x[:, :, p])
    copyto!(x, y)
end

function permuteemitters!(
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    M::Integer,
) where {T}
    p = randperm(M)
    isequal(p, 1:M) && return x
    @views _permute!(x[:, :, 1:M], p, y[:, :, 1:M])
    return x
end

permuteemitters!(x::AbstractArray{T,3}, y::AbstractArray{T,3}, M::NEmitters) where {T} =
    permuteemitters!(x, y, M.value)