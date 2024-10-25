mutable struct NEmitters{T,V}
    value::Int
    logprior::V
    logℒ::V
    log𝒫::V
end

function NEmitters{T}(;
    value::Integer,
    limit::Integer,
    logonprob::Real,
) where {T<:AbstractFloat}
    logprior = collect((0:limit) .* convert(T, logonprob))
    return NEmitters{T,typeof(logprior)}(
        value,
        logprior,
        similar(logprior),
        similar(logprior),
    )
end

function Base.getproperty(M::NEmitters, s::Symbol)
    if s == :limit
        return length(getfield(M, :log𝒫)) - 1
    else
        return getfield(M, s)
    end
end

# maxcount(M::NEmitters) = length(M.log𝒫) - 1

Base.any(M::NEmitters) = M.value > 0

function setlog𝒫!(M::NEmitters, 𝑇::Real)
    @. M.log𝒫 = M.logprior + M.logℒ / 𝑇
    return M
end

function setlogℒ!(
    M::NEmitters,
    x::AbstractArray{T,3},
    h::T,
    data::Data{T},
    A::AuxiliaryVariables,
) where {T}
    A.U .= data.darkcounts
    @inbounds for m = 1:size(x, 3)
        add_pxcounts!(A.U, view(x, :, :, m:m), h, data.pxboundsx, data.pxboundsy, data.PSF)
        M.logℒ[m+1] = logℒ(data, A)
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