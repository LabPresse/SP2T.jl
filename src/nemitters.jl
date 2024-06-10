mutable struct NEmitters{Tv}
    value::Int
    logprior::Tv
    logℒ::Tv
    log𝒫::Tv
end

function NEmitters(; value, maxcount, onprob)
    logprior = collect((0:maxcount) .* log1p(-onprob))
    return NEmitters(value, logprior, similar(logprior), similar(logprior))
end

maxcount(M::NEmitters) = length(M.log𝒫)

anyactive(M::NEmitters) = M.value > 0

# function init!(n::NEmitters)
#     n.logposterior .= (0:length(n.logposterior)) .* log(failprob(n.prior))
#     return n
# end

# function init!(n::NEmitters)
#     n.logposterior .= (0:length(n.logposterior)-1) .* log(failprob(n.prior))
#     return n
# end

function setlog𝒫!(M::NEmitters, T)
    @. M.log𝒫 = M.logprior + M.logℒ / T
    return M
end

function setlogℒ!(
    M::NEmitters,
    V::AbstractArray{T,3},
    U::AbstractArray{T,3},
    𝐖::AbstractArray{<:Integer,3},
    x::AbstractArray{T,3},
    h::T,
    F::AbstractMatrix{T},
    xbnds::AbstractVector{T},
    ybnds::AbstractVector{T},
    PSF::AbstractPSF{T},
    ΔU::AbstractArray{T,3},
) where {T}
    V .= F
    M.logℒ[1] = _logℒ(𝐖, V, ΔU)
    @inbounds for m = 1:size(x, 2)
        if m != M.value
            add_pxcounts!(V, view(x, :, m:m, :), h, xbnds, ybnds, PSF)
        else
            copyto!(V, U)
        end
        M.logℒ[m+1] = _logℒ(𝐖, V, ΔU)
    end
    return M
end

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

# function set_lnℒ!(lnℒ, U, 𝐖, x, h, expparams, temp)
#     U .= expparams.darkcounts
#     lnℒ[1] = get_lnℒ(𝐖, U)
#     for m = 1:size(x, 2)
#         add_px_intensity!(
#             U,
#             view(x, :, m:m, :),
#             h,
#             expparams.pxboundsx,
#             expparams.pxboundsy,
#             expparams.PSF,
#         )
#         lnℒ[m+1] = get_lnℒ(𝐖, U)
#     end
#     return lnℒ
# end

# function add_lnℒ!(ln𝒫, 𝐔ᵖ, 𝐔, 𝐖, x, h, xᵖ, yᵖ, PSF)
#     ln𝒫[1] += get_lnℒ(𝐖, 𝐔)
#     for m = 1:size(x, 2)
#         add_px_intensity!(𝐔, view(x, :, m:m, :), h, xᵖ, yᵖ, PSF)
#         ln𝒫[m+1] += get_lnℒ(𝐖, 𝐔)
#     end
#     return ln𝒫
# end

randc(logp) = argmax(logp .- log.(randexp!(similar(logp))))

function sample!(M::NEmitters)
    M.value = randc(M.log𝒫) - 1
    return M
end

# function sample_M(
#     𝐖::AbstractArray{Bool,3},
#     x::AbstractArray{T,3},
#     hτ::T,
#     𝐅::AbstractMatrix{T},
#     xᵖ::AbstractVector{T},
#     yᵖ::AbstractVector{T},
#     PSF::AbstractPSF{T},
#     prior::Geometric{T},
#     temperature::T,
# ) where {T}
#     ln𝒫 = collect(0:size(x, 2)) .* log(failprob(prior))
#     # ln𝒫[1:end-1] .+= log1p(-q)
#     ln𝒫 .*= temperature
#     𝐔 = repeat(𝐅, 1, 1, size(x, 3))
#     add_lnℒ!(ln𝒫, 𝐔, 𝐖, x, xᵖ, yᵖ, hτ, PSF)
#     ln𝒫 ./= temperature
#     return randc(ln𝒫) - 1
# end

# function shuffletracks!(x, M::Integer)
#     x[:, 1:M, :] = x[:, randperm(M), :]
#     return x
# end

# function shuffletracks!(x, y)
#     copyto!(y, view(x, :, randperm(size(x, 2)), :))
#     copyto!(x, y)
#     return x
# end

function shuffletracks!(x::AbstractArray{T,3}, y::AbstractArray{T,3}, M::Integer) where {T}
    @views begin
        copyto!(y[:, 1:M, :], x[:, randperm(M), :])
        copyto!(x[:, 1:M, :], y[:, 1:M, :])
    end
    return x
end

# shuffletracks!(x::AbstractArray{T,3}, M::Integer, y::AbstractArray{T,3}) where {T} =
#     shuffletracks!(view(x, :, 1:M, :), view(y, :, 1:M, :))

shuffletracks!(x::AbstractArray{T,3}, y::AbstractArray{T,3}, M::NEmitters) where {T} =
    shuffletracks!(x, y, M.value)

# function update_emittercount!(s::ChainStatus, v::Video)
#     shuffle_on_x!(s.tracks.value, s.emittercount.value)
#     s.emittercount.value = sample_M(
#         v.frames,
#         s.tracks.value,
#         v.param.pxboundsx,
#         v.param.pxboundsy,
#         s.brightness.value,
#         v.param.darkcounts,
#         v.param.PSF,
#         s.emittercount.prior,
#         s.temperature,
#     )
#     return s
# end

# to_gpu(n::NEmitters) = n