mutable struct Brightness{T}
    value::T
    priorparams::NTuple{2,T}
    proposalparam::T
end

Brightness(; value, priorparams, proposalparam, scale::T) where {T} = Brightness(
    convert(T, value * scale),
    convert.(T, (priorparams[1], priorparams[2] * scale)),
    convert(T, proposalparam),
)

# function get_ϵ(𝒬::Beta)
#     ϵ = rand(𝒬)
#     return ifelse(bitrand(), ϵ, 1 / ϵ)
# end

function proposebrightness(h::T, 𝒬::Beta{T}) where {T<:AbstractFloat}
    ϵ = rand(𝒬)
    return bitrand() ? h * ϵ : h / ϵ
end

function diff_lnℒ_h(
    w::AbstractArray{Bool,3},
    G::AbstractArray{T,3},
    hᵖ::T,
    hᵒ::T,
    F::AbstractMatrix{T},
) where {T<:AbstractFloat}
    uᵖ = F .+ hᵖ .* G
    uᵒ = F .+ hᵒ .* G
    lnℒ_diff = w .* (logexpm1.(uᵖ) .- logexpm1.(uᵒ)) .- (uᵖ .- uᵒ)
    return dot(CUDA.ones(eltype(lnℒ_diff), size(lnℒ_diff)), lnℒ_diff)
end

diff_ln𝒫_h(hᵖ::T, hᵒ::T, 𝒫::Gamma{T}) where {T<:AbstractFloat} =
    (shape(𝒫) - 1) * log(hᵖ / hᵒ) - (hᵖ - hᵒ) / scale(𝒫)

diff_ln𝒬_h(hᵖ::T, hᵒ::T) where {T<:AbstractFloat} = log(hᵖ / hᵒ)

get_ln𝓇_h(
    w::AbstractArray{Bool,3},
    G::AbstractArray{T,3},
    hᵖ::T,
    hᵒ::T,
    F::AbstractMatrix{T},
    𝒫::Gamma{T},
) where {T<:AbstractFloat} =
    diff_lnℒ_h(w, G, hᵖ, hᵒ, F) + diff_ln𝒫_h(hᵖ, hᵒ, 𝒫) + diff_ln𝒬_h(hᵖ, hᵒ)

function sample_h(
    w::AbstractArray{Bool},
    G::AbstractArray{T},
    hᵒ::T,
    F::AbstractMatrix{T},
    𝒬::Beta{T},
    𝒫::Gamma{T},
) where {T<:AbstractFloat}
    hᵖ = proposebrightness(hᵒ, 𝒬)
    ln𝓇 = get_ln𝓇_h(w, G, hᵖ, hᵒ, F, 𝒫)
    ln𝓊 = log(rand())
    return ln𝓇 > ln𝓊 ? hᵖ : hᵒ
end

# function update_h!(s::ChainStatus, 𝒫::Sampleable, proposal::Proposal, w::AbstractArray{Bool})
#     hᵒ, 𝒬 = s.h, proposal.distritbution
#     hᵖ = get_ϵ(𝒬) * hᵒ
#     ln𝓇 = diff_lnℒ(w, s.G, hᵖ, hᵒ, s.F) + diff_ln𝒬(hᵖ, hᵒ, 𝒫)
#     proposal.accep_count[2] += 1
#     accepted = ln𝓇 > log(rand())
#     proposal.accep_count[1] += accepted
#     s.h = accepted ? hᵖ : hᵒ
#     return s
# end
