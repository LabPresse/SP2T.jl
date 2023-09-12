function get_ϵ(𝒬::Beta)
    ϵ = rand(𝒬)
    rand() < 0.5 && (ϵ = 1 / ϵ)
    return ϵ
end

function get_hᵖ(hᵒ::FT, 𝒬::Beta{FT}) where {FT<:AbstractFloat}
    ϵ = rand(𝒬)
    hᵖ = if rand() > 0.5
        hᵒ * ϵ
    else
        hᵒ / ϵ
    end
    return hᵖ
end

function diff_lnℒ_h(
    w::AbstractArray{Bool,3},
    G::AbstractArray{FT,3},
    hᵖ::FT,
    hᵒ::FT,
    F::AbstractMatrix{FT},
) where {FT<:AbstractFloat}
    uᵖ = F .+ hᵖ .* G
    uᵒ = F .+ hᵒ .* G
    lnℒ_diff = w .* (logexpm1.(uᵖ) .- logexpm1.(uᵒ)) .- (uᵖ .- uᵒ)
    return dot(CUDA.ones(eltype(lnℒ_diff), size(lnℒ_diff)), lnℒ_diff)
end

diff_ln𝒫_h(hᵖ::FT, hᵒ::FT, 𝒫::Gamma{FT}) where {FT<:AbstractFloat} =
    (shape(𝒫) - 1) * log(hᵖ / hᵒ) - (hᵖ - hᵒ) / scale(𝒫)

diff_ln𝒬_h(hᵖ::FT, hᵒ::FT) where {FT<:AbstractFloat} = log(hᵖ / hᵒ)

get_ln𝓇_h(
    w::AbstractArray{Bool,3},
    G::AbstractArray{FT,3},
    hᵖ::FT,
    hᵒ::FT,
    F::AbstractMatrix{FT},
    𝒫::Gamma{FT},
) where {FT<:AbstractFloat} =
    diff_lnℒ_h(w, G, hᵖ, hᵒ, F) + diff_ln𝒫_h(hᵖ, hᵒ, 𝒫) + diff_ln𝒬_h(hᵖ, hᵒ)

function sample_h(
    w::AbstractArray{Bool},
    G::AbstractArray{FT},
    hᵒ::FT,
    F::AbstractMatrix{FT},
    𝒬::Beta{FT},
    𝒫::Gamma{FT},
) where {FT<:AbstractFloat}
    hᵖ = get_hᵖ(hᵒ, 𝒬)
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
