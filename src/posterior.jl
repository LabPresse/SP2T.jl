# This file contains UNNORMALIZED probability density functions.
# The goal is to make things simple and fast. 
# For normalized pdfs use "Distributions.jl".

"""
    get_ln𝒫(𝒫::Beta, x)

    The unnormalized log pdf of the Beta distribution.
"""
get_ln𝒫(𝒫::Beta{FT}, x::FT) where {FT<:AbstractFloat} =
    (𝒫.α - 1) * log(x) + (𝒫.β - 1) * log1p(-x)

get_ln𝒫(𝒫::Beta{FT}, x::AbstractVector{FT}) where {FT<:AbstractFloat} =
    sum((𝒫.α - 1) .* log(x) .+ (𝒫.β - 1) .* log1p(-x))

"""
    get_ln𝒫(𝒫::Gamma, x)

    The unnormalized log pdf of the Gamma distribution.
"""
get_ln𝒫(𝒫::Gamma{FT}, x::FT) where {FT<:AbstractFloat} =
    (shape(𝒫) - 1) * log(x) - x / scale(𝒫)

get_ln𝒫(𝒫::Gamma{FT}, x::AbstractVector{FT}) where {FT<:AbstractFloat} =
    sum((shape(𝒫) - 1) .* log(x) - x ./ scale(𝒫))

"""
    get_ln𝒫(𝒫::InverseGamma, x)

    The unnormalized log pdf of the Inverse-Gamma distribution.
"""
get_ln𝒫(𝒫::InverseGamma{FT}, x::FT) where {FT<:AbstractFloat} =
    (-shape(𝒫) - 1) * log(x) - scale(𝒫) / x

get_ln𝒫(𝒫::InverseGamma{FT}, x::AbstractVector{FT}) where {FT<:AbstractFloat} =
    sum((-shape(𝒫) - 1) .* log.(x) - scale(𝒫) ./ x)

"""
    get_ln𝒫(𝒫::Categorical, x)

    The log pdf of a Categorical distribution with probability vector `p`. `p` does not need to be normalized.
"""
get_ln𝒫(𝒫::Categorical, x::Integer) = log(𝒫.p[x])

get_ln𝒫(𝒫::Categorical, x::AbstractVector{Integer}) = sum(log.(𝒫.p[x]))

"""
    get_ln𝒫(𝒫::Bernoulli, x)

    The log pdf of a Bernoulli distribution with success probability `p`. `p` should be normalized.
"""
get_ln𝒫(𝒫::Bernoulli, x::Bool) = x ? log(𝒫.p) : log1p(-𝒫.p)

function get_ln𝒫(𝒫::Bernoulli, x::AbstractVector{Bool})
    n = count(x)
    return n * log(𝒫.p) + (length(x) - n) * log1p(-𝒫.p)
end

"""
    get_ln𝒫(𝒫::MvNormal, x)

    The log pdf of a Multivariate Normal distribution.
"""
get_ln𝒫(𝒫::MvNormal, x::AbstractVector{<:AbstractFloat}) = logpdf(𝒫, x)

get_ln𝒫(𝒫::MvNormal, x::AbstractMatrix{<:AbstractFloat}) = sum(logpdf(𝒫, x))

get_ln𝒫(𝒫::Geometric, M::Integer) = logpdf(𝒫, M)

get_ln𝒫(x::IID) = get_ln𝒫(x.𝒫, x.value)

function get_ln𝒫(
    ::Brownian,
    fourDτ::FT,
    𝒫::Distribution,
    x::AbstractArray{FT,3},
    device::Device,
) where {FT<:AbstractFloat}
    num_Δx²::FT, total_Δx² = sum_Δx²(x)
    ln𝒫 = -log(fourDτ) * num_Δx² / 2 - total_Δx² / fourDτ
    ln𝒫 += if device isa CPU
        get_ln𝒫(𝒫, view(x, :, :, 1))
    else
        #? improve
        get_ln𝒫(𝒫, Array(view(x, :, :, 1)))
    end
    return ln𝒫
end

get_ln𝒫(x::Trajectory, dynRV::RealNumOrVec, device::Device) =
    get_ln𝒫(x.dynamics, dynRV, x.𝒫, x.value, device)

get_ln𝒫(x::Trajectory, dynRV::RealNumOrVec, B::Integer, device::Device) =
    get_ln𝒫(x.dynamics, dynRV, x.𝒫, view(x.value, :, 1:B, :), device)

# """
#     get_ln𝒫(x, fourDτ)

#     The log pdf of a n-D Brownian motion trajectory (`x`) with diffusion coefficient `D`. As 'D' is often inferred, the D-dependence in the normalization factor is not dropped. The number of dimemsions, n, is `x`'s number of rows.
# """

# get_lnℒ(w::AbstractArray{Bool,3}, 𝐔::AbstractArray{FT,3}, ::GPU) where {FT<:AbstractFloat} =
#     dot(w, logexpm1.(𝐔)) - dot(CUDA.ones(eltype(𝐔), size(𝐔)), 𝐔)

# get_lnℒ(w::AbstractArray{Bool,3}, 𝐔::AbstractArray{FT,3}, ::CPU) where {FT<:AbstractFloat} =
#     sum(logexpm1.(𝐔[w])) - sum(𝐔)

function update_ln𝒫!(s::ChainStatus, v::Video, device::Device)
    s.ln𝒫 =
        get_lnℒ(v.frames, s.𝐔, device) +
        get_ln𝒫(s.tracks, 4 * s.D.value * v.param.period, device) +
        get_ln𝒫(s.M) +
        get_ln𝒫(s.D) +
        get_ln𝒫(s.h)
    return s
end