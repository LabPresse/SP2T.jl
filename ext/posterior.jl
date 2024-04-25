function get_ln𝒫(
    ::Brownian,
    fourDτ::FT,
    𝒫::GeneralDistribution,
    x::CuArray{FT,3},
) where {FT<:AbstractFloat}
    num_Δx²::FT, total_Δx² = sum_Δx²(x)
    ln𝒫 = -log(fourDτ) * num_Δx² / 2 - total_Δx² / fourDτ
    ln𝒫 += get_ln𝒫(𝒫, Array(view(x, :, :, 1)))
    return ln𝒫
end