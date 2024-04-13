function sum_Δx²(x::AbstractArray{FT,3}) where {FT<:AbstractFloat}
    Δx² = diff(x, dims = 3) .^ 2
    return length(Δx²), sum(Δx²)
end

function sample_D(
    x::AbstractArray{FT,3},
    𝒫::InverseGamma{FT},
    τ::FT,
    𝑇::FT,
) where {FT<:AbstractFloat}
    Δshape::FT, Δscale = sum_Δx²(x) ./ (2, 4 * τ)
    newparams = (shape(𝒫), scale(𝒫)) .+ (Δshape, Δscale) ./ 𝑇
    return rand(InverseGamma(newparams...))
end

function update_D!(s::ChainStatus, param::ExperimentalParameter)
    # s.D.value = sample_D(view_on_x(s), s.D.𝒫, param.period, s.𝑇)
    s.D.value = sample_D(s.tracks.value, s.D.prior, param.period, s.𝑇)
    return s
end
