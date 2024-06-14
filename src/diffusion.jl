mutable struct Diffusivity{T}
    value::T
    params::NTuple{2,T}
    πparams::NTuple{2,T}
end

Diffusivity(D, priorparams) = Diffusivity(D, priorparams, priorparams)

Diffusivity(; value, priorparams, scale::T) where {T} = Diffusivity(
    convert(T, value * scale),
    convert.(T, (priorparams[1], priorparams[2] * scale)),
)

function setparams!(D::Diffusivity{T}, Δx²::AbstractArray{T}, 𝑇::Union{T,Int}) where {T}
    D.params = D.πparams .+ (length(Δx²), sum(Δx²) / 2) ./ (2 * 𝑇)
    return D
end

function sample!(D::Diffusivity)
    D.value = rand(InverseGamma(D.params...))
    return D
end

# function sum_Δx²(x)
#     Δx² = diff(x, dims = 3) .^ 2
#     return length(Δx²), sum(Δx²)
# end

# function sample_D(x::AbstractArray{T,3}, 𝒫::InverseGamma{T}, τ::T, 𝑇::T) where {T}
#     Δshape::T, Δscale = sum_Δx²(x) ./ (2, 4 * τ)
#     newparams = (shape(𝒫), scale(𝒫)) .+ (Δshape, Δscale) ./ 𝑇
#     return rand(InverseGamma(newparams...))
# end

# function update_diffusivity!(s::ChainStatus, param::ExperimentalParameter)
#     # s.D.value = sample_D(view_on_x(s), s.D.𝒫, param.period, s.𝑇)
#     s.diffusivity.value =
#         sample_D(s.tracks.value, s.diffusivity.prior, param.period, s.temperature)
#     return s
# end
