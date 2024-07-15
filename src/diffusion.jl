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

function setparams!(D::Diffusivity{T}, Δx²::AbstractArray{T}, 𝑇::T) where {T}
    D.params = D.πparams .+ (length(Δx²), sum(vec(Δx²)) / 2) ./ (2 * 𝑇)
    return D
end

function sample!(D::Diffusivity)
    D.value = rand(InverseGamma(D.params...))
    return D
end
