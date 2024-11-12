mutable struct MeanSquaredDisplacement{T,P}
    value::T
    prior::P
end

MeanSquaredDisplacement{T}(
    value::Real,
    prior::ContinuousUnivariateDistribution,
) where {T<:AbstractFloat} = MeanSquaredDisplacement(
    convert(T, value),
    unionalltypeof(prior)(convert.(T, params(prior))...),
)

MeanSquaredDisplacement{T}(;
    value::Real,
    prior::ContinuousUnivariateDistribution,
) where {T<:AbstractFloat} = MeanSquaredDisplacement{T}(value, prior)

logprior(msd::MeanSquaredDisplacement{T,P}) where {T,P<:InverseGamma{T}} =
    -(shape(msd.prior) + 1) * log(msd.value) - scale(msd.prior) / msd.value

logprior_norm(msd::MeanSquaredDisplacement{T,P}) where {T,P} = logpdf(msd.prior, msd.value)

function _rand(prior::P, Δparams::NTuple{2,T}) where {T,P}
    postparams = params(prior) .+ Δparams
    return rand(P(postparams...))
end

# function sample!(msd::MeanSquaredDisplacement{T}, Δx²::AbstractArray{T,3}, 𝑇::T) where {T}
#     Δparams = (length(Δx²), sum(vec(Δx²))) ./ (2 * 𝑇)
#     msd.value = _rand(msd.prior, Δparams)
#     return msd
# end

# function update!(
#     msd::MeanSquaredDisplacement{T},
#     trackᵥ::AbstractArray{T,3},
#     displacements::AbstractArray{T,3},
#     𝑇::T,
# ) where {T}
#     diff²!(displacements, trackᵥ)
#     return sample!(msd, displacements, 𝑇)
# end

function update!(
    msd::MeanSquaredDisplacement{T},
    # trackᵥ::AbstractArray{T,3},
    displacement²::AbstractArray{T,3},
    𝑇::T,
) where {T}
    # diff²!(displacements, trackᵥ)
    Δparams = (length(displacement²), sum(vec(displacement²))) ./ (2 * 𝑇)
    msd.value = _rand(msd.prior, Δparams)
    return msd
    # return sample!(msd, displacement², 𝑇)
end