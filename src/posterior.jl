_logπ(ℕ::Normal₃{T}, x::AbstractArray{T}) where {T} = sum(vec(@. -(x - ℕ.μ) / (2 * ℕ.σ^2)))

function logprior(x::Tracks{T}, M::Integer, msd::T, Δx²::AbstractArray{T,3}) where {T}
    xᵒⁿ = view(x.value, :, :, 1:M)
    Δxᵒⁿ² = view(Δx², :, :, 1:M)
    diff²!(Δxᵒⁿ², xᵒⁿ)
    return -(log(msd) * length(Δxᵒⁿ²) + sum(vec(Δxᵒⁿ²)) / msd) / 2 -
           _logπ(x.prior, view(xᵒⁿ, 1, :, :))
end

logprior(msd::MeanSquaredDisplacement{T,P}) where {T,P<:InverseGamma{T}} =
    -(shape(msd.prior) + 1) * log(msd.value) - scale(msd.prior) / msd.value

logprior_norm(msd::MeanSquaredDisplacement{T,P}) where {T,P} = logpdf(msd.prior, msd.value)

logprior(M::NEmitters) = M.logprior[M.value+1]

logprior(h::Brightness{T,P}) where {T,P<:Gamma{T}} =
    (shape(h.prior) - 1) * log(h.value) - h.value / scale(h.prior)

logprior_norm(h::Brightness{T,P}) where {T,P} = logpdf(h.prior, h.value)

function log𝒫logℒ(
    x::Tracks{T},
    M::NEmitters{T},
    msd::MeanSquaredDisplacement{T},
    h::Brightness{T},
    data::Data{T},
    A::AuxiliaryVariables{T},
) where {T}
    pxcounts!(A.U, view(x.value, :, :, 1:M.value), h.value, data)
    logℒ1 = logℒ(data, A)
    log𝒫1 = logℒ1 + logprior(x, M.value, msd.value, A.Δ𝐱²) + logprior(msd) + logprior(M)
    return log𝒫1, logℒ1
end