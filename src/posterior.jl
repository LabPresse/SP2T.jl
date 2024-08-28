_logπ(ℕ::Normal₃, x::AbstractArray) = sum(vec(@. -(x - ℕ.μ) / (2 * ℕ.σ^2)))

function _logπ(x::Tracks, M::Integer, D::T, Δx²::AbstractArray{T,3}) where {T}
    xᵒⁿ = view(x.value, :, :, 1:M)
    Δxᵒⁿ² = view(Δx², :, :, 1:M)
    diff²!(Δxᵒⁿ², xᵒⁿ)
    -log(D) * length(Δxᵒⁿ²) / 2 - sum(vec(Δxᵒⁿ²)) / (4 * D) -
    _logπ(x.prior, view(xᵒⁿ, 1, :, :))
end

_logπ(D::Diffusivity) = -(D.πparams[1] + 1) * log(D.value) - D.πparams[2] / D.value

_logπ(M::NEmitters) = M.logπ[M.value+1]

_logπ(h::Brightness) = (h.priorparams[1] - 1) * log(h.value) - h.value / h.priorparams[2]

function log𝒫logℒ(
    x::Tracks,
    M::NEmitters,
    D::Diffusivity,
    h::Brightness,
    data::Data,
    A::AuxiliaryVariables,
)
    pxcounts!(A.U, view(x.value, :, :, 1:M.value), h.value, data)
    logℒ1 = logℒ(data, A)
    log𝒫1 = logℒ1 + _logπ(x, M.value, D.value, A.Δ𝐱²) + _logπ(D) + _logπ(M) + _logπ(h)
    return log𝒫1, logℒ1
end