_logpdf(n::Normal₃, x::AbstractArray) = sum(vec(@. -(x - n.μ) / (2 * n.σ^2)))

function _logpdf(x::BrownianTracks, D::T, Δx²::AbstractArray{T}) where {T}
    diff²!(Δx², x.value)
    -log(D) * length(Δx²) / 2 - sum(vec(Δx²)) / (4 * D) -
    _logpdf(x.prior, view(x.value, 1, :, :))
end

_logpdf(D::Diffusivity) = -(D.πparams[1] + 1) * log(D.value) - D.πparams[2] / D.value

_logpdf(M::NEmitters) = M.logπ[M.value+1]

_logpdf(h::Brightness) = (h.priorparams[1] - 1) * log(h.value) - h.value / h.priorparams[2]

function log𝒫logℒ(
    x::BrownianTracks,
    M::NEmitters,
    D::Diffusivity{T},
    h::Brightness{T},
    data::Data{T},
    A::AuxiliaryVariables{T},
) where {T}
    pxcounts!(A.U, view(x.value, :, :, 1:M.value), h.value, data)
    logℒ1 = logℒ(data, A)
    log𝒫1 = logℒ1 + _logpdf(x, D.value, A.Δ𝐱²) + _logpdf(D) + _logpdf(M) + _logpdf(h)
    return log𝒫1, logℒ1
end