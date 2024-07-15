_logpdf(n::Normal₃, x) = sum(@. -(x - n.μ) / (2 * n.σ^2))

_logpdf(x::BrownianTracks, D, aux::AuxiliaryVariables) =
    -log(D) * length(aux.Δx²) / 2 - sum(vec(aux.Δx²)) / (4 * D) -
    _logpdf(x.prior, view(x.value, 1, :, :))

_logpdf(D::Diffusivity) = -(D.πparams[1] + 1) * log(D.value) - D.πparams[2] / D.value

_logpdf(M::NEmitters) = M.logπ[M.value+1]

_logpdf(h::Brightness) = (h.priorparams[1] - 1) * log(h.value) - h.value / h.priorparams[2]

function log𝒫logℒ(
    x::BrownianTracks,
    M::NEmitters,
    D::Diffusivity,
    h::Brightness,
    data::Data,
    aux::AuxiliaryVariables,
)
    diff²!(aux.Δx², x.value)
    pxcounts!(aux.U, view(x.value, :, :, 1:M.value), h.value, data)
    logℒ = _logℒ(data.frames, aux.U, data.mask, aux.Sᵤ)
    log𝒫 = logℒ + _logpdf(x, D.value, aux) + _logpdf(D) + _logpdf(M) + _logpdf(h)
    return log𝒫, logℒ
end