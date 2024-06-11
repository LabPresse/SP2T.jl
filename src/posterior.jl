_logpdf(n::Normal₃, x) = sum(@. -(x - n.μ) / (2 * n.σ^2))

_logpdf(x::BrownianTracks, D, aux::AuxiliaryVariables) =
    -log(D) * length(aux.Δx²) / 2 - sum(aux.Δx²) / (4 * D) -
    _logpdf(x.prior, view(x.value, :, :, 1))

_logpdf(D::Diffusivity) = -(D.πparams[1] + 1) * log(D.value) - D.πparams[2] / D.value

_logpdf(M::NEmitters) = M.logπ[M.value+1]

_logpdf(h::Brightness) = (h.priorparams[1] - 1) * log(h.value) - h.value / h.priorparams[2]

# function setlog𝒫!(
#     chain::Chain,
#     tracks::BrownianTracks,
#     diffusivity::Diffusivity,
#     nemitters::NEmitters,
#     brightness::Brightness,
#     frames,
#     expparams::ExperimentalParameters,
#     aux::AuxiliaryVariables,
# )
#     setΔx²!(aux, tracks.value)
#     pxcounts!(
#         aux.U,
#         view(tracks.value, :, 1:nemitters.value, :),
#         brightness.value,
#         expparams.darkcounts,
#         expparams.pxboundsx,
#         expparams.pxboundsy,
#         expparams.PSF,
#     )
#     chain.logℒ = logℒ(frames, aux.U, aux.ΔU)
#     chain.log𝒫 =
#         chain.logℒ +
#         _logpdf(tracks, diffusivity.value, aux) +
#         _logpdf(diffusivity) +
#         _logpdf(nemitters) +
#         _logpdf(brightness)
#     return chain
# end

function log𝒫logℒ(
    x::BrownianTracks,
    M::NEmitters,
    D::Diffusivity,
    h::Brightness,
    data::Data,
    aux::AuxiliaryVariables,
)
    diff²!(aux, x.value)
    pxcounts!(aux.U, view(x.value, :, 1:M.value, :), h.value, data)
    logℒ = _logℒ(data.frames, aux.U, aux.ΔU)
    log𝒫 = logℒ + _logpdf(x, D.value, aux) + _logpdf(D) + _logpdf(M) + _logpdf(h)
    return log𝒫, logℒ
end