# utility.jl: maps functions whose inputs contain multiple structs to functions whose inputs are scalars and arrays

# function shuffletracks!(tracks::BrownianTracks, nemitters::NEmitters)
#     shuffletracks!(tracks.x, nemitters.M)
#     return tracks
# end

# function set_lnℒ!(n::NEmitters, x, h, W, ep::ExperimentalParameters, cp::ChainParameters)
#     set_lnℒ!(n, cp.𝐔, W, x, h, ep.darkcounts, ep.pxboundsx, ep.pxboundsy, ep.PSF, cp.temp)
#     return n
# end

# ontracks(bt::BrownianTracks, n::NEmitters) = view(bt.x, :, 1:n.M, :)

# displacements(bt::BrownianTracks, n::NEmitters) =
#     view(bt.Δx², :, 1:n.M, :), view(bt.Δxᵖ², :, 1:n.M, :)

# function propose!(bt::BrownianTracks, n::NEmitters)
#     @views propose!(bt.xᵖ[:, 1:n.M, :], bt.x[:, 1:n.M, :], bt.perturbsize)
#     return view(bt.xᵖ, :, 1:n.M, :)
# end

# function propose!(
#     bt::BrownianTracks,
#     n::NEmitters,
#     b::Brightness,
#     ep::ExperimentalParameters,
#     cp::ChainParameters,
# )
#     @views x, xᵖ = bt.x[:, 1:n.M, :], bt.xᵖ[:, 1:n.M, :]
#     propose!(xᵖ, x, bt.perturbsize)
#     get_px_intensity!(cp.𝐔, x, b, ep)
#     get_px_intensity!(cp.𝐔ᵖ, xᵖ, b, ep)
#     return bt
# end

# set_ΔlogL!(
#     tracks::BrownianTracks,
#     frames,
#     xᵖ,
#     brightness::Brightness,
#     expparams::ExperimentalParameters,
#     chainparams::ChainParameters,
# ) = set_ΔlogL!(
#     tracks.logacceptance,
#     frames,
#     chainparams.𝐔,
#     chainparams.𝐔ᵖ,
#     xᵖ,
#     brightness.h,
#     expparams.darkcounts,
#     expparams.pxboundsx,
#     expparams.pxboundsy,
#     expparams.PSF,
#     chainparams.temperature,
#     chainparams.temp,
# )



# function simulate!(tracks::BrownianTracks, diffusivity::Diffusivity, init)
#     x, Δx² = tracks.x, tracks.Δx²
#     μ, σ = init, 0
#     simulate!(x, Δx², μ, σ, diffusivity.D)
#     return tracks
# end