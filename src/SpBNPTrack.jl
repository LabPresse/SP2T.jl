module SpBNPTrack

using Distributions
using SpecialFunctions
using LinearAlgebra
using Flux
using CUDA

using GLMakie
# using StatsBase
# using Octavian
# using LoopVectorization


include("datatypes.jl")
include("psfs.jl")
include("samplers.jl")
# include("priors.jl")
include("forward.jl")
include("plotting.jl")
include("chains.jl")

function forward_main(
    params,
    priors;
    particle_num = 3,
    bleach = 0,
    diffusion = [0.05],
    emission = 1e1,
    background = 1e2,
)
    times = (0:params.length-1) .* params.period

    states = sim_states(particle_num, params.length, priors.photostate)
    tracks = sim_tracks(states, priors.location, diffusion, params.period)

    pureframe = sim_img(tracks, params.pxboundsx, params.pxboundsy, params.PSF)

    observation = sim_frames(pureframe, background, emission, params)

    gt = GroundTruth(
        particle_num,
        tracks,
        states,
        diffusion,
        # emission,
        background,
        times,
        pureframe,
    )

    video = Video(observation, params)

    return (video, gt)
end

end
