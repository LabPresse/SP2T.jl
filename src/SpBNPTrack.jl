module SpBNPTrack

using Distributions
using SpecialFunctions
using LinearAlgebra
using GLMakie
# using StatsBase
# using Octavian
using LoopVectorization

include("datatypes.jl")
include("psfs.jl")
include("samplers.jl")
# include("priors.jl")
include("states.jl")
include("tracks.jl")
include("forward.jl")
include("plotting.jl")

function forward_main(
    params::ExperimentalParameters,
    priors::Priors;
    particle_num::Int = 3,
    bleach::Real = 0,
    diffusion::AbstractVector{Float64} = [0.05],
    emission::Float64 = 1e1,
    background::Float64 = 1e2,
)
    times = (0:params.length-1) .* params.period

    states = get_states_from_prior(
        particle_num,
        params.length,
        priors.photostate,
    )
    tracks = get_tracks_from_prior(states, params.period, priors.location, diffusion)

    integratedpsf = integrate_psf(tracks, params)

    observation = get_readout(integratedpsf, emission, background, params)

    gt = GroundTruth(
        particle_num,
        tracks,
        states,
        diffusion,
        # emission,
        background,
        times,
        integratedpsf,
    )

    video = Video(observation, params)

    return (video, gt)
end

end
