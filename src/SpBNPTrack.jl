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
include("photostates.jl")
include("tracks.jl")
include("forward.jl")
include("plotting.jl")

function forward_main(
    params::ExperimentalParameters,
    priors::Priors;
    particle_num::Int = 3,
    bleach::Real = 0,
    diffusion::Float64 = 0.05,
    emission::Float64 = 1e4,
    background::Float64 = 1e5,
    gain::Float64 = 0.3,
    length_per_exposure::Int = 100,
)
    times = get_times(params.period, params.exposure, length_per_exposure, params.length)

    photostates = photostates_from_prior(
        particle_num,
        length_per_exposure * params.length,
        priors.photostate,
    )
    tracks = tracks_from_prior(particle_num, times, priors.location, diffusion)
    
    weights = get_weights(length_per_exposure)
    integratedpsf = integrate_psf(tracks, params, weights)

    observation = get_readout(integratedpsf, emission, background, gain, params)

    gt = GroundTruth(
        particle_num,
        tracks,
        photostates,
        diffusion,
        # emission,
        background,
        gain,
        times,
        length_per_exposure,
        integratedpsf,
    )

    video = Video(observation, params)

    return (video, gt)
end

end
