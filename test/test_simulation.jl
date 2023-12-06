using SpBNPTrack
using Random

using JLD2

Random.seed!(2)
FloatType = Float32

param = ExperimentalParameter(
    FloatType,
    units = ("μm", "s"),
    length = 256,
    period = 0.0033,
    # exposure = 0.003,
    pxsize = 0.133,
    darkcounts = fill(1e-3, (50, 50)),
    NA = 1.45,
    nᵣ = 1.515,
    λ = 0.665,
)

groundtruth = simulate_sample(
    param = param,
    emitter_number = 2,
    diffusion_coefficient = 0.05,
    emission_rate = 200,
)

video = Video(param, groundtruth)
visualize(video, groundtruth)

jldsave("example_video.jld2"; video)
jldsave("example_groundtruth.jld2"; groundtruth)