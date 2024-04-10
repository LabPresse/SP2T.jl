using SP2T
using Random

using JLD2

Random.seed!(9)
FloatType = Float32

metadata = Dict{String,Any}(
    "units" => ("μm", "s"),
    "numerical aperture" => 1.45,
    "refractive index" => 1.515,
    "wavelength" => 0.665,
)

param = ExperimentalParameter(
    FloatType,
    units = ("μm", "s"),
    length = 255 * 3,
    period = 0.0033,
    pxsize = 0.133,
    darkcounts = fill(1e-3, (50, 50)),
    NA = metadata["numerical aperture"],
    nᵣ = metadata["refractive index"],
    λ = metadata["wavelength"],
)

groundtruth = simulate_sample(
    param = param,
    emitter_number = 2,
    diffusion_coefficient = 0.5,
    emission_rate = 200,
)

video = Video(param, groundtruth)
visualize(video, groundtruth)

jldsave("example_video_fast.jld2"; video)
jldsave("example_groundtruth_fast.jld2"; groundtruth)