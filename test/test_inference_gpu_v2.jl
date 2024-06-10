using SP2T
using JLD2
using CUDA
using Random

Random.seed!(1)

FloatType = Float32

metadata = Dict{String,Any}(
    "units" => ("μm", "s"),
    "numerical aperture" => 1.45,
    "refractive index" => 1.515,
    "wavelength" => 0.665,
    "period" => 3e-6,
    "pixel size" => 0.1,
)

# expparams = ExperimentalParameters(
#     FloatType,
#     metadata["period"],
#     metadata["pixel size"],
#     load("./data/beads/beads_darkcounts.jld2", "darkcounts"),
#     metadata["numerical aperture"],
#     metadata["refractive index"],
#     metadata["wavelength"],
# )

data = Data(
    FloatType,
    CuArray(load("./data/example_frames_v2.jld2", "frames")),
    metadata["period"],
    metadata["pixel size"],
    CuArray(load("./data/beads/darkcounts1.jld2", "darkcounts")),
    11.14 * 12 / 1000,
    1.2 * 100 / 1000,
)

# frames = load("./data/example_frames_v2.jld2", "frames")
groundtruth = load("./data/example_groundtruth_v2.jld2", "groundtruth")

D = Diffusivity(value = 2, priorparams = (2, 0.1), scale = data.period)

h = Brightness(value = 2e6, priorparams = (1, 1), proposalparam = 1, scale = data.period)

M = NEmitters(value = 1, maxcount = 10, onprob = oftype(data.period, 0.1))

CUDA.@allowscalar prior = Normal₃(
    CuArray([maximum(data.pxboundsx) / 2, maximum(data.pxboundsy) / 2, 0]),
    CuArray([
        maximum(data.pxboundsx) / 4,
        maximum(data.pxboundsy) / 4,
        convert(FloatType, 0.5),
    ]),
)

x = BrownianTracks(
    value = CuArray{FloatType}(undef, 3, maxcount(M), size(frames, 3)),
    prior = prior,
    perturbsize = CUDA.fill(sqrt(2 * D.value), 3),
)

x.value[:, 1:1, :] .= CuArray(groundtruth.tracks)

chain = runMCMC(
    tracks = x,
    nemitters = M,
    diffusivity = D,
    brightness = h,
    # frames = CuArray(frames),
    data = data,
    niters = 10_000,
    sizelimit = 1000,
);

jldsave("example_samples_v2.jld2"; chain)

visualize(data, groundtruth, chain, burn_in = 200)