using SP2T
using JLD2

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

params = ExperimentalParameters(
    FloatType,
    metadata["period"],
    metadata["pixel size"],
    load("./data/beads/beads_darkcounts.jld2", "darkcounts"),
    11.14 * 12 / 1000,
    1.2 * 100 / 1000,
)

frames = load("./data/example_frames_v2.jld2", "frames")
groundtruth = load("./data/example_groundtruth_v2.jld2", "groundtruth")

D = Diffusivity(value = 2, priorparams = (2, 0.1), scale = params.period)

h = Brightness(value = 2e6, priorparams = (1, 1), proposalparam = 1, scale = params.period)

M = NEmitters(value = 0, maxcount = 10, onprob = oftype(params.period, 0.1))

x = BrownianTracks(
    value = Array{FloatType}(undef, 3, maxcount(M), size(frames, 3)),
    prior = Normal₃(
        [params.pxboundsx[end] / 2, params.pxboundsy[end] / 2, 0],
        [params.pxboundsx[end] / 4, params.pxboundsy[end] / 4, convert(FloatType, 0.5)],
    ),
    perturbsize = fill(sqrt(2 * D.value), 3),
)

# chainparams = ChainParameters(x = x.x, frames = frames, sizelimit = 1000)

chain = runMCMC(
    tracks = x,
    nemitters = M,
    diffusivity = D,
    brightness = h,
    frames = frames,
    params = params,
    niters = 100,
);

# jldsave("example_chain2.jld2"; chain)

visualize(video, groundtruth, chain, burn_in = 200)