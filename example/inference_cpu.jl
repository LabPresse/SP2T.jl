using SP2T
using JLD2
using Random

Random.seed!(1)

data = load("./example/data.jld2", "data")
groundtruth = load("./example/groundtruth.jld2", "groundtruth")

FloatType = typeof(data.period)

D = Diffusivity(value = 2, priorparams = (2, 0.1), scale = data.period)

h = Brightness(value = 5e5, priorparams = (1, 1), proposalparam = 1, scale = data.period)

M = NEmitters(value = 0, maxcount = 10, onprob = oftype(data.period, 0.1))

x = BrownianTracks(
    value = Array{FloatType}(undef, 3, maxcount(M), size(data.frames, 3)),
    prior = Normal₃(
        [data.pxboundsx[end] / 2, data.pxboundsy[end] / 2, 0],
        [data.pxboundsx[end] / 4, data.pxboundsy[end] / 4, convert(FloatType, 0.5)],
    ),
    perturbsize = fill(sqrt(2 * D.value), 3),
)

chain = runMCMC(
    tracks = x,
    nemitters = M,
    diffusivity = D,
    brightness = h,
    data = data,
    niters = 999,
    sizelimit = 1000,
);

jldsave("./example/chain_cpu.jld2"; chain)

# visualize(data, groundtruth, chain, burn_in = 200)