using SP2T
using JLD2
using CUDA
using Random

Random.seed!(1)

data = load("./example/data.jld2", "data")
groundtruth = load("./example/groundtruth.jld2", "groundtruth")

data = Data(
    CuArray(data.frames),
    data.batchsize,
    data.period,
    CuArray(data.pxboundsx),
    CuArray(data.pxboundsy),
    CuArray(data.darkcounts),
    data.PSF,
)
FloatType = typeof(data.period)

D = Diffusivity(value = 2, priorparams = (2, 0.1), scale = data.period)

h = Brightness(value = 5e5, priorparams = (1, 1), proposalparam = 1, scale = data.period)

M = NEmitters(value = 0, maxcount = 10, onprob = oftype(data.period, 0.1))

CUDA.@allowscalar prior = Normal₃(
    CuArray([maximum(data.pxboundsx) / 2, maximum(data.pxboundsy) / 2, 0]),
    CuArray([
        maximum(data.pxboundsx) / 4,
        maximum(data.pxboundsy) / 4,
        convert(FloatType, 0.5),
    ]),
)

x = BrownianTracks(
    value = CuArray{FloatType}(undef, 3, maxcount(M), size(data.frames, 3)),
    prior = prior,
    perturbsize = CUDA.fill(sqrt(2 * D.value), 3),
)

# x.value[:, 1:1, :] .= CuArray(groundtruth.tracks)

chain = runMCMC(
    tracks = x,
    nemitters = M,
    diffusivity = D,
    brightness = h,
    data = data,
    niters = 999,
    sizelimit = 1000,
);

# @profview runMCMC!(chain, x, M, D, h, data, 10);

jldsave("./example/chain_gpu.jld2"; chain)

# visualize(data, groundtruth, chain, burn_in = 200)