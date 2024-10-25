using SP2T
using JLD2
using Distributions
using CUDA

metadata = load("./example/metadata.jld2", "metadata")
frames = load("./example/frames.jld2", "frames")
darkcounts = load("./example/darkcounts.jld2", "darkcounts")

FloatType = Float32

data = Data{FloatType}(
    CuArray(frames),
    metadata["period"],
    metadata["pixel size"],
    CuArray(darkcounts),
    (eps(), Inf),
    metadata["numerical aperture"],
    metadata["refractive index"],
    metadata["wavelength"],
)

msd = MeanSquaredDisplacement{FloatType}(
    value = 2 * 1 * metadata["period"],
    prior = InverseGamma(2, 1e-5),
)

h = Brightness{FloatType}(
    value = 4e4 * metadata["period"],
    prior = Gamma(1, 1),
    proposalparam = 1,
)

M = NEmitters{FloatType}(value = 0, limit = 10, logonprob = convert(FloatType, -10))

x = Tracks(
    value = CuArray{FloatType}(undef, data.nframes, 3, M.limit),
    prior = Normal₃(
        CuArray([data.framecenter..., 0]),
        CuArray{FloatType}([metadata["pixel size"] * 10, metadata["pixel size"] * 10, 0.5]),
    ),
    perturbsize = CUDA.fill(√msd.value, 3),
)

groundtruth = load("./example/groundtruth.jld2")
copyto!(x.value, groundtruth["tracks"])
M.value = 1

chain = runMCMC(
    tracks = x,
    nemitters = M,
    diffusivity = msd,
    brightness = h,
    data = data,
    niters = 5,
    sizelimit = 1000,
);

runMCMC!(chain, x, M, msd, h, data, 100, true);

jldsave("./example/chain_gpu.jld2"; chain)

# visualize(data, groundtruth, chain, burn_in = 200)