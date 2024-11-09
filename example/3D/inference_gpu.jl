using SP2T
using JLD2
using Distributions
using CUDA

metadata = load("./example/metadata.jld2", "metadata")
frames = load("./example/frames.jld2", "frames")
darkcounts = load("./example/darkcounts.jld2", "darkcounts")

FloatType = Float32

detector = SPAD{FloatType}(
    metadata["period"],
    metadata["pixel size"],
    CuArray{FloatType}(darkcounts),
    (eps(), Inf),
    size(frames, 3),
)

psf = CircularGaussianLorentzian{FloatType}(
    metadata["numerical aperture"],
    metadata["refractive index"],
    metadata["wavelength"],
    metadata["pixel size"],
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

M = NEmitters{FloatType}(value = 0, limit = 10, logonprob = -10)

x = Tracks{FloatType}(
    value = CuArray{FloatType}(undef, size(frames, 3), 3, M.limit),
    prior = DNormal{FloatType}(
        CuArray([detector.framecenter..., 0]),
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
    msd = msd,
    brightness = h,
    measurements = CuArray(frames),
    detector = detector,
    psf = psf,
    niters = 998,
    sizelimit = 1000,
);

runMCMC!(chain, x, M, msd, h, frames, detector, psf, 100, true);

jldsave("./example/chain_gpu.jld2"; chain)

# visualize(data, groundtruth, chain, burn_in = 200)