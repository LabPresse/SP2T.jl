using SP2T
using JLD2
using Distributions
using CUDA

metadata = load("./example/2D/metadata.jld2", "metadata")
frames = load("./example/2D/frames.jld2", "frames")
darkcounts = load("./example/2D/darkcounts.jld2", "darkcounts")

FloatType = Float32

detector = SPAD{FloatType}(
    metadata["period"],
    metadata["pixel size"],
    CuArray{FloatType}(darkcounts),
    (eps(), Inf),
    size(frames, 3),
)

psf = CircularGaussian{FloatType}(
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
    value = 1e4 * metadata["period"],
    prior = Gamma(1, 1),
    proposalparam = 1,
)

x = Tracks{FloatType}(
    guess = CUDA.zeros(FloatType, size(frames, 3), 2, 1),
    prior = DNormal{FloatType}(
        CuArray(collect(detector.framecenter)),
        CuArray{FloatType}([metadata["pixel size"] * 10, metadata["pixel size"] * 10]),
    ),
    max_ntracks = 10,
    perturbsize = CUDA.fill(√msd.value, 2),
    logonprob = -10,
)

chain = runMCMC(
    tracks = x,
    msd = msd,
    brightness = h,
    measurements = CuArray(frames),
    detector = detector,
    psf = psf,
    niters = 998,
    sizelimit = 1000,
);

runMCMC!(chain, x, msd, h, CuArray(frames), detector, psf, 50_000, true);

jldsave("./example/2D/chain_gpu.jld2"; chain)