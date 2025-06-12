using SP2T
using JLD2
using Distributions
using CUDA

metadata = load("./examples/metadata.jld2", "metadata")
frames = load("./examples/frames.jld2", "frames")
darkcounts = load("./examples/darkcounts.jld2", "darkcounts")

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
    priorparams = (2, 1e-5),
)

h = Brightness{FloatType}(
    value = 4e4 * metadata["period"],
    priorparams = (1, 1),
    proposalparams = (10, 1),
)

M = NTracks{FloatType}(value = 0, limit = 10, logonprob = -10)

x = Tracks{FloatType}(
    value = CuArray{FloatType}(undef, size(frames, 3), 3, M.limit),
    prior = DNormal{FloatType}(
        CuArray([detector.framecenter..., 0]),
        CuArray{FloatType}([metadata["pixel size"] * 10, metadata["pixel size"] * 10, 0.5]),
    ),
    scaling = CUDA.fill(√msd.value, 3),
)

chain = runMCMC(
    tracks = x,
    ntracks = M,
    msd = msd,
    brightness = h,
    measurements = CuArray(frames),
    detector = detector,
    psf = psf,
    niters = 998,
    sizelimit = 1000,
);

runMCMC!(chain, x, M, msd, h, CuArray(frames), detector, psf, 100, true);

jldsave("./examples/chain_gpu.jld2"; chain)