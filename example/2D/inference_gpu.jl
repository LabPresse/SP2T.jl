using SP2T
using JLD2
using Distributions
using CUDA

metadata = load("./example/2D/metadata.jld2", "metadata")

FloatType = Float32

detector = SPAD{FloatType}(
    period = metadata["period"],
    pixel_size = metadata["pixel size"],
    darkcounts = CuArray(load("./example/darkcounts.jld2", "darkcounts")),
    cutoffs = (0, Inf),
    readouts = CuArray(load("./example/2D/frames.jld2", "frames")),
)

psf = CircularGaussian{FloatType}(
    numerical_aperture = metadata["numerical aperture"],
    refractive_index = metadata["refractive index"],
    emission_wavelength = metadata["wavelength"],
    pixels_size = metadata["pixel size"],
)

msd = MeanSquaredDisplacement{FloatType}(
    guess = 2 * 1 * metadata["period"],
    prior = InverseGamma(2, 1e-5),
)

brightness = Brightness{FloatType}(
    guess = 1e4 * metadata["period"],
    prior = Gamma(1, 1),
    proposalparam = 1,
)

nframes = size(detector.readouts, 3)
tracks = Tracks{FloatType}(
    guess = CUDA.zeros(nframes, 2, 1),
    prior = DNormal{FloatType}(
        CuArray(collect(detector.framecenter)),
        CuArray{FloatType}([metadata["pixel size"] * 10, metadata["pixel size"] * 10]),
    ),
    max_ntracks = 10,
    perturbsize = CUDA.fill(√msd.value, 2),
    logonprob = -10,
)

chain = runMCMC(
    tracks = tracks,
    msd = msd,
    brightness = brightness,
    detector = detector,
    psf = psf,
    niters = 998,
    sizelimit = 1000,
);

runMCMC!(chain, tracks, msd, brightness, detector, psf, 1000, true);

jldsave("./example/2D/chain_gpu.jld2"; chain)