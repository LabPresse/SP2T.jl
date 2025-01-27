using SP2T
using JLD2
using Distributions

metadata = load("./example/parametric/metadata.jld2", "metadata")

FloatType = Float32

detector = SPAD{FloatType}(
    period = metadata["period"],
    pixel_size = metadata["pixel size"],
    darkcounts = load("./example/darkcounts.jld2", "darkcounts"),
    cutoffs = (0, Inf),
    readouts = load("./example/parametric/frames.jld2", "frames"),
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
    guess = load("./example/parametric/groundtruth.jld2", "tracks"),
    presence = load("./example/parametric/groundtruth.jld2", "presence"),
    prior = DNormal{FloatType}(
        collect(detector.framecenter),
        convert(FloatType, metadata["pixel size"]) * 10 .* [1, 1],
    ),
    max_ntracks = 10,
    perturbsize = fill(√msd.value, 2),
    logonprob = -10,
)

chain = runMCMC(
    tracks = tracks,
    msd = msd,
    brightness = brightness,
    detector = detector,
    psf = psf,
    niters = 100,
    sizelimit = 1000,
    parametric = true,
);

runMCMC!(chain, tracks, msd, brightness, detector, psf, 100, true);

jldsave("./example/parametric/chain_cpu.jld2"; chain)