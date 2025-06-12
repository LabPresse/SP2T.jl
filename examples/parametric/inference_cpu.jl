using SP2T
using JLD2

metadata = load("./examples/parametric/metadata.jld2", "metadata")

FloatType = Float32

detector = SPAD{FloatType}(
    period = metadata["period"],
    pixel_size = metadata["pixel size"],
    darkcounts = load("./examples/darkcounts.jld2", "darkcounts"),
    cutoffs = (0, Inf),
    readouts = load("./examples/parametric/frames.jld2", "frames"),
)

psf = CircularGaussian{FloatType}(
    numerical_aperture = metadata["numerical aperture"],
    refractive_index = metadata["refractive index"],
    emission_wavelength = metadata["wavelength"],
    pixels_size = metadata["pixel size"],
)

msd = MeanSquaredDisplacement{FloatType}(
    guess = 2 * 1 * metadata["period"],
    priorparams = (2, 1e-5),
)

brightness = Brightness{FloatType}(
    guess = 1e4 * metadata["period"],
    priorparams = (1, 1),
    proposalparams = (10, 1),
)

nframes = size(detector.readouts, 3)
tracks = Tracks{FloatType}(
    guess = load("./examples/parametric/groundtruth.jld2", "tracks"),
    presence = load("./examples/parametric/groundtruth.jld2", "presence"),
    prior = DNormal{FloatType}(
        collect(detector.framecenter),
        convert(FloatType, metadata["pixel size"]) * 10 .* [1, 1],
    ),
    max_ntracks = 10,
    scaling = √msd.value,
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

jldsave("./examples/parametric/chain_cpu.jld2"; chain)