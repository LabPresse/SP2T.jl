using SP2T
using JLD2
using Distributions

metadata = load("./examples/EMCCD/metadata.jld2", "metadata")

FloatType = Float32

detector = EMCCD{FloatType}(
    period = metadata["period"],
    pixel_size = metadata["pixel size"],
    darkcounts = zeros(50, 50) .+ eps(),
    cutoffs = (0, Inf),
    readouts = load("./examples/EMCCD/frames.jld2", "frames"),
    offset = 100,
    gain = 100,
    variance = 2,
)

psf = CircularGaussian{FloatType}(
    numerical_aperture = metadata["numerical aperture"],
    refractive_index = metadata["refractive index"],
    emission_wavelength = metadata["wavelength"],
    pixels_size = metadata["pixel size"],
)

msd = MeanSquaredDisplacement{FloatType}(
    guess = 2 * 0.2 * metadata["period"],
    priorparams = (2, 1e-5),
)

brightness = Brightness{FloatType}(
    guess = 10 * metadata["period"],
    priorparams = (1, 10),
    proposalparams = (10, 1),
)

nframes = size(detector.readouts, 3)
tracks = Tracks{FloatType}(
    guess = zeros(nframes, 2, 1),
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
    niters = 2000,
    sizelimit = 1000,
);

runMCMC!(chain, tracks, msd, brightness, detector, psf, 1000, true);

jldsave("./examples/EMCCD/chain_cpu.jld2"; chain)