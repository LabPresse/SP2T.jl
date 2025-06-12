using SP2T
using JLD2
using CUDA

metadata = load("./examples/EMCCD/metadata.jld2", "metadata")

FloatType = Float32

detector = EMCCD{FloatType}(
    period = metadata["period"],
    pixel_size = metadata["pixel size"],
    darkcounts = CUDA.zeros(50, 50) .+ eps(),
    cutoffs = (0, Inf),
    readouts = CuArray(load("./examples/EMCCD/frames.jld2", "frames")),
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
    guess = CUDA.zeros(nframes, 2, 1),
    prior = DNormal{FloatType}(
        CuArray(collect(detector.framecenter)),
        CuArray{FloatType}([metadata["pixel size"] * 10, metadata["pixel size"] * 10]),
    ),
    max_ntracks = 10,
    scaling = CUDA.fill(√msd.value, 2),
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

jldsave("./examples/EMCCD/chain_gpu.jld2"; chain)