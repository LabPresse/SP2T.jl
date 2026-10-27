using SP2T
using JLD2
using CUDA

metadata = load("./examples/3D/metadata.jld2", "metadata")

FloatType = Float32

detector = SPAD{FloatType}(
    period = metadata["period"],
    pixel_size = metadata["pixel size"],
    darkcounts = CuArray(load("./examples/darkcounts.jld2", "darkcounts")),
    cutoffs = (0, Inf),
    readouts = CuArray(load("./examples/3D/frames.jld2", "frames")),
)

psf = CircularGaussianLorentzian{FloatType}(
    numerical_aperture = metadata["numerical aperture"],
    refractive_index = metadata["refractive index"],
    emission_wavelength = metadata["wavelength"],
    pixel_size = metadata["pixel size"],
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
    guess = CUDA.zeros(nframes, 3, 1),
    prior = DNormal{FloatType}(
        CuArray([collect(detector.framecenter)..., 0]),
        CuArray{FloatType}(fill(metadata["pixel size"] * 10, 3)),
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
    niters = 9998,
    sizelimit = 1000,
);

runMCMC!(chain, tracks, msd, brightness, detector, psf, 1000, true);

jldsave("./examples/3D/chain_gpu.jld2"; chain)