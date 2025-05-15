using SP2T
using JLD2
using Distributions

dir = "./examples/localization"

metadata = load(joinpath(dir, "metadata.jld2"), "metadata")
frames = load(joinpath(dir, "frames.jld2"), "frames")
FloatType = Float32

for i in axes(frames, 3)
    detector = EMCCD{FloatType}(
        period = metadata["period"],
        pixel_size = metadata["pixel size"],
        darkcounts = zeros(50, 50),
        cutoffs = (-Inf, Inf),
        readouts = frames[:, :, i:i],
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
        guess = 20 * metadata["period"],
        prior = Gamma(1, 10),
        proposalparam = 10,
        fixed = true,
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
        logonprob = -5,
    )

    chain = runMCMC(
        tracks = tracks,
        msd = msd,
        brightness = brightness,
        detector = detector,
        psf = psf,
        niters = 5000,
        sizelimit = 1000,
    )

    jldsave(joinpath(dir, "chain_$i.jld2"); chain)
end