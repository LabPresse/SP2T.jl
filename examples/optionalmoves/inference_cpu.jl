# The two optional moves, both off by default, turned on with keywords of runMCMC and
# runMCMC!:
# - msdrescale: after the MSD's Gibbs draw, rescale the MSD and stretch the on tracks with it
#   (docs/msdrescale.qmd);
# - countbrightness: after the emitter-count update, switch one emitter on or off and rescale
#   the brightness by n/n′ (docs/countbrightness.qmd; nonparametric runs only).
# Both leave the posterior unchanged and change only how the chain moves. This script fits the
# same data with the default sweep and with both moves on. Each run starts near the truth: the
# tracks 1.5 pixels off in x, the MSD and brightness at half their true values. From an
# uninformed start (all tracks at the frame center), both runs settle on 10 dim emitters
# instead, with or without the moves (see "Sampling and summaries" in MEMORY.md).
using SP2T
using JLD2
using Statistics

metadata = load("./examples/optionalmoves/metadata.jld2", "metadata")
groundtruth = load("./examples/optionalmoves/groundtruth.jld2")

FloatType = Float32

detector = SPAD{FloatType}(
    period = metadata["period"],
    pixel_size = metadata["pixel size"],
    darkcounts = load("./examples/darkcounts.jld2", "darkcounts"),
    cutoffs = (0, Inf),
    readouts = load("./examples/optionalmoves/frames.jld2", "frames"),
)

psf = CircularGaussian{FloatType}(
    numerical_aperture = metadata["numerical aperture"],
    refractive_index = metadata["refractive index"],
    emission_wavelength = metadata["wavelength"],
    pixel_size = metadata["pixel size"],
)

# Fresh random variables for each run, as runMCMC changes them in place.
function setup()
    msd = MeanSquaredDisplacement{FloatType}(
        guess = groundtruth["msd"] / 2,
        priorparams = (2, 1e-5),
    )
    brightness = Brightness{FloatType}(
        guess = groundtruth["brightness"] / 2,
        priorparams = (1, 1),
        proposalparams = (10, 1),
    )
    guess = copy(groundtruth["tracks"])
    guess[:, 1, :] .+= 1.5 * metadata["pixel size"]
    tracks = Tracks{FloatType}(
        guess = guess,
        prior = DNormal{FloatType}(
            collect(detector.framecenter),
            convert(FloatType, metadata["pixel size"]) * 10 .* [1, 1],
        ),
        max_ntracks = 10,
        scaling = √msd.value,
        logonprob = -10,
    )
    return msd, brightness, tracks
end

function summarize(name, chain, msd, tracks)
    half = length(chain.samples) ÷ 2 + 1 # the second half of the saved samples
    println(name, ":")
    println("  emitters on: ", chain.emittercounts[end], " at the end (truth: 2)")
    println("  median MSD: ", median(chain.msds[half:end]), " μm² (truth: 2e-4)")
    println("  median brightness: ", median(chain.brightnesses[half:end]), " (truth: 0.3)")
    # acceptances and proposals of the two optional moves; [0, 0] when they are off
    println("  rescaling MSD move: ", msd.counter)
    println("  joint count–brightness move: ", tracks.nemitters.counter)
end

# The default sweep: neither optional move runs.
msd, brightness, tracks = setup()
chain = runMCMC(
    tracks = tracks,
    msd = msd,
    brightness = brightness,
    detector = detector,
    psf = psf,
    niters = 2000,
    sizelimit = 1000,
);
summarize("default", chain, msd, tracks)

# Both optional moves on.
msdᵐ, brightnessᵐ, tracksᵐ = setup()
chainᵐ = runMCMC(
    tracks = tracksᵐ,
    msd = msdᵐ,
    brightness = brightnessᵐ,
    detector = detector,
    psf = psf,
    niters = 1000,
    sizelimit = 1000,
    msdrescale = true,
    countbrightness = true,
);
# runMCMC! takes the same keywords; a continued run decides again which moves to use.
runMCMC!(
    chain = chainᵐ,
    tracks = tracksᵐ,
    msd = msdᵐ,
    brightness = brightnessᵐ,
    detector = detector,
    psf = psf,
    niters = 1000,
    parametric = false,
    msdrescale = true,
    countbrightness = true,
);
summarize("msdrescale and countbrightness", chainᵐ, msdᵐ, tracksᵐ)

jldsave("./examples/optionalmoves/chain_cpu.jld2"; chain, chainᵐ)
