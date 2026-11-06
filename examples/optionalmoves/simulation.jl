using SP2T
using Random
using JLD2

Random.seed!(5)
FloatType = Float64

metadata = Dict{String,Any}(
    "units" => ("μm", "s"),
    "numerical aperture" => 1.45,
    "refractive index" => 1.515,
    "wavelength" => 0.665,
    "period" => 1e-5,
    "pixel size" => 0.1,
    "batchsize" => 1,
    "description" => "two particles over 600 frames, for the optional moves",
)

psf = CircularGaussian{FloatType}(
    numerical_aperture = metadata["numerical aperture"],
    refractive_index = metadata["refractive index"],
    emission_wavelength = metadata["wavelength"],
    pixel_size = metadata["pixel size"],
)

detector = SPAD{FloatType}(
    period = metadata["period"],
    pixel_size = metadata["pixel size"],
    darkcounts = load("./examples/darkcounts.jld2", "darkcounts"),
    cutoffs = (0, Inf),
    readouts = zeros(UInt16, 50, 50, 600),
)

msd = 2 * 10 * metadata["period"]
ntracks = 2
tracks = Array{FloatType}(undef, 600, 2, ntracks)
# start in the central 2 μm of the 5 μm frame, so the particles stay in view
@views tracks[1, :, :] .= 1.5 .+ 2 .* rand(2, ntracks)
simulate!(tracks, msd)

brightness = 3e4 * metadata["period"]
SP2T.simulate_readouts!(
    detector,
    SP2T.getincident(tracks, brightness, detector.darkcounts, detector.pxbounds, psf),
)

jldsave("./examples/optionalmoves/metadata.jld2"; metadata = metadata)
jldsave("./examples/optionalmoves/frames.jld2"; frames = detector.readouts)
jldsave(
    "./examples/optionalmoves/groundtruth.jld2";
    tracks = tracks,
    msd = msd,
    brightness = brightness,
)
