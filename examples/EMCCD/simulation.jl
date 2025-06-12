using SP2T
using Random
using JLD2

Random.seed!(1)
FloatType = Float64

metadata = Dict{String,Any}(
    "units" => ("μm", "s"),
    "numerical aperture" => 1.45,
    "refractive index" => 1.515,
    "wavelength" => 0.665,
    "period" => 3.3e-2,
    "pixel size" => 0.133,
    "description" => "example run",
)

psf = CircularGaussian{FloatType}(
    numerical_aperture = metadata["numerical aperture"],
    refractive_index = metadata["refractive index"],
    emission_wavelength = metadata["wavelength"],
    pixels_size = metadata["pixel size"],
)

detector = EMCCD{FloatType}(
    period = metadata["period"],
    pixel_size = metadata["pixel size"],
    darkcounts = zeros(50, 50),
    cutoffs = (0, Inf),
    readouts = zeros(UInt16, 50, 50, 5),
    offset = 100,
    gain = 100,
    variance = 2,
)

msd = 2 * 0.1 * metadata["period"]
ntracks = 2
tracks = Array{FloatType}(undef, 5, 2, ntracks)
@views rand!(tracks[1, :, :]) .*= metadata["pixel size"] .* collect(size(detector))
simulate!(tracks, msd)

brightness = 20 * metadata["period"]
SP2T.simulate_readouts!(
    detector,
    SP2T.getincident(tracks, brightness, detector.darkcounts, detector.pxbounds, psf),
)

jldsave("./examples/EMCCD/metadata.jld2"; metadata = metadata)
jldsave("./examples/EMCCD/frames.jld2"; frames = detector.readouts)
jldsave("./examples/EMCCD/groundtruth.jld2"; tracks = tracks, msd = msd)