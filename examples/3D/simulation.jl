using SP2T
using Random
using JLD2

Random.seed!(9)
FloatType = Float64

metadata = Dict{String,Any}(
    "units" => ("μm", "s"),
    "numerical aperture" => 1.45,
    "refractive index" => 1.515,
    "wavelength" => 0.665,
    "period" => 1e-5,
    "pixel size" => 0.1,
    "batchsize" => 1,
    "description" => "example run",
)

psf = CircularGaussianLorentzian{FloatType}(
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
    readouts = zeros(UInt16, 50, 50, 2550),
)

msd = 2 * 10 * metadata["period"]
ntracks = 2
tracks = Array{FloatType}(undef, 2550, 3, ntracks)
@views rand!(tracks[1, :, :]) .*=
    metadata["pixel size"] .* collect((size(detector)..., 0)) ./ 2
tracks[1, 3, :] .= 0.0
simulate!(tracks, msd)

brightness = 1e4 * metadata["period"]
SP2T.simulate_readouts!(
    detector,
    SP2T.getincident(tracks, brightness, detector.darkcounts, detector.pxbounds, psf),
)

jldsave("./examples/3D/metadata.jld2"; metadata = metadata)
jldsave("./examples/3D/frames.jld2"; frames = detector.readouts)
jldsave("./examples/3D/groundtruth.jld2"; tracks = tracks, msd = msd)