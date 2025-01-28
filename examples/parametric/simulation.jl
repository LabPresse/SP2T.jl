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

psf = CircularGaussian{FloatType}(
    numerical_aperture = metadata["numerical aperture"],
    refractive_index = metadata["refractive index"],
    emission_wavelength = metadata["wavelength"],
    pixels_size = metadata["pixel size"],
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
tracks = simulate!(
    Array{FloatType}(undef, 2550, 2, ntracks),
    metadata["pixel size"] ./ 2 .* collect(size(detector)),
    [0.0, 0.0],
    msd,
)

presence = ones(2550, 1, 2)
presence[1:255, 1, 1] .= 0
presence[end-254:end, 1, 2] .= 0

brightness = 1e4 * metadata["period"]
SP2T.simulate_readouts!(
    detector,
    SP2T.getincident(
        tracks ./ presence,
        brightness,
        detector.darkcounts,
        detector.pxbounds,
        psf,
    ),
)

jldsave("./examples/parametric/metadata.jld2"; metadata = metadata)
jldsave("./examples/parametric/frames.jld2"; frames = detector.readouts)
jldsave(
    "./examples/parametric/groundtruth.jld2";
    tracks = tracks,
    msd = msd,
    presence = presence,
)