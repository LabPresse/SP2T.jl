using SP2T
using Random
using JLD2

Random.seed!(9)
FloatType = Float32

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

psf = CircularGaussian{Float64}(
    metadata["numerical aperture"],
    metadata["refractive index"],
    metadata["wavelength"],
    metadata["pixel size"],
)

msd = 2 * 10 * metadata["period"]
darkcounts = load("./example/2D/darkcounts.jld2", "darkcounts")
xᵖ = range(0, step = metadata["pixel size"], length = size(darkcounts, 1) + 1)
yᵖ = range(0, step = metadata["pixel size"], length = size(darkcounts, 2) + 1)

tracks = Array{Float64}(undef, 2550, 2, 1)
simulate!(
    tracks,
    metadata["pixel size"] ./ 2 .* collect(size(darkcounts)),
    [0.0, 0.0],
    msd,
)

brightness = 1e4 * metadata["period"]

intensity = SP2T.getincident(tracks, brightness, darkcounts, xᵖ, yᵖ, psf)
frames = SP2T.simframes(intensity)

jldsave("./example/2D/metadata.jld2"; metadata)
jldsave("./example/2D/frames.jld2"; frames)
jldsave("./example/2D/groundtruth.jld2"; tracks = tracks, msd = msd)