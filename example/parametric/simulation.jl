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
    numerical_aperture = metadata["numerical aperture"],
    refractive_index = metadata["refractive index"],
    emission_wavelength = metadata["wavelength"],
    pixels_size = metadata["pixel size"],
)

msd = 2 * 10 * metadata["period"]
darkcounts = load("./example/darkcounts.jld2", "darkcounts")
xᵖ = range(0, step = metadata["pixel size"], length = size(darkcounts, 1) + 1)
yᵖ = range(0, step = metadata["pixel size"], length = size(darkcounts, 2) + 1)

nparticles = 2
tracks = Array{Float64}(undef, 2550, 2, nparticles)
simulate!(tracks, metadata["pixel size"] ./ 2 .* collect(size(darkcounts)), [0.0, 0.0], msd)

presence = ones(2550, 1, 2)
presence[1:255, 1, 1] .= 0
presence[end-254:end, 1, 2] .= 0

brightness = 1e4 * metadata["period"]

intensity = SP2T.getincident(tracks ./ presence, brightness, darkcounts, (xᵖ, yᵖ), psf)
frames = SP2T.simframes(intensity)

jldsave("./example/parametric/metadata.jld2"; metadata)
jldsave("./example/parametric/frames.jld2"; frames)
jldsave(
    "./example/parametric/groundtruth.jld2";
    tracks = tracks,
    msd = msd,
    presence = presence,
)