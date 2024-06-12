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
    "period" => 3e-6,
    "pixel size" => 0.1,
)

# expparams = ExperimentalParameters(
#     FloatType,
#     metadata["period"],
#     metadata["pixel size"],
#     load("./data/beads/beads_darkcounts.jld2", "darkcounts"),
#     metadata["numerical aperture"],
#     metadata["refractive index"],
#     metadata["wavelength"],
# )

data = Data(
    FloatType,
    Array{UInt16}(undef, 50, 50, 2550),
    metadata["period"],
    metadata["pixel size"],
    load("./data/beads/darkcounts1.jld2", "darkcounts"),
    11.14 * 12 / 1000,
    1.2 * 100 / 1000,
)

data, groundtruth = simulate!(data; diffusivity = 2, brightness = 2e6, nemitters = 1)

visualize(groundtruth, data)

jldsave("./data/example_data.jld2"; data)
jldsave("./data/example_groundtruth.jld2"; groundtruth)