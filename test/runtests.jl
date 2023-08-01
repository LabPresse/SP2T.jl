using SpBNPTrack
# using Test

# @testset "BNPTrack.jl" begin
#     # Write your tests here.
# end

params = SpBNPTrack.ExperimentalParameters(
    Float64,
    units = ("μm", "s"),
    length = 100,
    period = 0.0033,
    exposure = 0.003,
    pxnumx = 50,
    pxnumy = 50,
    pxsize = 0.133,
    NA = 1.45,
    nᵣ = 1.515,
    λ = 0.665,
)
priors = SpBNPTrack.Priors(params)

groundtruth = SpBNPTrack.Sample(Float64)
SpBNPTrack.simulate!(
    groundtruth,
    params = params,
    priors = priors,
    diffusion_coefficient = 0.05,
    emission_rate = 200.0,
    background_flux = fill(10.0, params.pxnumx, params.pxnumy),
    emitter_number = 3,
)

video = SpBNPTrack.Video(params)
SpBNPTrack.simulate!(video, groundtruth)
SpBNPTrack.visualize_data_3D(video, groundtruth)

# fig
