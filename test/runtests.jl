using SpBNPTrack
using Test

# @testset "BNPTrack.jl" begin
#     # Write your tests here.
# end

params = SpBNPTrack.ExperimentalParameters()
priors = SpBNPTrack.Priors(
    location_μx = params.pixelnumx * params.pixelsize / 2,
    location_μy = params.pixelnumy * params.pixelsize / 2,
    location_σx = params.pixelsize * 2,
    location_σy = params.pixelsize * 2,
)

(video, gt) = SpBNPTrack.forward_main(params, priors)

fig = SpBNPTrack.visualize_data(video, gt)

# fig
