using BNPTrack
using Test

# @testset "BNPTrack.jl" begin
#     # Write your tests here.
# end

params = BNPTrack.ExperimentalParameters()
priors = BNPTrack.Priors(
    location_μx = params.pixelnumx * params.pixelsize / 2,
    location_μy = params.pixelnumy * params.pixelsize / 2,
    location_σx = params.pixelsize * 2,
    location_σy = params.pixelsize * 2,
)

(video, gt) = BNPTrack.forward_main(params, priors)

fig = BNPTrack.visualize(video, gt)

# fig
