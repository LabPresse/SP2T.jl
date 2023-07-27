using SpBNPTrack
using Test

# @testset "BNPTrack.jl" begin
#     # Write your tests here.
# end

params = SpBNPTrack.ExperimentalParameters()
priors = SpBNPTrack.Priors(
    μₓ = [params.pxnumx * params.pxsize / 2, params.pxnumy * params.pxsize / 2, 0],
    σₓ = [params.pxsize * 2, params.pxsize * 2, 0],
    p_s = [1, 0],
)

(video, gt) = SpBNPTrack.forward_main(params, priors, emission = 200.0, background = 10.0)

fig = SpBNPTrack.visualize_data_3D(video, gt)

# fig
