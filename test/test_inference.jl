using SpBNPTrack
using JLD2
using Profile
using PProf

# FloatType = Float32

video = load("example_video.jld2", "video")
groundtruth = load("example_groundtruth.jld2", "groundtruth")

initial_guess = simulate_sample(
    param = video.param,
    emitter_number = 1,
    diffusion_coefficient = 0.05,
    emission_rate = 200,
)

prior_param = PriorParameter(
    ftypeof(video),
    pb = 0.1,
    μx = [video.param.pxnumx, video.param.pxnumy, 0] .* (video.param.pxsize / 2),
    σx = [1, 1, 1] .* (video.param.pxsize * 2),
    ϕD = 1,
    χD = 1,
    ϕh = 1,
    ψh = 1,
)

chain = Chain(
    initial_guess = groundtruth,
    exp_param = video.param,
    max_emitter_num = 100,
    prior_param = prior_param,
    sizelimit = 1000,
)

@time SpBNPTrack.run_MCMC!(chain, video, num_iter = 100, run_on_gpu = true);

visualize(video, groundtruth, chain.samples)