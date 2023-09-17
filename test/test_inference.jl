using SpBNPTrack
using JLD2

FloatType = Float32

video = load("example.jld2", "video")

initial_guess = simulate_sample(
    param = param,
    emitter_number = 1,
    diffusion_coefficient = 0.05,
    emission_rate = 200,
)

prior_param = PriorParameter(
    FloatType,
    pb = 0.1,
    μx = [param.pxnumx, param.pxnumy, 0] .* (param.pxsize / 2),
    σx = [1, 1, 0] .* (param.pxsize * 2),
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

@time SpBNPTrack.run_MCMC!(chain, video, num_iter = 100_000, run_on_gpu = true);

visualize(video, groundtruth, chain.samples)