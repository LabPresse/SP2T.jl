using SpBNPTrack
using Random

Random.seed!(2)
FloatType = Float32

param = ExperimentalParameter(
    FloatType,
    units = ("μm", "s"),
    length = 256,
    period = 0.0033,
    exposure = 0.003,
    pxsize = 0.133,
    darkcounts = fill(1e-3, (50, 50)),
    NA = 1.45,
    nᵣ = 1.515,
    λ = 0.665,
)

# prior = Prior(param)

groundtruth = simulate_sample(
    param = param,
    emitter_number = 1,
    diffusion_coefficient = 0.05,
    emission_rate = 200,
)

video = Video(param, groundtruth)
visualize(video, groundtruth)

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