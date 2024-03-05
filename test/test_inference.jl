using SpBNPTrack
using JLD2
# using Profile
# using PProf

video = load("example_video.jld2", "video")
groundtruth = load("example_groundtruth.jld2", "groundtruth")

initial_guess = simulate_sample(
    param = video.param,
    emitter_number = 4,
    diffusion_coefficient = 0.05,
    emission_rate = 200,
)

prior_param = PriorParameter(
    ftypeof(video),
    pb = 0.1,
    μx = [video.param.pxnumx, video.param.pxnumy, 0] .* (video.param.pxsize / 2),
    σx = [1, 1, 1] .* (video.param.pxsize * 2),
    ϕD = 1,
    χD = 0.1,
    ϕh = 1,
    ψh = 1,
    qM = 0.2,
)

# initial_guess = deepcopy(groundtruth)
initial_guess = deepcopy(chain.samples[end])
# initial_guess.h = groundtruth.h
# initial_guess.D = groundtruth.D
# initial_guess.x[:, 2 .+ (1:size(groundtruth.x, 2)), :] .= groundtruth.x
# initial_guess.x .+=
#     video.param.PSF.σ_ref * randn(eltype(initial_guess.x), size(initial_guess.x))

chain = Chain(
    initial_guess = initial_guess,
    video = video,
    max_emitter_num = 10,
    prior_param = prior_param,
    sizelimit = 2000,
)

gt_chain = Chain(
    initial_guess = groundtruth,
    video = video,
    max_emitter_num = size(groundtruth),
    prior_param = prior_param,
    sizelimit = 1000,
)

# @time SpBNPTrack.run_MCMC!(chain, video, num_iter = 1_000, run_on_gpu = true);
# Profile.clear()
# @profview SpBNPTrack.run_MCMC!(chain, video, num_iter = 10_000, run_on_gpu = true);
SpBNPTrack.run_MCMC!(chain, video, num_iter = 2_000_000, run_on_gpu = true);
# SpBNPTrack.run_MCMC!(chain, video, num_iter = 100_000, run_on_gpu = true);

SpBNPTrack.to_cpu!(chain)
jldsave("example_chain2.jld2"; chain)

visualize(video, groundtruth, chain, burn_in = 200)