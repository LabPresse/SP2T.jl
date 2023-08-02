using SpBNPTrack

FloatType = Float64

param = ExperimentalParameter(
    FloatType,
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
prior = Prior(param)

groundtruth = Sample(FloatType)
SpBNPTrack.simulate!(
    groundtruth,
    param = param,
    prior = prior,
    diffusion_coefficient = 0.05,
    emission_rate = 200.0,
    background_flux = fill(10.0, param.pxnumx, param.pxnumy),
    emitter_number = 3,
)

video = Video(param)
SpBNPTrack.simulate!(video, groundtruth)
visualize_data_3D(video, groundtruth)

initial_guess = Sample(FloatType)
SpBNPTrack.simulate!(
    initial_guess,
    param = param,
    prior = prior,
    diffusion_coefficient = 0.05,
    emission_rate = 200.0,
    background_flux = fill(10.0, param.pxnumx, param.pxnumy),
    emitter_number = 5,
)

chain = Chain(FloatType)
SpBNPTrack.initialize!(chain, max_emitter_num = 100, prior = prior)
