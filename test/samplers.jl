# Sampler tests: each sampler, run with a fixed StableRNG seed, must reproduce its target
# distribution, known from theory. Tolerances are 4 standard errors of the estimate:
# exact for independent draws, from batch means for Markov chains.

println("Running sampler tests...")

# Mean of a correlated chain and its standard error from `nbatches` contiguous batches.
function batchmean(x::AbstractVector, nbatches::Integer = 100)
    len = length(x) ÷ nbatches
    means = [mean(@view x[(i-1)*len+1:i*len]) for i = 1:nbatches]
    return mean(means), std(means) / √nbatches
end

# The fraction of draws equal to each category matches its probability.
@testset let rng = StableRNG(11), p = [0.1, 0.2, 0.3, 0.4], ndraws = 40_000
    draws = [SP2T.randc(rng, log.(p) .+ 7) for _ = 1:ndraws]
    for (i, pᵢ) in enumerate(p)
        @test abs(count(==(i), draws) / ndraws - pᵢ) < 4 * √(pᵢ * (1 - pᵢ) / ndraws)
    end
end

# The MSD is drawn from its conjugate inverse-gamma posterior, with the motion likelihood
# tempered by 𝑇: shape α + n/(2𝑇), scale β + Σd²/(2𝑇).
@testset for 𝑇 in (1.0, 2.0)
    rng = StableRNG(12)
    displacement² = reshape([0.01 * i for i = 1:40], 10, 2, 2)
    msd = MeanSquaredDisplacement{Float64}(guess = 1.0, priorparams = (2, 0.1))
    posterior = InverseGamma(2 + 40 / (2𝑇), 0.1 + sum(displacement²) / (2𝑇))
    ndraws = 20_000
    draws = [SP2T.update!(rng, msd, displacement², 𝑇).value for _ = 1:ndraws]
    @test abs(mean(draws) - mean(posterior)) < 4 * std(posterior) / √ndraws
    @test abs(count(<(median(posterior)), draws) / ndraws - 0.5) < 4 * √(0.25 / ndraws)
end

# With the likelihood switched off (𝑇 = Inf), the brightness Metropolis–Hastings chain
# samples its gamma prior: mean kθ and variance kθ².
@testset let rng = StableRNG(13),
    detector = SPAD{Float64}(
        period = 1e-5,
        pixel_size = 0.1,
        darkcounts = fill(1e-3, 2, 2),
        cutoffs = (0, Inf),
        readouts = zeros(UInt16, 2, 2, 1),
    ),
    psf = CircularGaussian{Float64}(
        numerical_aperture = 1.45,
        refractive_index = 1.515,
        emission_wavelength = 0.665,
        pixel_size = 0.1,
    ),
    brightness = Brightness{Float64}(guess = 1.0, priorparams = (2, 0.5), proposalparams = (10, 1)),
    llarray = SP2T.LogLikelihoodArray{Float64}(detector.readouts),
    nsteps = 200_000

    chain = [
        SP2T.update!(rng, brightness, [0.1 0.1;;;], llarray, detector, psf, Inf).value
        for _ = 1:nsteps
    ]
    m, se = batchmean(chain)
    @test abs(m - 1.0) < 4se
    m, se = batchmean((chain .- 1.0) .^ 2)
    @test abs(m - 0.5) < 4se
end

# With the likelihood switched off, the position updates sample the track prior: the
# first frame from DNormal(μ, σ), then Brownian steps of variance msd, so frame k has mean
# μ and variance σ² + (k - 1)·msd.
@testset let rng = StableRNG(14),
    detector = SPAD{Float64}(
        period = 1e-5,
        pixel_size = 0.1,
        darkcounts = fill(1e-3, 4, 4),
        cutoffs = (0, Inf),
        readouts = zeros(UInt16, 4, 4, 3),
    ),
    psf = CircularGaussian{Float64}(
        numerical_aperture = 1.45,
        refractive_index = 1.515,
        emission_wavelength = 0.665,
        pixel_size = 0.1,
    ),
    tracks = Tracks{Float64}(
        guess = fill(0.2, 3, 2, 1),
        prior = SP2T.DNormal{Float64}([0.2, 0.2], [0.1, 0.1]),
        max_ntracks = 1,
        scaling = 0.1,
        logonprob = -10,
    ),
    llarray = SP2T.LogLikelihoodArray{Float64}(detector.readouts),
    msd = 0.01,
    nsteps = 100_000

    samples = zeros(nsteps, 3)
    for i = 1:nsteps
        SP2T.update_onchunk!(rng, tracks, msd, 1.0, llarray, detector, psf, Inf)
        samples[i, :] .= tracks.onchunk.value[:, 1, 1]
    end
    for k = 1:3
        m, se = batchmean(samples[:, k])
        @test abs(m - 0.2) < 4se
        m, se = batchmean((samples[:, k] .- 0.2) .^ 2)
        @test abs(m - (0.01 + (k - 1) * msd)) < 4se
    end
end

# Two particles, two frames: a non-identity relabeling (probability 1/2) is accepted with
# probability min(1, exp(Δ)), where Δ is the change in the motion log-prior across the
# boundary. The tracks are reset after every call.
@testset let rng = StableRNG(15),
    guess = cat([0.0 0.0; 0.4 0.0], [1.0 0.0; 0.6 0.0], dims = 3),
    tracks = Tracks{Float64}(
        guess = guess,
        prior = SP2T.DNormal{Float64}([0.5, 0.5], [1.0, 1.0]),
        max_ntracks = 2,
        scaling = 0.1,
        logonprob = -10,
    ),
    msd = 0.2,
    ncalls = 20_000

    Δ = -((0.6 - 0.0)^2 + (0.4 - 1.0)^2 - (0.4 - 0.0)^2 - (0.6 - 1.0)^2) / (2msd)
    p = min(1, exp(Δ)) / 2
    for _ = 1:ncalls
        copyto!(tracks.onchunk.value, guess)
        SP2T.update!(rng, tracks.onchunk, tracks.proposals, msd)
    end
    @test abs(tracks.proposals.counter[1, 1] / ncalls - p) < 4 * √(p * (1 - p) / ncalls)
end

# Simulated displacements have variance msd per coordinate; a Brownian bridge's midpoint
# has variance msd·N/4 for N steps.
@testset let rng = StableRNG(16), msd = 0.04
    x = zeros(1001, 2, 10)
    simulate!(rng, x, msd)
    steps = vec(diff(x, dims = 1))
    @test abs(var(steps) - msd) < 4 * msd * √(2 / (length(steps) - 1))

    N, nbridges = 10, 20_000
    midpoints = [bridge!(rng, zeros(N + 1, 1, 1), msd, zeros(2, 1, 1))[N÷2+1] for _ = 1:nbridges]
    v = msd * N / 4
    @test abs(mean(midpoints .^ 2) - v) < 4 * v * √(2 / nbridges)
end

# Simulated readouts follow each detector's model: SPAD clicks are Binomial(batchsize,
# 1 - e^-λ); EMCCD readouts are Normal(gλ + o, √v) rounded (variance v + 1/12); gamma
# EMCCD readouts are Gamma(λ/f, f·g) rounded (mean λg, variance λfg² + 1/12).
@testset let rng = StableRNG(17),
    λ = 0.5,
    n = 100_000,
    common = (period = 1e-5, pixel_size = 0.1, darkcounts = fill(1e-3, 100, 100), cutoffs = (0, Inf))

    means = fill(λ, 100, 100, 10)
    p = -expm1(-λ)
    for batchsize in (1, 5)
        spad = SPAD{Float64}(; common..., readouts = zeros(UInt16, 100, 100, 10), batchsize = batchsize)
        SP2T.simulate_readouts!(rng, spad, means)
        @test abs(mean(spad.readouts) - batchsize * p) < 4 * √(batchsize * p * (1 - p) / n)
    end

    o, g, v = 100.0, 20.0, 9.0
    emccd = EMCCD{Float64}(; common..., readouts = zeros(UInt16, 100, 100, 10), offset = o, gain = g, variance = v)
    SP2T.simulate_readouts!(rng, emccd, means)
    @test abs(mean(emccd.readouts) - (g * λ + o)) < 4 * √((v + 1 / 12) / n)
    @test abs(var(emccd.readouts) - (v + 1 / 12)) < 4 * v * √(2 / n)

    f, gᵧ = 2.0, 30.0
    gamma = SP2T.EMCCDGamma{Float64}(; common..., readouts = ones(UInt16, 100, 100, 10), gain = gᵧ, noise_excess_factor = f)
    SP2T.simulate_readouts!(rng, gamma, means)
    vᵧ = λ * f * gᵧ^2 + 1 / 12
    @test abs(mean(gamma.readouts) - λ * gᵧ) < 4 * √(vᵧ / n)
end

# The same seed gives the same chain; a different seed a different one.
@testset let detector = SPAD{Float64}(
        period = 1e-5,
        pixel_size = 0.1,
        darkcounts = fill(1e-3, 6, 6),
        cutoffs = (0, Inf),
        readouts = UInt16.(reshape([isodd(i ÷ 7) for i = 1:360], 6, 6, 10)),
    ),
    psf = CircularGaussian{Float64}(
        numerical_aperture = 1.45,
        refractive_index = 1.515,
        emission_wavelength = 0.665,
        pixel_size = 0.1,
    )

    function run(seed)
        tracks = Tracks{Float64}(
            guess = fill(0.3, 10, 2, 1),
            prior = SP2T.DNormal{Float64}([0.3, 0.3], [0.5, 0.5]),
            max_ntracks = 3,
            scaling = 0.01,
            logonprob = -2,
        )
        return runMCMC(
            tracks = tracks,
            msd = MeanSquaredDisplacement{Float64}(guess = 2e-4),
            brightness = Brightness{Float64}(guess = 0.5),
            detector = detector,
            psf = psf,
            niters = 50,
            rng = StableRNG(seed),
        )
    end
    a, b, c = run(18), run(18), run(19)
    @test a.msds == b.msds && a.brightnesses == b.brightnesses
    @test a.emittercounts == b.emittercounts && a.lasttracks == b.lasttracks
    @test a.msds != c.msds
end
