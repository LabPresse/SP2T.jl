using Test, Random, StableRNGs, Distributions

@testset "Likelihood-only tempering leaves the MSD conditional unchanged" begin
    # Four independent Gaussian increments with sum of squares 30 give
    # IG(alpha + 4/2, beta + 30/2) = IG(4,18), regardless of image temperature.
    for F in (Float32, Float64), temperature in (F(1), F(2), F(Inf))
        displacement2 = reshape(F[1, 4, 9, 16], 2, 2, 1)
        msd = MeanSquaredDisplacement{F}(guess = 1, priorparams = (2, 3))
        rng, reference_rng = StableRNG(901), StableRNG(901)
        expected = InverseGamma(F(4), F(18))
        for _ = 1:8
            @test SP2T.update!(rng, msd, displacement2, temperature).value ==
                rand(reference_rng, expected)
        end
    end
end

@testset "Gamma brightness boundary and variable brightness score" begin
    for F in (Float32, Float64)
        b = Brightness{F}(guess = 0, priorparams = (1, 3))
        @test iszero(SP2T.logprior(b))
        @test SP2T.logprior(b; normalization = true) == logpdf(Gamma(F(1), F(3)), F(0))
        @test SP2T.Δlogprior(b, F(2)) ≈ -F(2) / 3
        @test SP2T.Δlogprior(b, F(0)) == zero(F)
        @test SP2T.logprior(Brightness{F}(guess = 0, priorparams = (2, 3))) == -Inf
        @test SP2T.logprior(Brightness{F}(guess = 0, priorparams = (0.5, 3))) == Inf
    end

    # Independent normal-coordinate reference: one actual on track and one off
    # candidate represented by E[log prior], with standardized increments.
    epsilon = [0.3 -0.2; -0.4 0.5]
    first = [0.5, 0.8]
    mu, sigma = [0.4, 0.6], [0.7, 0.9]
    msd = MeanSquaredDisplacement{Float64}(guess = 0.2, priorparams = (2, 0.1))
    value = reshape(vcat(first', first' .+ sqrt(msd.value) .* cumsum(epsilon; dims = 1)), 3, 2, 1)
    tracks = Tracks{Float64}(guess = value, prior = DNormal{Float64}(mu, sigma),
        max_ntracks = 2, scaling = 0.01, logonprob = -2)
    b = Brightness{Float64}(guess = 1.7, priorparams = (2, 3))
    likelihood = -7.5
    expected_off = -6 * (1 + log(2pi)) / 2 - sum(log, sigma)
    expected = likelihood + sum(logpdf.(Normal(), epsilon)) +
        sum(logpdf.(Normal.(mu, sigma), first)) + expected_off -
        3 * log(msd.value) - 0.1 / msd.value - 2 + logpdf(Gamma(2, 3), b.value)
    actual = SP2T.get_logposterior(likelihood, tracks, msd, b)
    @test actual ≈ expected
    @test actual - SP2T.get_logposterior(likelihood, tracks, msd) ≈
        logpdf(Gamma(2, 3), b.value)
    b.value = 2.4
    @test SP2T.get_logposterior(likelihood, tracks, msd, b) - actual ≈
        logpdf(Gamma(2, 3), 2.4) - logpdf(Gamma(2, 3), 1.7)
end

@testset "Finite retained-score selection" begin
    samples = [Sample(zeros(1, 2, 1), 1.0, 1.0, i - 1, 1.0, p, l)
        for (i, (p, l)) in enumerate(zip(
            [NaN, Inf, -Inf, -3.0, 2.0, 2.0],
            [Inf, NaN, -Inf, 4.0, 1.0, 3.0]))]
    chain = Chain(samples, 10, ConstantAnnealing{Float64}(1))
    @test findmap(chain)[2] == 5
    @test findmle(chain)[2] == 4
    @test findmap(chain; burn_in = 5)[2] == 6
    @test findmle(chain; burn_in = 4)[2] == 6
    @test_throws ArgumentError findmap(chain; burn_in = -1)
    @test_throws ArgumentError findmle(chain; burn_in = length(chain))
    invalid = Chain(samples[1:3], 10, ConstantAnnealing{Float64}(1))
    @test_throws ArgumentError findmap(invalid)
    @test_throws ArgumentError findmle(invalid)
    @test chain.iteration == 5
    @test :iteration in propertynames(chain)
    @test Chain(samples, 10, chain.annealing; iteration = 13).iteration == 13
end

@testset "Fixed activity contract and inactive PSF safety" begin
    for F in (Float32, Float64), dims in (2, 3)
        guess = zeros(F, 3, dims, 1)
        prior = DNormal{F}(zeros(F, dims), ones(F, dims))
        make(active) = Tracks{F}(; guess, prior, max_ntracks = 2,
            scaling = 0, logonprob = -1, active)
        for columns in (1, dims)
            active = ones(F, 3, columns, 1)
            active[1, :, :] .= 0
            preserved = copy(active)
            tracks = make(active)
            SP2T.seteffvalue!(tracks.onchunk)
            @test active == preserved
            @test tracks.onchunk.active == preserved
            @test tracks.proposals.active == tracks.onchunk.active
            @test all(isone, tracks.active[1][:, :, 2:2])
            # A reused proposal buffer must not change the fixed activity history.
            fill!(tracks.proposals.active, one(F))
            SP2T.propose!(StableRNG(503), tracks.proposals, tracks.onchunk)
            @test tracks.proposals.active == preserved
            SP2T.seteffvalue!(tracks.proposals)
            @test tracks.proposals.effvalue == tracks.onchunk.effvalue # scaling is zero
            @test all(isinf, tracks.onchunk.effvalue[1:1, 1:2, :])
            dims == 3 && @test all(iszero, tracks.onchunk.effvalue[1:1, 3:3, :])
            @test tracks.onchunk.effvalue[2:3, :, :] == guess[2:3, :, :]
        end
        for bad in (F(-1), F(0.5), F(2), F(NaN))
            @test_throws ArgumentError make(fill(bad, 3, 1, 1))
        end
        @test_throws DimensionMismatch make(ones(F, 2, 1, 1))
        @test_throws DimensionMismatch make(ones(F, 3, 1, 2))
        @test_throws DimensionMismatch make(ones(F, 3, dims + 1, 1))
        inconsistent = ones(F, 3, dims, 1)
        inconsistent[1, 1, 1] = 0
        @test_throws ArgumentError make(inconsistent)

        # All-off tracks must give background-only means, including positive
        # axial coordinates that formerly produced an infinite PSF width.
        for coordinate in (zero(F), F(0.7))
            tracks = Tracks{F}(guess = fill(coordinate, 3, dims, 1), prior = prior,
                max_ntracks = 1, scaling = 0, logonprob = -1, active = zeros(F, 3, 1, 1))
            SP2T.seteffvalue!(tracks.onchunk)
            detector = SPAD{F}(period = 0.1, pixel_size = 1, darkcounts = fill(F(0.1), 2, 2),
                cutoffs = (0, Inf), readouts = ones(UInt16, 2, 2, 3), batchsize = 3)
            psf = dims == 2 ? CircularGaussian{F}(0.4, 1) :
                CircularGaussianLorentzian{F}(0.4, 0.8, 1)
            dense = SP2T.LogLikelihoodArray{F}(detector.readouts)
            sparse = SP2T.loglikelihoodaux(detector)
            SP2T.set_poisson_mean!(dense, detector, tracks.onchunk.effvalue, F(2), psf)
            SP2T.set_poisson_mean!(sparse, detector, tracks.onchunk.effvalue, F(2), psf)
            @test dense.means[1] == repeat(reshape(detector.darkcounts, 2, 2, 1), 1, 1, 3)
            @test sparse.means[1] == sparse.darkcounts
            @test all(iszero, sparse.linear[1])
        end
    end
end

function core_regression_state(::Type{F}) where {F}
    tracks = Tracks{F}(guess = reshape(F.(1:6) ./ 20, 3, 2, 1),
        prior = DNormal{F}(fill(F(0.5), 2), fill(F(2), 2)),
        max_ntracks = 1, scaling = F(0.02), logonprob = -1)
    return (; tracks,
        msd = MeanSquaredDisplacement{F}(guess = 0.15, priorparams = (2, 0.1)),
        brightness = Brightness{F}(guess = 1.4, priorparams = (2, 3)),
        detector = SPAD{F}(period = 0.1, pixel_size = 1, darkcounts = fill(F(0.1), 2, 2),
            cutoffs = (0, Inf), readouts = ones(UInt16, 2, 2, 3), batchsize = 3),
        psf = CircularGaussian{F}(0.4, 1))
end

@testset "Free brightness starts and fixed zero boundary" begin
    for F in (Float32, Float64)
        for invalid in (F(0), F(-1), F(Inf), F(NaN))
            state = core_regression_state(F)
            state.brightness.value = invalid
            rng, reference_rng = StableRNG(611), StableRNG(611)
            @test_throws ArgumentError runMCMC(; state..., niters = 0,
                sizelimit = 4, parametric = true, rng)
            @test rand(rng, UInt64) == rand(reference_rng, UInt64)

            state = core_regression_state(F)
            chain = runMCMC(; state..., niters = 0, sizelimit = 4,
                parametric = true, rng = StableRNG(611))
            state.brightness.value = invalid
            @test_throws ArgumentError runMCMC!(; chain, state..., niters = 1,
                parametric = true, rng = StableRNG(612))
            @test chain.iteration == 0
            @test chain.iterations == [0]
        end
        fixed = merge(core_regression_state(F), (
            brightness = Brightness{F}(guess = 0, priorparams = (1, 3), fixed = true),
        ))
        chain = runMCMC(; fixed..., niters = 0, sizelimit = 4,
            parametric = true, rng = StableRNG(613))
        @test only(chain.brightnesses) == zero(F)
        @test isfinite(only(chain.loglikelihoods))
        @test isfinite(only(chain.logposteriors))
    end
end

@testset "Initial scores and thinned same-process continuation" begin
    for F in (Float32, Float64)
        initial = core_regression_state(F)
        zerochain = runMCMC(; initial..., niters = 0, sizelimit = 4, parametric = true,
            rng = StableRNG(421), annealing = PolynomialAnnealing{F}(2, 26, 2))
        @test zerochain.iteration == 0
        @test zerochain.iterations == [0]
        @test isfinite(only(zerochain.loglikelihoods))
        @test isfinite(only(zerochain.logposteriors))
        @test only(zerochain.samples).𝑇 == F(2)
        @test findmap(zerochain)[2] == findmle(zerochain)[2] == 1

        for limit in (4, 64), optional in (false, true)
            function fit(split)
                state = core_regression_state(F)
                rng = StableRNG(421)
                score_rng = StableRNG(422)
                chain = runMCMC(; state..., niters = split ? 13 : 26, sizelimit = limit,
                    annealing = PolynomialAnnealing{F}(2, 26, 2), parametric = true,
                    rng, logposterior_rng = score_rng, msdrescale = optional)
                if split
                    @test chain.iteration == 13
                    limit == 4 && @test last(chain.iterations) < chain.iteration
                    runMCMC!(; chain, state..., niters = 13, parametric = true,
                        rng, logposterior_rng = score_rng, msdrescale = optional)
                end
                return chain, state, rng, score_rng
            end
            whole, a, arng, ascore = fit(false)
            split, b, brng, bscore = fit(true)
            @test whole.iteration == split.iteration == 26
            @test whole.iterations == split.iterations
            @test whole.msds == split.msds
            @test whole.brightnesses == split.brightnesses
            @test whole.loglikelihoods == split.loglikelihoods
            @test whole.logposteriors == split.logposteriors
            @test [s.tracks for s in whole.samples] == [s.tracks for s in split.samples]
            @test a.tracks.onchunk.value == b.tracks.onchunk.value
            @test a.msd.value == b.msd.value
            @test a.brightness.value == b.brightness.value
            @test a.msd.counter == b.msd.counter
            @test a.brightness.counter == b.brightness.counter
            @test rand(arng, UInt64) == rand(brng, UInt64)
            @test rand(ascore, UInt64) == rand(bscore, UInt64)
        end
    end
end
