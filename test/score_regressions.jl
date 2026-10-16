using Distributions: Normal, logpdf

function score_test_tracks(::Type{T}, nemitters::Integer) where {T}
    positions = Array{T}(undef, 3, 2, nemitters)
    for m in 1:nemitters, d in 1:2, f in 1:3
        positions[f, d, m] = T(0.1 * (f + d + m))
    end
    return Tracks{T}(
        guess = positions,
        prior = DNormal{T}(T[0.2, 0.5], T[0.4, 0.7]),
        max_ntracks = max(nemitters, 1),
        scaling = T(0.1),
        logonprob = -2,
    )
end

@testset "Normalized track scores" begin
    for T in (Float32, Float64), nemitters in 0:2
        tracks = score_test_tracks(T, nemitters)
        x = tracks.onchunk.value
        prior = tracks.onchunk.prior
        msd = T(0.15)
        initial = zero(T)
        motion = zero(T)
        for m in 1:nemitters, d in 1:2
            initial += logpdf(Normal(prior.μ[d], prior.σ[d]), x[1, d, m])
            for f in 2:3
                motion += logpdf(Normal(zero(T), sqrt(msd)), x[f, d, m] - x[f-1, d, m])
            end
        end
        @test SP2T.logprior(prior, view(x, 1, :, :); normalization = true) ≈ initial
        @test SP2T.logprior(tracks.onchunk, msd) ≈ initial + motion
    end

    for T in (Float32, Float64)
        tracks = score_test_tracks(T, 1)
        prior = tracks.onchunk.prior
        tracks.onchunk.value .= reshape(prior.μ, 1, 2, 1)
        centered = SP2T.logprior(tracks.onchunk, T(0.15))
        tracks.onchunk.value[:, 1, :] .+= prior.σ[1]
        shifted = SP2T.logprior(tracks.onchunk, T(0.15))
        @test shifted - centered ≈ T(-0.5)
    end
end

@testset "Posterior includes brightness prior" begin
    for T in (Float32, Float64)
        tracks = score_test_tracks(T, 1)
        msd = MeanSquaredDisplacement{T}(guess = 0.15, priorparams = (2, 0.1))
        brightness = Brightness{T}(guess = 1.4, priorparams = (2, 3))
        first_value = brightness.value
        first_score = SP2T.get_logposterior(T(-3), tracks, msd, brightness)
        brightness.value = T(3.1)
        second_score = SP2T.get_logposterior(T(-3), tracks, msd, brightness)
        expected = logpdf(brightness.prior, brightness.value) -
                   logpdf(brightness.prior, first_value)
        @test second_score - first_score ≈ expected
    end
end

@testset "Zero-brightness prior boundary" begin
    for T in (Float32, Float64)
        brightness = Brightness{T}(guess = 0, priorparams = (1, 3), fixed = true)
        @test SP2T.logprior(brightness) == zero(T)
        @test SP2T.logprior(brightness; normalization = true) ≈ -log(T(3))
        for (shape, expected) in ((2, -Inf), (0.5, Inf))
            brightness = Brightness{T}(guess = 0, priorparams = (shape, 3), fixed = true)
            @test SP2T.logprior(brightness) == T(expected)
            @test SP2T.logprior(brightness; normalization = true) == T(expected)
        end
    end
end

@testset "MAP and MLE select finite scores" begin
    for T in (Float32, Float64)
        scores = [(NaN, NaN), (3, 5), (Inf, Inf), (4, 2), (4, 7), (-Inf, -Inf), (NaN, NaN)]
        samples = [
            Sample(zeros(T, 2, 2, 0), T(0.15), one(T), i - 1, one(T), T(p), T(l))
            for (i, (p, l)) in enumerate(scores)
        ]
        chain = Chain(samples, 20, ConstantAnnealing{T}(1))
        @test findmap(chain)[2] == 4
        @test findmap(chain)[1] === samples[4]
        @test findmle(chain)[2] == 5
        @test findmle(chain)[1] === samples[5]
        @test findmap(chain; burn_in = 4)[2] == 5
        @test findmle(chain; burn_in = 2.0)[2] == 5
        for select in (findmap, findmle)
            for invalid in (-1, 0.5, length(chain), Inf, NaN)
                @test_throws ArgumentError select(chain; burn_in = invalid)
            end
            @test_throws ArgumentError select(chain; burn_in = 5)
            @test_throws ArgumentError select(Chain(samples[1:1], 20, chain.annealing))
            @test_throws ArgumentError select(Chain(samples[1:0], 20, chain.annealing))
        end
    end
end

@testset "Initial sample has computed scores" begin
    for T in (Float32, Float64), nemitters in 0:1, brightness_value in (0, 1.4)
        tracks = score_test_tracks(T, nemitters)
        msd = MeanSquaredDisplacement{T}(guess = 0.15, priorparams = (2, 0.1))
        brightness = Brightness{T}(guess = brightness_value, priorparams = (1, 3), fixed = true)
        detector = SPAD{T}(
            period = 0.01,
            pixel_size = 0.1,
            darkcounts = fill(T(0.05), 2, 2),
            cutoffs = (0, Inf),
            readouts = zeros(UInt16, 2, 2, 3),
        )
        psf = CircularGaussian{T}(0.15, 0.1)
        chain = runMCMC(
            tracks = tracks,
            msd = msd,
            brightness = brightness,
            detector = detector,
            psf = psf,
            niters = 0,
            annealing = ConstantAnnealing{T}(2),
        )
        sample = only(chain.samples)
        @test sample.iteration == 0
        @test sample.𝑇 == T(2)
        @test isfinite(sample.logℒ)
        @test isfinite(sample.log𝒫)
        incident = SP2T.getincident(
            tracks.onchunk.value, brightness.value, detector.darkcounts, detector.pxbounds, psf,
        )
        @test sample.logℒ ≈ -sum(incident)
        @test sample.log𝒫 ≈ SP2T.get_logposterior(sample.logℒ, tracks, msd, brightness)
        @test findmap(chain)[2] == 1
        @test findmle(chain)[2] == 1
    end
end
