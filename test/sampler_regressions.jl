using Distributions
using Random
using SP2T
using Test

@testset "Brightness proposals preserve detector background" begin
    for T in (Float32, Float64)
        darkcounts = T[0.12 0.07; 0.03 0.19]
        readouts = reshape(UInt16[0, 1, 2, 3, 1, 0, 2, 1, 0, 2, 3, 1], 2, 2, 3)
        common = (
            period = T(1),
            pixel_size = T(1),
            darkcounts = darkcounts,
            cutoffs = (T(0), T(Inf)),
            readouts = readouts,
        )
        detectors = (
            SPAD{T}(; common..., batchsize = 3),
            EMCCD{T}(; common..., offset = T(0), gain = T(2), variance = T(1)),
        )
        psf = CircularGaussian{T}(T(0.4), T(1))
        positions = reshape(T[0.4, 0.6, 0.8, 0.5, 0.7, 0.9], 3, 2, 1)
        for detector in detectors, (old, proposed) in ((2, 2), (2, 3), (2, 0), (0, 3), (0, 0))
            llarray = SP2T.LogLikelihoodArray{T}(readouts)
            reference = SP2T.LogLikelihoodArray{T}(readouts)
            SP2T.set_poisson_mean!(reference, detector, positions, T(old), psf)
            expected_old = copy(reference.means[1])
            old_loglikelihood = SP2T.get_loglikelihood!(reference, detector)
            SP2T.set_poisson_mean!(reference, detector, positions, T(proposed), psf)
            expected_proposed = copy(reference.means[1])
            proposed_loglikelihood = SP2T.get_loglikelihood!(reference, detector)

            SP2T.set_poisson_means!(llarray, detector, positions, T(old), T(proposed), psf)
            @test llarray.means[1] ≈ expected_old
            @test llarray.means[2] ≈ expected_proposed
            @test all(isfinite, llarray.means[2])
            @test SP2T.get_Δloglikelihood!(llarray, detector) ≈
                  proposed_loglikelihood - old_loglikelihood atol = 100 * eps(T)
            if old == proposed
                @test llarray.means[1] ≈ llarray.means[2]
            end
        end
    end
end

@testset "Motion contributions do not retain endpoint likelihoods" begin
    for T in (Float32, Float64), nframes in (1, 2, 5, 6), parity in (1, 2)
        edges = T[(-1)^i * (i + 0.25) for i in 1:nframes-1]
        expected = zeros(T, nframes)
        for frame in parity:2:nframes
            frame > 1 && (expected[frame] += edges[frame-1])
            frame < nframes && (expected[frame] += edges[frame])
        end
        scratch = fill(T(17), nframes)
        SP2T.Δlogmotion!(
            scratch,
            parity:2:nframes-1,
            (3-parity):2:nframes-1,
            edges,
        )
        @test scratch == expected
    end
end

@testset "Fixed-count sampling refreshes effective positions" begin
    for T in (Float32, Float64)
        initial = fill(T(0.5), 4, 2, 1)
        tracks = Tracks{T}(
            guess = initial,
            prior = DNormal{T}(fill(T(0.5), 2), fill(T(10), 2)),
            max_ntracks = 1,
            scaling = T(0.05),
            logonprob = T(-1),
        )
        detector = SPAD{T}(
            period = T(1),
            pixel_size = T(1),
            darkcounts = fill(T(0.1), 2, 2),
            cutoffs = (T(0), T(Inf)),
            readouts = zeros(UInt16, 2, 2, 4),
        )
        llarray = SP2T.LogLikelihoodArray{T}(detector.readouts)
        psf = CircularGaussian{T}(T(0.4), T(1))
        msd = MeanSquaredDisplacement{T}(guess = T(1e6), fixed = true)
        brightness = Brightness{T}(guess = T(0), fixed = true)
        Random.seed!(932)
        SP2T.parametricMCMC!(tracks, msd, brightness, llarray, detector, psf, T(1))
        @test any(>(0), tracks.proposals.accepted)
        @test tracks.onchunk.value != initial
        @test tracks.onchunk.effvalue == tracks.onchunk.value ./ tracks.onchunk.active
    end
end

@testset "MSD conditional uses the untempered motion prior" begin
    for T in (Float32, Float64), nsteps in (0, 3), temperature in (1, 4)
        displacement² = reshape(T.(1:2nsteps) ./ T(10), nsteps, 2, 1)
        priorparams = (T(3), T(0.7))
        posterior = InverseGamma(
            priorparams[1] + T(length(displacement²)) / T(2),
            priorparams[2] + sum(displacement²) / T(2),
        )
        msd = MeanSquaredDisplacement{T}(guess = T(1), priorparams = priorparams)
        Random.seed!(467)
        expected = rand(posterior)
        Random.seed!(467)
        SP2T.update!(msd, displacement², T(temperature))
        @test msd.value == expected

        msd.fixed = true
        fixed_value = msd.value
        SP2T.update!(msd, displacement², T(temperature))
        @test msd.value == fixed_value
    end
end
