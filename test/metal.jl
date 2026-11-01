# Metal tests: run only on a Mac with a functional Metal GPU. Arrays use private storage
# and scalar indexing is disallowed, so any scalar access in the GPU path fails the test.
# Expected values come from the same computation on CPU arrays (the reference), at
# Float32, the only floating-point type Metal supports.

using Metal
using Random

if !Metal.functional()
    @info "No functional Metal GPU; skipping the Metal tests."
else
    println("Running Metal tests...")
    Metal.allowscalar(false)

    private(x::AbstractArray) = MtlArray{eltype(x),ndims(x),Metal.PrivateStorage}(x)

    @testset let x = Float32.(range(-6, 6, length = 101))
        @test Array(SP2T.erfdiff.(private(x[1:end-1]), private(x[2:end]))) ≈
              SP2T.erfdiff.(x[1:end-1], x[2:end]) rtol = 1e-5
    end

    # framesum! overwrites its output even when it holds NaN or Inf (Metal's mul! keeps
    # 0 * NaN), as it does after the position update reuses the frame buffer as scratch.
    @testset let A = rand(StableRNG(22), Float32, 3, 4, 5), b = rand(StableRNG(23), Float32, 3, 4)
        r = private(Float32[NaN, -Inf, Inf, NaN, 0])
        @test Array(SP2T.framesum!(r, private(A), private(b))) ≈ vec(sum(A .* b, dims = (1, 2)))
    end

    function metalsetup(device, readouts)
        detector = SPAD{Float32}(
            period = 1e-5,
            pixel_size = 0.1,
            darkcounts = device(fill(1f-3, 12, 12)),
            cutoffs = (0, Inf),
            readouts = device(readouts),
        )
        psf = CircularGaussian{Float32}(
            numerical_aperture = 1.45,
            refractive_index = 1.515,
            emission_wavelength = 0.665,
            pixel_size = 0.1,
        )
        tracks = Tracks{Float32}(
            guess = device(fill(0.6f0, size(readouts, 3), 2, 1)),
            prior = SP2T.DNormal{Float32}(device(Float32[0.6, 0.6]), device(Float32[0.5, 0.5])),
            max_ntracks = 4,
            scaling = 0.01,
            logonprob = -2,
        )
        return detector, psf, tracks
    end

    readouts = UInt16.(reshape([isodd(i ÷ 11) for i = 1:12*12*20], 12, 12, 20))
    truetracks = Float32.(cat([0.4 + 0.01k for k = 1:20, d = 1:2], [0.8 - 0.01k for k = 1:20, d = 1:2], dims = 3))

    @testset let
        loglikelihood = map((identity, private)) do device
            detector, psf, _ = metalsetup(device, readouts)
            llarray = SP2T.LogLikelihoodArray{Float32}(detector.readouts)
            SP2T.set_poisson_mean!(llarray, detector, device(truetracks), 0.5f0, psf)
            SP2T.get_loglikelihood!(llarray, detector)
        end
        @test loglikelihood[2] ≈ loglikelihood[1] rtol = 1e-5
    end

    function metalrun(parametric, seed)
        Random.seed!(Metal.default_rng(), seed)
        detector, psf, tracks = metalsetup(private, readouts)
        return runMCMC(
            tracks = tracks,
            msd = MeanSquaredDisplacement{Float32}(guess = 2f-4),
            brightness = Brightness{Float32}(guess = 0.5f0),
            detector = detector,
            psf = psf,
            niters = 30,
            parametric = parametric,
            rng = StableRNG(seed),
        )
    end

    @testset for parametric in (true, false)
        chain = metalrun(parametric, 21)
        @test last(chain.iterations) == 30
        @test all(isfinite, chain.logposteriors[2:end])
        @test chain.msds == metalrun(parametric, 21).msds
    end
end
