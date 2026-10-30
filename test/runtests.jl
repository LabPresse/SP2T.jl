using SP2T
using Test
using Distributions
using StableRNGs

println("Running PSF-related tests...")

@test SP2T.gaussianpeak(0.09890858623710756, 0.133) ≈ 0.2486333227348026

@testset let psf = CircularGaussian{Float64}(
        numerical_aperture = 1.45,
        refractive_index = 1.515,
        emission_wavelength = 0.665,
        pixel_size = 0.133,
    )
    @test psf.A ≈ 0.2486333227348026
    @test psf.σ ≈ 0.09890858623710756
end

@testset let (σ₀, z₀) = SP2T.getσ₀z₀(1.45, 1.515, 0.665)
    @test σ₀ ≈ 0.09890858623710756
    @test z₀ ≈ 0.2800714501487841
end

println("Running property-related tests...")

@testset let count = SP2T.EmitterCount{Float64}(1, 10, -10)
    @test count.limit == 10
end

@testset let chain = Chain(
        [Sample(zeros(3, 2, 1), 1.0, 1.0, 0, 1.0, NaN, NaN)],
        10,
        ConstantAnnealing{Float64}(1),
    )
    @test all(s -> s isa Symbol, propertynames(chain))
    @test fieldnames(Chain) ⊆ propertynames(chain)
    for s in propertynames(chain)
        @test (getproperty(chain, s); true)
    end
end

println("Running prior-related tests...")

@testset let μ = [1.0, 2.0],
    σ = [0.5, 0.3],
    msd = 0.04,
    value = reshape([0.1 * i^1.5 for i = 1:16], 4, 2, 2),
    chunk = SP2T.TrackChunk(
        value,
        ones(size(value)),
        similar(value, 3, 2, 2),
        similar(value),
        SP2T.DNormal{Float64}(μ, σ),
    )

    expected =
        sum(logpdf.(Normal(0, √msd), diff(value, dims = 1))) +
        sum(logpdf.(Normal.(μ, σ), value[1, :, :]))
    @test SP2T.logprior(chunk, msd) ≈ expected
end

println("Running motion-related tests...")

# Frame k of parity i gets the motion change of the steps leaving it (k → k+1, entry k
# of Σ) and arriving at it (k-1 → k, entry k-1), whichever exist. Δ starts as NaN so
# any entry left unwritten fails the test.
@testset for nframes in (5, 6), i in (1, 2)
    nsteps = nframes - 1
    Σ = [10.0^k for k = 1:nsteps]
    Δ = fill(NaN, nframes)
    SP2T.Δlogmotion!(Δ, i:2:nsteps, mod1(i + 1, 2):2:nsteps, Σ)
    expected = [(k <= nsteps ? Σ[k] : 0.0) + (k >= 2 ? Σ[k-1] : 0.0) for k = i:2:nframes]
    @test Δ[i:2:end] == expected
end

println("Running mask-related tests...")

# A particle dark in frames 2 and 3, with one mask column per coordinate or one shared
# column. Candidates added up to `max_ntracks` are always active, and proposals share
# the current mask.
@testset for active in (
    reshape(Bool[1, 0, 0, 1, 1, 0, 0, 1], 4, 2, 1),
    reshape(Bool[1, 0, 0, 1], 4, 1, 1),
)
    local tracks = Tracks{Float64}(
        guess = zeros(4, 2, 1),
        prior = SP2T.DNormal{Float64}([0.0, 0.0], [1.0, 1.0]),
        max_ntracks = 3,
        scaling = 0.1,
        logonprob = -10,
        active = active,
    )

    @test tracks.active[1][:, :, 1:1] == active
    @test all(isone, tracks.active[1][:, :, 2:3])
    @test tracks.proposals.active == tracks.onchunk.active
end

println("Running annealing-related tests...")

# 1 + (𝑇₀ - 1)(1 - i/I)^order: 𝑇₀ at the start, 1 from I on, decreasing in between.
@testset for order in (1, 2, 3)
    a = PolynomialAnnealing{Float64}(1000, 1000, order)
    temperatures = [SP2T.temperature(a, i) for i = 0:1100]
    @test first(temperatures) == 1000
    @test all(==(1), temperatures[1001:end])
    @test issorted(temperatures, rev = true)
    @test SP2T.temperature(a, 500) ≈ 1 + 999 * 0.5^order
end

@test all(i -> SP2T.temperature(PolynomialAnnealing{Float64}(), i) == 1, 0:10)

println("Running fixed-variable tests...")

@test SP2T.DNormal{Float64}([0.0, 0.0], [1.0, 1.0]).dims == 2

# The detector replaces 0 and Inf dark counts in its own copy, not in the caller's array.
@testset let darkcounts = [0.0 1.0; Inf 2.0],
    original = copy(darkcounts),
    detector = SPAD{Float64}(
        period = 1e-5,
        pixel_size = 0.1,
        darkcounts = darkcounts,
        cutoffs = (0, Inf),
        readouts = zeros(UInt16, 2, 2, 1),
    )

    @test darkcounts == original
    @test detector.darkcounts == [floatmin(Float64) 1.0; floatmax(Float64) 2.0]
end

# A fixed emitter count is neither updated nor evaluated, and it selects the parametric
# sweep, which leaves the off candidates untouched (the nonparametric sweep redraws them).
@testset let detector = SPAD{Float64}(
        period = 1e-5,
        pixel_size = 0.1,
        darkcounts = fill(1e-3, 4, 4),
        cutoffs = (0, Inf),
        readouts = ones(UInt16, 4, 4, 3),
    ),
    psf = CircularGaussian{Float64}(
        numerical_aperture = 1.45,
        refractive_index = 1.515,
        emission_wavelength = 0.665,
        pixel_size = 0.1,
    ),
    tracks = Tracks{Float64}(
        guess = fill(0.2, 3, 2, 1),
        prior = SP2T.DNormal{Float64}([0.2, 0.2], [1.0, 1.0]),
        max_ntracks = 3,
        scaling = 0.01,
        logonprob = -10,
    )

    count = tracks.nemitters
    count.fixed = true
    fill!(count.likelihood, NaN)
    SP2T.update!(
        StableRNG(1),
        count,
        tracks.effvalue[1],
        1.0,
        SP2T.LogLikelihoodArray{Float64}(detector.readouts),
        detector,
        psf,
        1.0,
    )
    @test count.value == 1
    @test all(isnan, count.likelihood)

    offcandidates = copy(tracks.offchunk.value)
    chain = runMCMC(
        tracks = tracks,
        msd = MeanSquaredDisplacement{Float64}(guess = 2e-4),
        brightness = Brightness{Float64}(guess = 1.0),
        detector = detector,
        psf = psf,
        niters = 5,
        parametric = false,
        rng = StableRNG(5),
    )
    @test all(==(1), chain.emittercounts)
    @test isequal(tracks.offchunk.value, offcandidates)
end

println("Running identity-swap tests...")

# Frame k is relabeled by perm applied once per accepted boundary up to k.
@testset let x = reshape(collect(1.0:30.0), 5, 2, 3),
    perm = [2, 3, 1],
    accepted = [false, true, false, true, false],
    σ = [[1, 2, 3], [2, 3, 1], [2, 3, 1], [3, 1, 2], [3, 1, 2]]

    y = copy(x)
    SP2T.relabel!(y, SP2T.relabelindices(x, perm, accepted), similar(x))
    @test all(y[k, :, :] == x[k, :, σ[k]] for k = 1:5)
end

# Two particles on the lines y = 0 and y = 1, whose labels are swapped from frame 6 on.
# With a small MSD, swapping back at boundary 5 → 6 is overwhelmingly favored and any
# other swap overwhelmingly disfavored, so once a non-identity relabeling is proposed
# (probability 1/2 per call; 1 - 2^-50 within 50 calls), the tracks are untangled and
# stay so. The particle on y = 0 is dark in frames 3 and 8; its mask must follow it.
@testset let nframes = 10,
    line = [0.0, 1.0],
    swapped = [k <= 5 ? m : 3 - m for k = 1:nframes, m = 1:2],
    guess = [d == 1 ? 0.1k : line[swapped[k, m]] for k = 1:nframes, d = 1:2, m = 1:2],
    dark = [swapped[k, m] == 1 && k in (3, 8) for k = 1:nframes, d = 1:1, m = 1:2],
    tracks = Tracks{Float64}(
        guess = guess,
        prior = SP2T.DNormal{Float64}([0.5, 0.5], [1.0, 1.0]),
        max_ntracks = 2,
        scaling = 0.01,
        logonprob = -10,
        active = .!dark,
    )

    rng = StableRNG(2)
    for _ = 1:50
        SP2T.update!(rng, tracks.onchunk, tracks.proposals, 1e-4)
    end
    value, active = tracks.onchunk.value, tracks.onchunk.active
    @test value[:, 1, :] == guess[:, 1, :]
    @test all(m -> allequal(value[:, 2, m]), 1:2)
    @test sort(value[1, 2, :]) == line
    for m = 1:2
        @test active[:, :, m] == (value[1, 2, m] == 0 ? [!(k in (3, 8)) for k = 1:nframes, d = 1:1] : trues(nframes, 1))
    end
    @test tracks.proposals.active == active
    @test tracks.proposals.counter[2, 1] > 0
end

println("Running simulation and sweep tests...")

# Readouts are clipped to the UInt16 range: with zero offset and large noise, about half
# of the Gaussian readouts are negative and must become 0.
@testset let detector = EMCCD{Float64}(
        period = 1e-5,
        pixel_size = 0.1,
        darkcounts = fill(1e-3, 8, 8),
        cutoffs = (0, Inf),
        readouts = zeros(UInt16, 8, 8, 50),
        offset = 0,
        gain = 1,
        variance = 1e6,
    )

    SP2T.simulate_readouts!(StableRNG(3), detector, zeros(8, 8, 50))
    @test count(iszero, detector.readouts) > length(detector.readouts) / 4
    @test maximum(detector.readouts) < 10_000
end

# After a parametric step, effvalue reflects the updated positions, which the brightness
# update reads. With small proposal steps over 20 frames, some position is always
# accepted (2000 of 2000 runs), so a stale effvalue would differ.
@testset let detector = SPAD{Float64}(
        period = 1e-5,
        pixel_size = 0.1,
        darkcounts = fill(1e-3, 4, 4),
        cutoffs = (0, Inf),
        readouts = zeros(UInt16, 4, 4, 20),
    ),
    psf = CircularGaussian{Float64}(
        numerical_aperture = 1.45,
        refractive_index = 1.515,
        emission_wavelength = 0.665,
        pixel_size = 0.1,
    ),
    tracks = Tracks{Float64}(
        guess = fill(0.2, 20, 2, 1),
        prior = SP2T.DNormal{Float64}([0.2, 0.2], [1.0, 1.0]),
        max_ntracks = 1,
        scaling = 0.005,
        logonprob = -10,
    ),
    llarray = SP2T.LogLikelihoodArray{Float64}(detector.readouts)

    SP2T.parametricMCMC!(
        StableRNG(4),
        tracks,
        MeanSquaredDisplacement{Float64}(guess = 2e-4),
        Brightness{Float64}(guess = 0.1),
        llarray,
        detector,
        psf,
        1.0,
    )
    @test tracks.onchunk.effvalue == tracks.onchunk.value ./ tracks.onchunk.active
end

println("Running bridge tests...")

# With zero MSD a Brownian bridge is the straight line between its ends; with any MSD it
# starts and ends exactly there.
@testset let xend = reshape([0.0 1.0; 2.0 -1.0; 3.0 5.0; 4.0 4.0], 2, 2, 2), nframes = 11
    straight = [xend[1, d, m] + (k - 1) / (nframes - 1) * (xend[2, d, m] - xend[1, d, m])
        for k = 1:nframes, d = 1:2, m = 1:2]
    @test bridge!(zeros(nframes, 2, 2), 0.0, xend) ≈ straight
    x = bridge!(StableRNG(6), zeros(nframes, 2, 2), 0.1, xend)
    @test x[1, :, :] ≈ xend[1, :, :]
    @test x[end, :, :] ≈ xend[2, :, :]
end

println("Running likelihood tests...")

function pixelloglikelihood(detector, λ)
    llarray = SP2T.LogLikelihoodArray{Float64}(detector.readouts)
    llarray.means[1] .= λ
    SP2T.set_pixel_loglikelihood!(llarray, detector)
    return copy(llarray.pixel)
end

function pixelΔloglikelihood(detector, λ₁, λ₂)
    llarray = SP2T.LogLikelihoodArray{Float64}(detector.readouts)
    llarray.means[1] .= λ₁
    llarray.means[2] .= λ₂
    SP2T.set_pixel_Δloglikelihood!(llarray, detector)
    return copy(llarray.pixel)
end

# Each detector's per-pixel log-likelihood equals the Distributions.jl log-density of its
# readout model minus terms that do not depend on the expected count λ (the binomial
# coefficient, the Gaussian normalization, the gamma's r/θ), and each Δ version equals
# the difference of the two log-densities.
@testset let readouts = UInt16[0 1 2; 3 4 5;;; 5 0 2; 1 3 4],
    nonzero = UInt16[1 7 2; 30 4 12;;; 5 9 2; 1 3 4],
    λ₁ = [0.1 0.5 1.0; 2.0 0.3 4.0;;; 0.05 1.5 0.7; 3.0 0.2 2.5],
    λ₂ = [0.4 0.2 1.3; 0.9 2.0 3.0;;; 0.6 0.1 0.9; 1.0 4.0 0.3],
    common = (period = 1e-5, pixel_size = 0.1, darkcounts = fill(1e-3, 2, 3), cutoffs = (0, Inf))

    r = Int.(readouts)
    spad = SPAD{Float64}(; common..., readouts = readouts, batchsize = 5)
    spadreference(λ) = logpdf.(Binomial.(5, -expm1.(-λ)), r)
    @test pixelloglikelihood(spad, λ₁) ≈ spadreference(λ₁) .- log.(binomial.(5, r))
    @test pixelΔloglikelihood(spad, λ₁, λ₂) ≈ spadreference(λ₂) .- spadreference(λ₁)

    o, g, v = 100.0, 20.0, 9.0
    emccd = EMCCD{Float64}(; common..., readouts = readouts, offset = o, gain = g, variance = v)
    emccdreference(λ) = logpdf.(Normal.(g .* λ .+ o, √v), r)
    @test pixelloglikelihood(emccd, λ₁) ≈ emccdreference(λ₁) .+ log(2π) / 2
    @test pixelΔloglikelihood(emccd, λ₁, λ₂) ≈ emccdreference(λ₂) .- emccdreference(λ₁)

    rᵧ = Int.(nonzero)
    f, gᵧ = 2.0, 3.0
    θ = f * gᵧ
    gamma = SP2T.EMCCDGamma{Float64}(; common..., readouts = nonzero, gain = gᵧ, noise_excess_factor = f)
    gammareference(λ) = logpdf.(Gamma.(λ ./ f, θ), rᵧ)
    @test pixelloglikelihood(gamma, λ₁) ≈ gammareference(λ₁) .+ rᵧ ./ θ
    @test pixelΔloglikelihood(gamma, λ₁, λ₂) ≈ gammareference(λ₂) .- gammareference(λ₁)
end

println("Running PSF integration tests...")

# The pixel-integrated PSF sums to 1 over a ±10σ window, and the pixel centered on the
# emitter receives the fraction psf.A.
@testset let psf = CircularGaussian{Float64}(
        numerical_aperture = 1.45,
        refractive_index = 1.515,
        emission_wavelength = 0.665,
        pixel_size = 0.1,
    ),
    frame = SP2T._integrate(psf, 0.1),
    c = (size(frame, 1) + 1) ÷ 2

    @test frame[c, c, 1] ≈ psf.A
    @test sum(frame) ≈ 1
end

# An emitter at the center of pixel (11, 11) of a 21×21 frame (±10σ in each direction)
# adds `brightness` to that pixel and `brightness / psf.A` photons over the frame. In 3D,
# the lateral width grows to σ₀√2 at z = z₀.
@testset let detector = SPAD{Float64}(
        period = 1e-5,
        pixel_size = 0.1,
        darkcounts = fill(0.01, 21, 21),
        cutoffs = (0, Inf),
        readouts = zeros(UInt16, 21, 21, 1),
    ),
    optics = (numerical_aperture = 1.45, refractive_index = 1.515, emission_wavelength = 0.665, pixel_size = 0.1),
    b = 2.0

    psf = CircularGaussian{Float64}(; optics...)
    incident = SP2T.getincident([1.05 1.05;;;], b, detector.darkcounts, detector.pxbounds, psf)
    @test incident[11, 11, 1] ≈ 0.01 + b
    @test sum(incident .- 0.01) ≈ b / psf.A

    psf3 = CircularGaussianLorentzian{Float64}(; optics...)
    getincident3(z) = SP2T.getincident([1.05 1.05 z;;;], b, detector.darkcounts, detector.pxbounds, psf3)
    @test getincident3(0.0)[11, 11, 1] ≈ 0.01 + b
    @test getincident3(psf3.z₀)[11, 11, 1] ≈ 0.01 + b * SP2T.gaussianpeak(psf3.σ₀ * √2, 0.1) / psf3.A
end

println("Running array-helper tests...")

@testset let A = reshape(collect(1.0:24.0), 2, 3, 4), b = [1.0 0.0 2.0; 0.5 1.0 0.0]
    @test SP2T.framesum!(zeros(4), A, b) ≈ vec(sum(A .* b, dims = (1, 2)))
end

@testset let x = reshape([0.1 * i^2 for i = 1:24], 4, 3, 2),
    y = reshape([0.2 * i for i = 1:24], 4, 3, 2)

    @test SP2T.diff²!(zeros(3, 3, 2), x, y) ≈ (x[2:end, :, :] .- y[1:end-1, :, :]) .^ 2
    @test SP2T.diff²!(zeros(3, 3, 2), x) ≈ diff(x, dims = 1) .^ 2
end

# Odd frames come from `odd`, even frames from `even`; the result is the squared
# displacement of that interleaved track.
@testset for nframes in (5, 6)
    odd = reshape([0.1 * i^2 for i = 1:nframes*4], nframes, 2, 2)
    even = reshape([0.3 * i for i = 1:nframes*4], nframes, 2, 2)
    interleaved = [isodd(k) ? odd[k, d, m] : even[k, d, m] for k = 1:nframes, d = 1:2, m = 1:2]
    Δx² = SP2T.staggered_diff²!(zeros(nframes - 1, 2, 2), even = even, odd = odd)
    @test Δx² ≈ diff(interleaved, dims = 1) .^ 2
end

println("Running chain tests...")

newsample(iteration, log𝒫 = NaN, logℒ = NaN) =
    Sample(zeros(1, 1, 1), 1.0, 1.0, iteration, 1.0, log𝒫, logℒ)

# The saving logic of runMCMC!: the chain never exceeds sizelimit, keeps iteration 0,
# stays equally spaced at a power-of-two stride, keeps at least half its capacity once
# filled, and misses no more than one save stride at the end.
@testset for sizelimit in (4, 5, 10), niters in (3, 17, 100, 1000)
    chain = Chain([newsample(0)], sizelimit, ConstantAnnealing{Float64}(1))
    for iter = 1:niters
        if SP2T.tosave(chain, iter)
            SP2T.isfull(chain) && SP2T.shrink!(chain)
            push!(chain.samples, newsample(iter))
        end
    end
    iterations = chain.iterations
    spacing = iterations[2] - iterations[1]
    @test length(chain) <= sizelimit
    @test niters < sizelimit || length(chain) > sizelimit ÷ 2
    @test first(iterations) == 0
    @test all(==(spacing), diff(iterations))
    @test ispow2(spacing)
    @test niters - last(iterations) < SP2T.savestride(chain)
end

@testset let chain = Chain(
        [newsample(i - 1, log𝒫, logℒ) for (i, log𝒫, logℒ) in
            zip(1:5, [1.0, 5.0, 3.0, 9.0, 2.0], [4.0, 8.0, 6.0, 0.0, 7.0])],
        10,
        ConstantAnnealing{Float64}(1),
    )

    @test findmap(chain) == (chain.samples[4], 4)
    @test findmap(chain, burn_in = 4) == (chain.samples[5], 5)
    @test findmle(chain) == (chain.samples[2], 2)
    @test findmle(chain, burn_in = 2) == (chain.samples[5], 5)
end

# Steps of (3, 4) and (5, 12) for one particle, (0, 1) and (0, 2) for the other.
@testset let tracks = cat([0.0 0.0; 3.0 4.0; 8.0 16.0], [0.0 0.0; 0.0 1.0; 0.0 3.0], dims = 3),
    chain = Chain(
        [Sample(zero(tracks), 1.0, 1.0, 0, 1.0, NaN, NaN), Sample(tracks, 1.0, 1.0, 1, 1.0, NaN, NaN)],
        10,
        ConstantAnnealing{Float64}(1),
    )

    @test jumpdistances(tracks) == reshape([5.0, 13.0, 1.0, 2.0], 2, 1, 2)
    @test jumpdistances(chain, burn_in = 1) == [5.0, 13.0, 1.0, 2.0]
    @test jumpdistances(chain) == [0.0, 0.0, 0.0, 0.0, 5.0, 13.0, 1.0, 2.0]
end

include("samplers.jl")
