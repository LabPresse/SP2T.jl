using Test, Distributions

# Independent binomial probability, not the implementation's log-space formula.
function spad_probability_reference(r, B, mean)
    setprecision(BigFloat, 2048) do
        λ = BigFloat(mean)
        hit, miss = -expm1(-λ), exp(-λ)
        log(hit^Int(r) * miss^(Int(B) - Int(r)))
    end
end

function spad_reference_match(actual, expected, ::Type{T}, mult; allow_subnormal_zero = false) where {T}
    # Metal Shading Language §8.1 permits Float32 arithmetic to flush a
    # subnormal result to zero. Classify the host-rounded BigFloat reference.
    if allow_subnormal_zero && T === Float32 && issubnormal(expected) && iszero(actual)
        return true
    end
    return actual == expected || (isfinite(actual) && isfinite(expected) &&
        isapprox(actual, expected; rtol = mult * eps(T), atol = 16 * nextfloat(zero(T))))
end

function spad_saturation_regression_suite(adapt = identity; precisions = (Float32, Float64),
    allow_subnormal_zero = false)
    @testset "SPAD finite, saturated, and close-mean references $(adapt) $(T)" for T in precisions
        B = UInt16(8)
        for r in UInt16.((0, 1, 4, 8))
            detector = SPAD{T}(period = 1, pixel_size = 1, darkcounts = adapt(ones(T, 1, 1)),
                cutoffs = (0, Inf), readouts = adapt(fill(r, 1, 1, 1)), batchsize = B)
            dense = SP2T.LogLikelihoodArray{T}(detector.readouts)
            sparse = SP2T.loglikelihoodaux(detector)
            # Supply the same one-pixel photon means to each workspace. In sparse
            # storage the sole pixel is either a click or a zero-readout linear term.
            fill!(sparse.darksum, 0)
            function setmeans!(λ, i)
                fill!(dense.means[i], λ)
                fill!(sparse.means[i], iszero(r) ? zero(T) : λ)
                fill!(sparse.linear[i], iszero(r) ? λ : zero(T))
            end
            metal_full_hit = allow_subnormal_zero && T === Float32 && r == B
            means = T[0, floatmin(T), 1e-8, 0.1, 1, 10, 100, 1e5, Inf]
            # Both totals are normal Float32 tails; at 88 only the per-hit
            # factor is subnormal. The count-weighted score must survive.
            metal_full_hit && append!(means, T[80, 88])
            for λ in means
                setmeans!(λ, 1)
                expected = T(spad_probability_reference(r, B, λ))
                for aux in (dense, sparse)
                    actual = SP2T.get_loglikelihood!(aux, detector)
                    @test spad_reference_match(actual, expected, T, 32;
                        allow_subnormal_zero = metal_full_hit)
                end
            end
            pairs = [(T(1e-8), T(2e-8)), (T(0.1), T(0.2)), (T(10), T(11)),
                (T(100), T(101)), (T(1e5), T(1e5 + 1)),
                (T(0.7), nextfloat(T(0.7))), (nextfloat(T(0.7)), T(0.7))]
            if r == B
                append!(pairs, [(T(Inf), T(Inf)), (zero(T), one(T)), (one(T), zero(T))])
                metal_full_hit && append!(pairs, [(T(80), T(81)), (T(88), T(89))])
            elseif iszero(r)
                append!(pairs, [(zero(T), zero(T)), (zero(T), T(0.2))])
            end
            for (old, new) in pairs
                setmeans!(old, 1); setmeans!(new, 2)
                expected = setprecision(BigFloat, 2048) do
                    T(spad_probability_reference(r, B, new) - spad_probability_reference(r, B, old))
                end
                for aux in (dense, sparse)
                    actual = SP2T.get_Δloglikelihood!(aux, detector)
                    @test spad_reference_match(actual, expected, T, 64;
                        allow_subnormal_zero = metal_full_hit)
                end
            end
        end

        # No Bernoulli trials: the unique zero count has probability one even
        # at infinite intensity. In particular, do not evaluate 0*(Inf-Inf).
        no_trials = SPAD{T}(period = 1, pixel_size = 1, darkcounts = adapt(ones(T, 1, 1)),
            cutoffs = (0, Inf), readouts = adapt(zeros(UInt16, 1, 1, 1)), batchsize = 0)
        dense0 = SP2T.LogLikelihoodArray{T}(no_trials.readouts)
        sparse0 = SP2T.loglikelihoodaux(no_trials)
        for i = 1:2
            fill!(dense0.means[i], T(Inf)); fill!(sparse0.linear[i], T(Inf))
        end
        for aux in (dense0, sparse0)
            @test iszero(SP2T.get_loglikelihood!(aux, no_trials))
            @test iszero(SP2T.get_Δloglikelihood!(aux, no_trials))
        end

        # All-zero, all-click and mixed frames; filtered negative/NaN entries
        # include positive readouts. Padding points at the negative first pixel.
        dark = T[-1 0.2; NaN 0.3; 0.4 0.6]
        counts = cat(UInt16[2 0; 1 0; 0 0], UInt16[3 8; 4 2; 5 1],
            UInt16[1 8; 2 0; 0 3], UInt16[2 0; 2 7; 0 0]; dims = 3)
        d = SPAD{T}(period = 1, pixel_size = 1, darkcounts = adapt(dark), cutoffs = (0, Inf),
            readouts = adapt(counts), batchsize = B)
        coords = [T(dim == 1 ? 0.3 + 0.1k : 0.7 - 0.04k) for k = 1:4, dim = 1:2, m = 1:1]
        psf = CircularGaussian{T}(T(0.45), one(T))
        σ = Float64(T(0.45)); A = (cdf(Normal(0, σ), 0.5) - cdf(Normal(0, σ), -0.5))^2
        signal = zeros(Float64, 3, 2, 4)
        for k = 1:4, j = 1:2, i = 1:3
            px = Normal(Float64(coords[k, 1, 1]), σ); py = Normal(Float64(coords[k, 2, 1]), σ)
            signal[i, j, k] = (cdf(px, i) - cdf(px, i - 1)) *
                              (cdf(py, j) - cdf(py, j - 1)) / A
        end
        valid = isfinite.(dark) .& (dark .> 0)
        function framescores(b)
            [sum(Float64(spad_probability_reference(counts[i, j, k], B,
                Float64(dark[i, j]) + Float64(b) * signal[i, j, k]))
                for j = 1:2, i = 1:3 if valid[i, j]) for k = 1:4]
        end
        dense = SP2T.LogLikelihoodArray{T}(d.readouts); sparse = SP2T.loglikelihoodaux(d)
        # Column-major click order, padding and PSF offsets for this 3 × 2 movie.
        packed = UInt16[0 5 8 7; 0 8 3 0; 0 2 0 0; 0 1 0 0]
        @test Array(sparse.readouts) == packed
        @test Array(sparse.weights) == T.(packed .> 0)
        @test Array(sparse.xindex) == [1 6 7 11; 1 4 9 10; 1 5 7 10; 1 6 7 10]
        @test Array(sparse.yindex) == [1 3 6 8; 1 4 6 7; 1 4 5 7; 1 4 5 7]
        @test Array(sparse.darkcounts) == T[-1 0.4 0.2 0.3; -1 0.2 0.6 -1;
                                            -1 0.3 -1 -1; -1 0.6 -1 -1]
        @test sparse.zeroruns == [(1:1, :all), (2:2, :none), (3:4, :mixed)]
        excluded = SPAD{T}(period = 1, pixel_size = 1, darkcounts = adapt(fill(T(-1), 3, 2)),
            cutoffs = (0, Inf), readouts = adapt(counts), batchsize = B)
        empty_clicks = SP2T.loglikelihoodaux(excluded)
        @test Array(empty_clicks.readouts) == zeros(UInt16, 1, 4)
        @test Array(empty_clicks.darksum) == zeros(T, 4)
        @test empty_clicks.zeroruns == [(1:4, :none)]
        expected_dark = [sum((Float64(dark[i, j]) for j = 1:2, i = 1:3
            if valid[i, j] && iszero(counts[i, j, k])); init = 0.0) for k = 1:4]
        atol, rtol = T === Float32 ? (3e-5, 5e-5) : (2e-10, 5e-12)
        @test isapprox(Array(sparse.darksum), expected_dark; atol, rtol)
        for (oldb, newb) in ((T(0), T(2)), (T(0.4), T(1.2)), (T(2), T(0)))
            oldscore, newscore = framescores(oldb), framescores(newb)
            @testset "$(nameof(typeof(aux))) brightness $oldb to $newb" for aux in (dense, sparse)
                SP2T.set_poisson_means!(aux, d, adapt(coords), oldb, newb, psf)
                score = SP2T.get_loglikelihood!(aux, d)
                if !isfinite(score)
                    @info "Nonfinite masked SPAD score" workspace=typeof(aux) filter=Array(d.filter) means=Array(aux.means[1]) pixels=Array(aux.pixel) frames=Array(aux.frame)
                end
                @test isapprox(score, sum(oldscore); atol, rtol)
                @test isapprox(Array(aux.frame), oldscore; atol, rtol)
                SP2T.set_frame_Δloglikelihood!(aux, d)
                @test isapprox(Array(aux.frame), newscore - oldscore; atol, rtol)
            end
            for (slot, b) in ((1, oldb), (2, newb))
                expected_signal = [sum((Float64(b) * signal[i, j, k] for j = 1:2, i = 1:3
                    if valid[i, j] && iszero(counts[i, j, k])); init = 0.0) for k = 1:4]
                @test isapprox(Array(sparse.linear[slot]), expected_signal; atol, rtol)
            end
        end

        # Cross every tile boundary (x>128, y>8, frames>16). Binary fractions
        # make the direct positive zero-readout mass exactly representable.
        nx, ny, nf = 129, 9, 17
        reads = [UInt16((i + 2j + k) % 5 == 0) for i = 1:nx, j = 1:ny, k = 1:nf]
        calibration = [T((i + j) % 7 == 0 ? -1 : 1) for i = 1:nx, j = 1:ny]
        td = SPAD{T}(period = 1, pixel_size = 1, darkcounts = adapt(calibration),
            cutoffs = (0, Inf), readouts = adapt(reads), batchsize = 1)
        s = SP2T.loglikelihoodaux(td)
        px = [T(mod(i + k, 4) + 1) / 4 for i = 1:nx, m = 1:1, k = 1:nf]
        py = [T(mod(j + 2k, 4) + 1) / 4 for j = 1:ny, m = 1:1, k = 1:nf]
        result = adapt(zeros(T, nf)); scale = T(0.5)
        SP2T.add_zero_framesums!(result, adapt(px), adapt(py), td, s, scale)
        expected = [sum(scale * px[i, 1, k] * py[j, 1, k] for j = 1:ny, i = 1:nx
            if calibration[i, j] > 0 && iszero(reads[i, j, k])) for k = 1:nf]
        @test Array(result) == expected

        # The public sampler also accumulates more than one emitter. This
        # fixture has more than 128 clicks per frame and a partial final warp.
        X, Y = adapt(px), adapt(py)
        SP2T.reset!(s, td, 1)
        fill!(s.linear[1], T(0.25))
        fill!(s.means[2], T(23))
        fill!(s.linear[2], T(13))
        for _ in 1:2
            SP2T.add_spad_signal!(s, td, X, Y, scale, 1)
        end
        @test Array(s.linear[1]) == T(0.25) .+ 2 .* expected
        click_increment = scale .* px[Array(s.xindex)] .* py[Array(s.yindex)]
        @test Array(s.means[1]) == Array(s.darkcounts) .+ 2 .* click_increment
        @test all(==(T(23)), Array(s.means[2]))
        @test all(==(T(13)), Array(s.linear[2]))

        # A clicked pixel contains unit signal; the sole included miss contains
        # less than half an ulp at 1. An excluded column carries a nonfinite PSF.
        tail = eps(T) / T(8)
        td = SPAD{T}(period = 1, pixel_size = 1, darkcounts = adapt(reshape(T[1, 1, -1], 3, 1)),
            cutoffs = (0, Inf), readouts = adapt(reshape(UInt16[1, 0, 0], 3, 1, 1)), batchsize = 1)
        s = SP2T.loglikelihoodaux(td)
        SP2T.reset!(s, td, 1)
        SP2T.add_spad_signal!(s, td, adapt(reshape(T[1, tail, NaN], 3, 1, 1)),
            adapt(ones(T, 1, 1, 1)), one(T), 1)
        @test only(Array(s.linear[1])) == tail
        @test only(Array(s.means[1])) == T(2)

        # A click in frame 2 leaves frame 1 fully separable. Both spatial axes
        # exceed a warp and the number of pixels is not a multiple of 32.
        counts = zeros(UInt16, 37, 39, 3)
        counts[37, 39, 2] = 1
        counts[:, :, 3] .= 1
        td = SPAD{T}(period = 1, pixel_size = 1, darkcounts = adapt(ones(T, 37, 39)),
            cutoffs = (0, Inf), readouts = adapt(counts), batchsize = 1)
        s = SP2T.loglikelihoodaux(td)
        SP2T.reset!(s, td, 1)
        SP2T.add_spad_signal!(s, td, adapt(ones(T, 37, 1, 3)),
            adapt(ones(T, 39, 1, 3)), T(0.5), 1)
        @test Array(s.linear[1]) == T[37 * 39 / 2, (37 * 39 - 1) / 2, 0]
    end
end
