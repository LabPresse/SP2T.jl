# References use scalar distribution densities or explicitly enumerated weighted sums.
# This same suite runs on CPU and, from cuda.jl, CUDA with scalar indexing disabled.
function detector_regression_suite(adapt = identity; precisions = (Float32, Float64))
    @testset "Detector regressions $(adapt) $(T)" for T in precisions
        atol = T === Float32 ? 3e-5 : 2e-10
        rtol = T === Float32 ? 5e-5 : 5e-12
        weights = reshape(T[1, 0, 0.5, 0], 2, 2)
        values = reshape(T[2, Inf, 6, NaN, -4, NaN, 8, -Inf], 2, 2, 2)
        output = adapt(T[Inf, NaN])
        devicevalues = adapt(values)
        SP2T.framesum!(output, devicevalues, adapt(weights))
        @test Array(output) == T[5, 0]
        @test isequal(Array(devicevalues), values)
        SP2T.framesum!(output, devicevalues, adapt(zeros(T, 2, 2)))
        @test Array(output) == zeros(T, 2)
        included = copy(values)
        included[1, 1, 1] = T(Inf)
        SP2T.framesum!(output, adapt(included), adapt(weights))
        @test isinf(Array(output)[1])

        dest = adapt(reshape(T[1, 2, -floatmax(T)], 3, 1, 1))
        proposed = adapt(reshape(T[NaN, Inf, floatmax(T)], 3, 1, 1))
        SP2T.boolcopyto!(dest, proposed, adapt(T[0, 0, 1]))
        @test vec(Array(dest)) == T[1, 2, floatmax(T)]

        counts = reshape(UInt16[1, 2, 3, 4], 2, 2, 1)
        dark = reshape(T[0.1, 0.2, 0.3, 0.4], 2, 2)
        common = (
            period = T(0.1), pixel_size = T(1), darkcounts = adapt(dark),
            cutoffs = (0, Inf), readouts = adapt(counts),
        )
        detector = SP2T.EMCCDGamma{T}(; common..., gain = T(2), noise_excess_factor = T(2))
        ll = SP2T.LogLikelihoodArray{T}(detector.readouts)
        old = reshape(T[0.5, 1, 2, 4], 2, 2, 1)
        new = reshape(T[0.7, 1.2, 1.8, 3.4], 2, 2, 1)
        ll.means[1] .= adapt(old)
        ll.means[2] .= adapt(new)
        reference = sum(logpdf(Gamma(Float64(new[i]) / 2, 4), Float64(counts[i])) -
                        logpdf(Gamma(Float64(old[i]) / 2, 4), Float64(counts[i]))
                        for i in eachindex(old))
        delta = SP2T.get_Δloglikelihood!(ll, detector)
        @test isapprox(delta, reference; atol, rtol)
        @test isequal(SP2T.get_Δloglikelihood!(ll, detector), delta)
        @test Array(ll.means[1]) == old
        @test Array(ll.means[2]) == new
        full = SP2T.get_loglikelihood!(ll, detector)
        # The implementation drops the readout-only -y/scale term.
        referencefull = sum(logpdf(Gamma(Float64(old[i]) / 2, 4), Float64(counts[i])) +
                            Float64(counts[i]) / 4 for i in eachindex(old))
        @test isapprox(full, referencefull; atol, rtol)
        @test isequal(SP2T.get_loglikelihood!(ll, detector), full)
        @test Array(ll.means[1]) == old
        @test Array(ll.means[2]) == new

        # Excluded pixels must not enter loggamma, even when their shapes are invalid.
        maskedgamma = SP2T.EMCCDGamma{T}(period = 1, pixel_size = 1,
            darkcounts = adapt(reshape(T[-1, NaN, 0.3, 0.4], 2, 2)), cutoffs = (0, 10),
            readouts = adapt(counts), gain = 2, noise_excess_factor = 2)
        gold, gnew = copy(old), copy(new)
        gold[1:2] .= T[-1, NaN]
        gnew[1:2] .= T[NaN, -1]
        ll.means[1] .= adapt(gold)
        ll.means[2] .= adapt(gnew)
        expectedmasked = sum(logpdf(Gamma(Float64(old[i]) / 2, 4), Float64(counts[i])) +
                             Float64(counts[i]) / 4 for i in 3:4)
        expecteddelta = sum(logpdf(Gamma(Float64(new[i]) / 2, 4), Float64(counts[i])) -
                            logpdf(Gamma(Float64(old[i]) / 2, 4), Float64(counts[i])) for i in 3:4)
        @test isapprox(SP2T.get_loglikelihood!(ll, maskedgamma), expectedmasked; atol, rtol)
        @test isapprox(SP2T.get_Δloglikelihood!(ll, maskedgamma), expecteddelta; atol, rtol)
        @test isequal(Array(ll.means[1]), gold) && isequal(Array(ll.means[2]), gnew)

        gaussian = EMCCD{T}(; common..., gain = T(2), offset = T(0.3), variance = T(1.2))
        gll = SP2T.LogLikelihoodArray{T}(gaussian.readouts)
        gll.means[1] .= adapt(old)
        gll.means[2] .= adapt(new)
        SP2T.set_pixel_Δloglikelihood!(gll, gaussian)
        gaussreference = [
            logpdf(Normal(2 * Float64(new[i]) + Float64(T(0.3)), sqrt(Float64(T(1.2)))), counts[i]) -
            logpdf(Normal(2 * Float64(old[i]) + Float64(T(0.3)), sqrt(Float64(T(1.2)))), counts[i])
            for i in eachindex(old)
        ]
        @test isapprox(vec(Array(gll.pixel)), gaussreference; atol, rtol)
        gll.means[2] .= gll.means[1]
        SP2T.set_pixel_Δloglikelihood!(gll, gaussian)
        @test all(iszero, Array(gll.pixel))
        @test isnan(SP2T._emccd_Δloglikelihood(1, T(NaN), T(NaN), T(2), T(0), T(1)))

        spad = SPAD{T}(; common..., batchsize = 4)
        psf = CircularGaussian{T}(T(0.4), T(1))
        coords = adapt(reshape(T[0.3, 0.7], 1, 2, 1))
        # Direct Normal CDF integration is independent of the PSF buffer implementation.
        massx = [cdf(Normal(0.3, 0.4), i) - cdf(Normal(0.3, 0.4), i - 1) for i in 1:2]
        massy = [cdf(Normal(0.7, 0.4), i) - cdf(Normal(0.7, 0.4), i - 1) for i in 1:2]
        for oldbrightness in (T(0), T(1.5)), newbrightness in (T(0), T(2))
            dense = SP2T.LogLikelihoodArray{T}(spad.readouts)
            sparse = SP2T.loglikelihoodaux(spad)
            expected = Float64.(dark) .+ Float64(newbrightness / psf.A) .* (massx * massy')
            for aux in (dense, sparse)
                SP2T.set_poisson_means!(aux, spad, coords, oldbrightness, newbrightness, psf)
                @test all(isfinite, Array(aux.means[2]))
            end
            @test isapprox(Array(dense.means[2])[:, :, 1], expected; atol, rtol)
            @test isapprox(SP2T.get_Δloglikelihood!(dense, spad),
                           SP2T.get_Δloglikelihood!(sparse, spad); atol, rtol)
        end

        calibration = adapt(reshape(T[0, Inf, -1, 2], 2, 2))
        original = Array(calibration)
        SPAD{T}(period = 0.1, pixel_size = 1, darkcounts = calibration,
                cutoffs = (0, Inf), readouts = adapt(counts))
        @test isequal(Array(calibration), original)

        # Excluded calibration values must not enter the new sparse background sum.
        filtered = SPAD{T}(
            period = T(0.1), pixel_size = T(1),
            darkcounts = adapt(reshape(T[NaN, Inf, 0.1, 0.2], 2, 2)),
            cutoffs = (0, 10), readouts = adapt(ones(UInt16, 2, 2, 1)), batchsize = 4,
        )
        referencefiltered = sum(logpdf(Binomial(4, -expm1(-Float64(d))), 1) - log(4)
                                for d in T[0.1, 0.2])
        for aux in (SP2T.LogLikelihoodArray{T}(filtered.readouts), SP2T.loglikelihoodaux(filtered))
            SP2T.reset!(aux, filtered, 1)
            @test isapprox(SP2T.get_loglikelihood!(aux, filtered), referencefiltered; atol, rtol)
            SP2T.set_poisson_mean!(aux, filtered, coords, T(2), psf)
            expected = [Float64(T(0.1)) + 2 / Float64(psf.A) * massx[1] * massy[2],
                        Float64(T(0.2)) + 2 / Float64(psf.A) * massx[2] * massy[2]]
            referencepositive = sum(logpdf(Binomial(4, -expm1(-d)), 1) - log(4)
                                    for d in expected)
            @test isapprox(SP2T.get_loglikelihood!(aux, filtered), referencepositive; atol, rtol)
        end
    end
end
