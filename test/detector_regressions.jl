using Test
using SP2T

function spad_reference(readout, batchsize, mean)
    setprecision(BigFloat, 2048) do
        photonmean = BigFloat(mean)
        hit = -expm1(-photonmean)
        miss = exp(-photonmean)
        probability = hit^readout * miss^(batchsize - readout)
        log(probability)
    end
end

@testset "SPAD likelihood against binomial hit/miss probabilities" begin
    for T in (Float32, Float64)
        n = UInt16(8)
        detector = SPAD{T}(
            period = 1,
            pixel_size = 1,
            darkcounts = ones(T, 1, 1),
            cutoffs = (0, Inf),
            readouts = zeros(UInt16, 1, 1, 1),
            batchsize = n,
        )
        llarray = SP2T.LogLikelihoodArray{T}(detector.readouts)
        for k in UInt16.((0, 1, 4, 8)), mean in T.((0, floatmin(T), 1e-8, 0.1, 1, 10, 100, 1e5))
            detector.readouts[1] = k
            llarray.means[1][1] = mean
            SP2T.set_pixel_loglikelihood!(llarray, detector)
            expected = T(spad_reference(k, n, mean))
            @test !isnan(llarray.pixel[1])
            @test llarray.pixel[1] ≈ expected rtol=32eps(T) atol=16nextfloat(zero(T))
        end
        for k in UInt16.((0, 1, 4, 8)), (oldmean, newmean) in (
            (T(1e-8), T(2e-8)),
            (T(0.1), T(0.2)),
            (T(10), T(11)),
            (T(100), T(101)),
            (T(1e5), T(1e5 + 1)),
        )
            detector.readouts[1] = k
            llarray.means[1][1] = oldmean
            llarray.means[2][1] = newmean
            SP2T.set_pixel_Δloglikelihood!(llarray, detector)
            expected = T(spad_reference(k, n, newmean) - spad_reference(k, n, oldmean))
            @test llarray.pixel[1] ≈ expected rtol=64eps(T) atol=16nextfloat(zero(T))
        end
        @test SP2T._spad_loglikelihood(UInt16(0), n, zero(T)) == zero(T)
        @test SP2T._spad_loglikelihood(n, n, T(Inf)) == zero(T)
        @test SP2T._spad_loglikelihood(UInt16(0), n, T(Inf)) == -Inf
        @test SP2T._spad_Δloglikelihood(UInt16(0), n, zero(T), T(0.2)) ≈ -T(n) * T(0.2)
        @test SP2T._spad_Δloglikelihood(UInt16(0), n, zero(T), zero(T)) == zero(T)
        @test SP2T._spad_Δloglikelihood(n, n, T(Inf), T(Inf)) == zero(T)
        @test SP2T._spad_Δloglikelihood(n, n, zero(T), one(T)) == Inf
        @test SP2T._spad_Δloglikelihood(n, n, one(T), zero(T)) == -Inf
    end
end
