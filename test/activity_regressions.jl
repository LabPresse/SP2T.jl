using SP2T, Test

@testset "Inactive emitters contribute no signal" begin
    for T in (Float32, Float64), dims in (2, 3)
        guess = zeros(T, 3, dims, 1)
        guess[2, :, :] .= T(0.5)
        activity = reshape(Bool[false, true, false], 3, 1, 1)
        tracks = Tracks{T}(
            guess = guess, prior = DNormal{T}(zeros(T, dims), ones(T, dims)),
            max_ntracks = 2, scaling = T(0.01), logonprob = T(-1), active = activity,
        )
        @test eltype(tracks.active[1]) == T
        @test tracks.proposals.active == tracks.onchunk.active
        SP2T.seteffvalue!(tracks.onchunk)
        @test !any(isnan, tracks.onchunk.effvalue)
        SP2T.seteffvalue!(tracks)
        @test tracks.effvalue[1][:, :, 1:1] == tracks.onchunk.effvalue
        tracks.proposals.active .= one(T)
        SP2T.propose!(tracks.proposals, tracks.onchunk)
        @test tracks.proposals.active == tracks.onchunk.active
        SP2T.seteffvalue!(tracks.proposals)

        detector = SPAD{T}(
            period = T(1), pixel_size = T(1), darkcounts = fill(T(0.1), 2, 2),
            cutoffs = (0, Inf), readouts = zeros(UInt16, 2, 2, 3),
        )
        psf = dims == 2 ? CircularGaussian{T}(T(0.4), T(1)) :
              CircularGaussianLorentzian{T}(T(0.4), T(0.8), T(1))
        ll = SP2T.LogLikelihoodArray{T}(detector.readouts)
        for positions in (tracks.onchunk.effvalue, tracks.proposals.effvalue)
            SP2T.set_poisson_mean!(ll, detector, positions, T(2), psf)
            @test all(isfinite, ll.means[1])
            @test ll.means[1][:, :, 1] == detector.darkcounts
            @test ll.means[1][:, :, 3] == detector.darkcounts
            @test all(ll.means[1][:, :, 2] .> detector.darkcounts)
        end
        bad = ones(T, 3, dims, 1)
        bad[1, 1, 1] = 0
        @test_throws ArgumentError Tracks{T}(
            guess = guess, prior = tracks.onchunk.prior, max_ntracks = 1,
            scaling = T(0.01), logonprob = T(-1), active = bad,
        )
    end
end

@testset "Excluded pixels cannot contaminate likelihood reductions" begin
    for T in (Float32, Float64), invalid in (T(-Inf), T(NaN))
        detector = SPAD{T}(
            period = T(1), pixel_size = T(1), darkcounts = fill(T(0.1), 2, 1),
            cutoffs = (0, Inf), readouts = zeros(UInt16, 2, 1, 2),
        )
        detector.filter[1, 1] = 0
        ll = SP2T.LogLikelihoodArray{T}(detector.readouts)
        ll.pixel[1, 1, :] .= invalid
        ll.pixel[2, 1, :] .= T(-3)
        SP2T.sum_pixel_loglikelihood!(ll, detector)
        @test ll.frame == T[-3, -3]
    end
end
