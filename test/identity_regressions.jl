using Test, Random, StableRNGs

# Deterministic draws select a proposed pair and a point inside each acceptance or
# rejection interval. This exercises the actual update!, not a replica of its map.
mutable struct IdentityFixtureRNG <: AbstractRNG
    pair::NTuple{2,Int}
    draw::Int
    uniforms::Vector{Float64}
end

function Random.rand(rng::IdentityFixtureRNG, range::UnitRange{Int})
    rng.draw += 1
    @assert rng.draw <= 2
    value = rng.pair[rng.draw]
    @assert value in range
    return value
end

function Random.rand!(rng::IdentityFixtureRNG, x::AbstractArray{T}) where {T}
    @assert length(x) == length(rng.uniforms)
    copyto!(x, T.(rng.uniforms))
    return x
end

struct IdentityShuffleRNG <: AbstractRNG
    permutation::Vector{Int}
end
Random.randperm(rng::IdentityShuffleRNG, n::Integer) =
    (@assert length(rng.permutation) == n; copy(rng.permutation))

function identity_permutations(n)
    result = Vector{Vector{Int}}()
    function visit!(prefix, unused)
        if isempty(unused)
            push!(result, copy(prefix))
        else
            for value in unused
                visit!([prefix; value], filter(!=(value), unused))
            end
        end
    end
    visit!(Int[], collect(1:n))
    return result
end

function identity_fixture(x; active = ones(size(x, 1), 1, size(x, 3)))
    T = eltype(x)
    return Tracks{T}(
        guess = x,
        prior = SP2T.DNormal{T}(zeros(T, size(x, 2)), ones(T, size(x, 2))),
        max_ntracks = size(x, 3),
        scaling = zero(T),
        logonprob = -one(T),
        active = T.(active),
    )
end

# Independent reference: edit each accepted suffix in chronological order. This
# deliberately does not use relabelindices, cumulative powers, or package energies.
function identity_tail_swap(x, a, b, boundary)
    result = copy(x)
    result[boundary:end, :, a] = x[boundary:end, :, b]
    result[boundary:end, :, b] = x[boundary:end, :, a]
    return result
end
identity_motion_logweight(x) = -sum(abs2, diff(x, dims = 1)) / 2

@testset "Identity swap exact finite detailed balance" begin
    # Dyadic coordinates make independent energy differences exactly representable.
    # Fix frame1 labels; frame2 and frame3 each admit all 3! labelings: 36 states.
    points = [0.0 1.0 3.0; 0.25 2.0 3.5; -0.5 1.25 4.0]
    permutations = identity_permutations(3)
    states = Array{Float64,3}[]
    for p in permutations, q in permutations
        x = zeros(3, 2, 3)
        x[1, 1, :] = points[1, :]
        x[2, 1, :] = points[2, p]
        x[3, 1, :] = points[3, q]
        push!(states, x)
    end
    lookup = Dict(Tuple(vec(x)) => i for (i, x) in enumerate(states))
    logweights = identity_motion_logweight.(states)
    weights = exp.(logweights .- maximum(logweights))
    weights ./= sum(weights)
    transition = zeros(36, 36)
    checked_events = 0
    for (i, x) in enumerate(states), a = 1:3, b = 1:3
        deltas = [
            identity_motion_logweight(identity_tail_swap(x, a, b, k)) -
            identity_motion_logweight(x) for k = 2:3
        ]
        probabilities = min.(1.0, exp.(deltas))
        for mask = 0:3
            accepted = [!iszero(mask & (1 << (k - 1))) for k = 1:2]
            probability = prod(accepted[k] ? probabilities[k] : 1 - probabilities[k]
                               for k = 1:2) / 9
            iszero(probability) && continue
            expected = copy(x)
            uniforms = [0.5; [accepted[k] ? probabilities[k] / 2 :
                              (1 + probabilities[k]) / 2 for k = 1:2]]
            for k = 1:2
                accepted[k] && (expected = identity_tail_swap(expected, a, b, k + 1))
            end
            tracks = identity_fixture(copy(x))
            rng = IdentityFixtureRNG((a, b), 0, uniforms)
            SP2T.update!(rng, tracks.onchunk, tracks.proposals, 1.0)
            @test tracks.onchunk.value == expected
            @test tracks.onchunk.active == tracks.proposals.active == ones(3, 1, 3)
            @test tracks.onchunk.value[1, :, :] == x[1, :, :]
            transition[i, lookup[Tuple(vec(tracks.onchunk.value))]] += probability
            checked_events += 1
        end
    end
    # These are roundoff bounds for a finite sum, not Monte Carlo tolerances.
    @test checked_events > 324
    @test maximum(abs.(sum(transition, dims = 2) .- 1)) < 2e-14
    @test maximum(abs.(vec(weights' * transition) .- weights)) < 2e-14
    flux = weights .* transition
    @test maximum(abs.(flux .- transpose(flux))) < 2e-14
end

@testset "Identity swap fixed activity and degenerate cases" begin
    x = reshape(collect(1.0:18.0), 3, 2, 3)
    # A fixed history cannot follow a proposed physical identity across a cut.
    active = reshape([1., 0, 1, 0, 1, 1, 1, 1, 0], 3, 1, 3)
    for a = 1:3, b = 1:3
        tracks = identity_fixture(copy(x); active)
        SP2T.update!(IdentityFixtureRNG((a, b), 0, fill(0.5, 3)),
                     tracks.onchunk, tracks.proposals, 1.0)
        @test tracks.onchunk.value == x
        @test tracks.onchunk.active == tracks.proposals.active == active
        @test all(iszero, tracks.proposals.counter)
    end
    for p in identity_permutations(3)
        tracks = identity_fixture(copy(x); active)
        SP2T.onshuffle!(IdentityShuffleRNG(p), tracks)
        @test tracks.onchunk.value == x
        @test tracks.onchunk.active == tracks.proposals.active == active
    end
    # A whole-track permutation within an equal-mask class remains permissible.
    active[:, :, 2] = active[:, :, 1]
    tracks = identity_fixture(copy(x); active)
    SP2T.onshuffle!(IdentityShuffleRNG([2, 1, 3]), tracks)
    @test tracks.onchunk.value == x[:, :, [2, 1, 3]]
    @test tracks.onchunk.active == tracks.proposals.active == active
    for T in (Float32, Float64), nframes in (1, 3), nparticles in (0, 1)
        tracks = identity_fixture(zeros(T, nframes, 2, nparticles))
        SP2T.update!(IdentityFixtureRNG((1, 1), 0, fill(0.5, nframes)),
                     tracks.onchunk, tracks.proposals, one(T))
        @test size(tracks.onchunk.value) == (nframes, 2, nparticles)
        @test all(iszero, tracks.proposals.counter)
    end
end

@testset "Two-emitter multi-cut swap with common blinking" begin
    nframes = 10
    x = [d == 1 ? 0.1k : Float64(k <= 5 ? m - 1 : 2 - m)
         for k = 1:nframes, d = 1:2, m = 1:2]
    active = [Float64(!(k in (3, 8))) for k = 1:nframes, d = 1:1, m = 1:2]
    tracks = identity_fixture(copy(x); active)
    rng = StableRNG(2)
    for _ = 1:50
        SP2T.update!(rng, tracks.onchunk, tracks.proposals, 1e-4)
    end
    @test all(m -> allequal(tracks.onchunk.value[:, 2, m]), 1:2)
    @test tracks.onchunk.value[:, 1, :] == x[:, 1, :]
    @test tracks.onchunk.active == tracks.proposals.active == active
end
