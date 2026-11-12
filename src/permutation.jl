"""
    permuteto!(dest::AbstractArray{T,3}, src::AbstractArray{T,3}, perm::AbstractVector{<:Integer}, start::Integer = 1)

Permute the 3rd dimension (particle index) of `src` according to the order specified in `perm` and copy the result to `dest`. The 1st and 2nd dimensions are not permuted. The `start` argument specifies the first frame index.
"""
permuteto!(
    dest::AbstractArray{T,3},
    src::AbstractArray{T,3},
    perm::AbstractVector{<:Integer},
    start::Integer = 1,
) where {T} = @views copyto!(dest[start:end, :, :], src[start:end, :, perm])

"""
    _permute!(x::AbstractArray{T,3}, perm::AbstractVector{<:Integer}, y::AbstractArray{T,3})

Permute the 3rd dimension (particle index) of `x` according to the order specified in `p`. `y` is used as a temporary array to store the permuted result.
"""
function _permute!(
    x::AbstractArray{T,3},
    perm::AbstractVector{<:Integer},
    y::AbstractArray{T,3},
) where {T}
    permuteto!(y, x, perm)
    copyto!(x, y)
end

function _permute!(t::Tracks{T}, perm::AbstractVector{<:Integer}) where {T}
    _permute!(t.onchunk.value, perm, t.proposals.value)
    _permute!(t.onchunk.active, perm, t.proposals.active)
    return t
end

"""
    onshuffle!(rng::AbstractRNG, t::Tracks)

Shuffle the emitting tracks only when the proposed relabeling preserves their fixed
activity masks. Different fixed activity histories are not exchangeable.
"""
function onshuffle!(rng::AbstractRNG, t::Tracks)
    perm = randperm(rng, t.nemitters.value)
    if !isequal(perm, 1:t.nemitters.value)
        @views all(t.onchunk.active .== t.onchunk.active[:, :, perm]) || return t
        _permute!(t, perm)
    end
    return t
end

"""
    relabelindices(x::AbstractArray{T,3}, perm::AbstractVector{<:Integer}, accepted::AbstractVector{Bool})

Return the linear indices that relabel the particles (3rd dimension) of `x` frame by frame: frame `k` is relabeled by `perm` applied once for every accepted boundary up to and including `k`. The indices are built on the CPU and returned on the device of `x`.
"""
function relabelindices(
    x::AbstractArray{T,3},
    perm::AbstractVector{<:Integer},
    accepted::AbstractVector{Bool},
) where {T}
    nframes, ndims, nparticles = size(x)
    σ = collect(1:nparticles)
    i = Array{Int}(undef, nframes, ndims, nparticles)
    for k = 1:nframes
        accepted[k] && (σ = σ[perm])
        for m = 1:nparticles, d = 1:ndims
            i[k, d, m] = k + nframes * (d - 1 + ndims * (σ[m] - 1))
        end
    end
    return copyto!(similar(x, Int, size(i)), i)
end

"""
    relabel!(x::AbstractArray{T,3}, i::AbstractArray{<:Integer,3}, scratch::AbstractArray{T,3})

Overwrite `x` with `x[i]` in one gather. `scratch` is a temporary array the size of `x`.
"""
function relabel!(
    x::AbstractArray{T,3},
    i::AbstractArray{<:Integer,3},
    scratch::AbstractArray{T,3},
) where {T}
    copyto!(scratch, x)
    x .= scratch[i]
    return x
end

"""
    update!(rng::AbstractRNG, tracksᵒ::TrackChunk{T}, tracksᵖ::MHTrackChunk{T}, msdᵥ::T)

Swap two particle identities partway through their tracks. Draw two indices uniformly
with replacement; equal indices give a no-op. Otherwise use their transposition at
every boundary, with independent Metropolis decisions from the motion log-prior.
Only particles with identical fixed activity histories may be exchanged, so neither
the mask nor the image likelihood changes. The first frame is never relabeled.

The shared permutation must be an involution. For a transposition, accepted boundary
flips commute and are their own inverses; rejected-boundary probabilities also match
in the reverse move. Arbitrary shared permutations do not have this property when
several boundaries are tested. With two particles the proposal law remains one-half
identity and one-half swap.
"""
function update!(
    rng::AbstractRNG,
    tracksᵒ::TrackChunk{T},
    tracksᵖ::MHTrackChunk{T},
    msdᵥ::T,
) where {T}
    nparticles = size(tracksᵒ.value, 3)
    (nparticles < 2 || size(tracksᵒ.value, 1) < 2) && return tracksᵒ
    a, b = rand(rng, 1:nparticles), rand(rng, 1:nparticles)
    a == b && return tracksᵒ
    @views all(tracksᵒ.active[:, :, a:a] .== tracksᵒ.active[:, :, b:b]) || return tracksᵒ
    perm = collect(1:nparticles)
    perm[a], perm[b] = b, a
    initmh!(rng, tracksᵖ)
    permuteto!(tracksᵖ.value, tracksᵒ.value, perm, 2)
    setdisplacement²!(tracksᵒ)
    diff²!(tracksᵖ.displacement², tracksᵖ.value, tracksᵒ.value)
    sumΔdisplacement²!(tracksᵖ, tracksᵒ, msdᵥ)
    @views tracksᵖ.logacceptance[2:end] .+= tracksᵖ.ΣΔdisplacement²
    setacceptance!(tracksᵖ, start = 2, step = 1)
    accepted = convert(Vector{Bool}, tracksᵖ.accepted)
    @views tracksᵖ.counter[:, 1] .+= count(accepted), length(accepted) - 1
    any(accepted) || return tracksᵒ
    relabel!(tracksᵒ.value, relabelindices(tracksᵒ.value, perm, accepted), tracksᵖ.value)
    copyto!(tracksᵖ.active, tracksᵒ.active) # the exchanged masks are identical
    return tracksᵒ
end
