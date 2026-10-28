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
    onshuffle!(t::Tracks)

Shuffles the tracks of the emitting particles in `t`.
"""
function onshuffle!(t::Tracks)
    perm = randperm(t.nemitters.value)
    isequal(perm, 1:t.nemitters.value) || _permute!(t, perm)
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
    update!(tracksᵒ::TrackChunk{T}, tracksᵖ::MHTrackChunk{T}, msdᵥ::T)

Swap particle identities partway through the tracks. Propose a random relabeling `perm` of the particles and, independently for each boundary between frames `k - 1` and `k`, accept relabeling frames `k` onward by `perm` according to the change in the motion log-prior across that boundary. Positions in each frame are unchanged, so the likelihood and the first-frame prior are too. Accepted relabelings compose: frames after `c` accepted boundaries are relabeled by `perm` applied `c` times, which leaves the steps across every other boundary unchanged. `active` is relabeled with the positions.
"""
function update!(tracksᵒ::TrackChunk{T}, tracksᵖ::MHTrackChunk{T}, msdᵥ::T) where {T}
    nparticles = size(tracksᵒ.value, 3)
    perm = randperm(nparticles)
    isequal(perm, 1:nparticles) && return tracksᵒ
    initmh!(tracksᵖ)
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
    relabel!(tracksᵒ.active, relabelindices(tracksᵒ.active, perm, accepted), tracksᵖ.active)
    copyto!(tracksᵖ.active, tracksᵒ.active) # proposals share the current mask
    return tracksᵒ
end
