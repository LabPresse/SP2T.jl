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

permuteto!(
    dest::AbstractTrackChunk{T},
    src::AbstractTrackChunk{T},
    perm::AbstractVector{<:Integer},
    start::Integer = 1,
) where {T} = permuteto!(dest.value, src.value, perm, start)

permuteto!(
    dest::AbstractArray{T,3},
    src::AbstractArray{T,3},
    perm::AbstractVector{<:Integer},
    start::Integer,
    step::Integer,
) where {T} = @views copyto!(dest[start:step:end, :, :], src[start:step:end, :, perm])

"""
    _permute!( x::AbstractArray{T,3}, perm::AbstractVector{<:Integer}, y::AbstractArray{T,3})

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
    propagateperm!(x::AbstractArray{T,3}, y::AbstractArray{T,3}, perm::AbstractVector{<:Integer}, ind::AbstractVector{<:Integer})

Apply the permutation `perm` to the 3rd dimension of `x` at each frame index specified in `ind`, and propagate the permutation to the last frame. Note that `y[i, :, :]` already contains the permuted values for `x[i, :, :]`. 
"""
function propagateperm!(
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    perm::AbstractVector{<:Integer},
    ind::AbstractVector{<:Integer},
) where {T}
    if length(ind) == 1
        i = only(ind)
        copyto!(x[i:end, :, :], y[i:end, :, :])
    else
        @views for i in ind
            copyto!(x[i:end, :, :], y[i:end, :, :])
            permuteto!(y, x, perm, i + 1)
        end
    end
    return x
end

function propagateperm!(
    dest::AbstractTrackChunk{T},
    src::AbstractTrackChunk{T},
    perm::AbstractVector{<:Integer},
    pos::AbstractVector{<:Integer},
) where {T}
    propagateperm!(dest.value, src.value, perm, pos)
    propagateperm!(dest.active, src.active, perm, pos)
    return dest
end

function update!(tracksᵒ::TrackChunk{T}, tracksᵖ::MHTrackChunk{T}, msdᵥ::T) where {T}
    initmh!(tracksᵖ)
    # propose a random permuataion for the emitting particles
    perm = randperm(size(tracksᵒ.value, 3))
    # apply the permutation to proposals
    permuteto!(tracksᵖ, tracksᵒ, perm, 2)
    # calculate acceptance ratio
    setdisplacement²!(tracksᵒ)
    diff²!(tracksᵖ.displacement², tracksᵖ.value, tracksᵒ.value)
    sumΔdisplacement²!(tracksᵖ, tracksᵒ, msdᵥ)
    @views tracksᵖ.logacceptance[2:end] .+= tracksᵖ.ΣΔdisplacement²
    setacceptance!(tracksᵖ, start = 2)
    # apply the permutation to the accepted frame indices
    pos = findall(convert(Vector{Bool}, tracksᵖ.accepted))
    isempty(pos) || propagateperm!(tracksᵒ, tracksᵖ, perm, pos)
    return tracksᵒ
end
