permuteto!(
    dest::AbstractArray{T,3},
    src::AbstractArray{T,3},
    p::AbstractVector{<:Integer},
    start::Integer = 1,
) where {T} = @views copyto!(dest[start:end, :, :], src[start:end, :, p])

permuteto!(
    dest::AbstractTrackParts{T},
    src::AbstractTrackParts{T},
    p::AbstractVector{<:Integer},
    start::Integer = 1,
) where {T} = permuteto!(dest.value, src.value, p, start)

permuteto!(
    dest::AbstractArray{T,3},
    src::AbstractArray{T,3},
    p::AbstractVector{<:Integer},
    start::Integer,
    step::Integer,
) where {T} = @views copyto!(dest[start:step:end, :, :], src[start:step:end, :, p])

function _permute!(
    x::AbstractArray{T,3},
    p::AbstractVector{<:Integer},
    y::AbstractArray{T,3},
) where {T}
    permuteto!(y, x, p)
    copyto!(x, y)
end

function _permute!(tracks::Tracks{T}, p::AbstractVector{<:Integer}) where {T}
    _permute!(tracks.onpart.value, p, tracks.proposals.value)
    _permute!(tracks.onpart.presence, p, tracks.proposals.presence)
    return tracks
end

function onshuffle!(tracks::Tracks{T}) where {T}
    p = randperm(tracks.ntracks.value)
    isequal(p, 1:tracks.ntracks.value) || _permute!(tracks, p)
    return tracks
end

function propagateperm!(
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    p::AbstractVector{<:Integer},
    pos::AbstractVector{<:Integer},
) where {T}
    if length(pos) == 1
        i = only(pos)
        copyto!(x[i:end, :, :], y[i:end, :, :])
    else
        @views for i in pos
            copyto!(x[i:end, :, :], y[i:end, :, :])
            permuteto!(y, x, p, i + 1)
        end
    end
    return x
end

function propagateperm!(
    dest::AbstractTrackParts{T},
    src::AbstractTrackParts{T},
    p::AbstractVector{<:Integer},
    pos::AbstractVector{<:Integer},
) where {T}
    propagateperm!(dest.value, src.value, p, pos)
    propagateperm!(dest.presence, src.presence, p, pos)
    return dest
end

function update!(tracksₒ::TrackParts{T}, tracksₚ::MHTrackParts{T}, msdᵥ::T) where {T}
    initmh!(tracksₚ)
    p = randperm(size(tracksₒ.value, 3))
    permuteto!(tracksₚ, tracksₒ, p, 2)
    setdisplacement²!(tracksₒ)
    diff²!(tracksₚ.displacement², tracksₚ.value, tracksₒ.value)
    sumΔdisplacement²!(tracksₚ, tracksₒ, msdᵥ)
    @views tracksₚ.logacceptance[2:end] .+= tracksₚ.ΣΔdisplacement²
    setacceptance!(tracksₚ, start = 2)
    pos = findall(convert(Vector{Bool}, tracksₚ.acceptance))
    isempty(pos) || propagateperm!(tracksₒ, tracksₚ, p, pos)
    return tracksₒ
end
