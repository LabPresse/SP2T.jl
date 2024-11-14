permuteto!(
    dest::AbstractArray{T,3},
    src::AbstractArray{T,3},
    p::AbstractVector{<:Integer},
    start::Integer = 1,
) where {T} = @views copyto!(dest[start:end, :, :], src[start:end, :, p])

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

function shuffleactive!(tracks::Tracks{T}, ntracksᵥ::Integer) where {T}
    x, y = viewactive(tracks, ntracksᵥ)
    p = randperm(ntracksᵥ)
    isequal(p, 1:ntracksᵥ) || _permute!(x, p, y)
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

function update!(tracks::Tracks{T}, ntracksᵥ::Integer, msdᵥ::T) where {T}
    initacceptance!(tracks)
    x, y, Δx², Δy² = viewactive(tracks, ntracksᵥ)
    p = randperm(ntracksᵥ)
    permuteto!(y, x, p, 2)
    diff²!(Δx², x)
    diff²!(Δy², y, x)
    sum!(tracks.ΣΔdisplacement², Δx² .-= Δy²) ./= 2 * msdᵥ
    @views tracks.logratio[2:end] .+= tracks.ΣΔdisplacement²
    logaccept!(tracks.accepted, tracks.logratio, start = 2)
    pos = findall(convert(Vector{Bool}, tracks.accepted))
    isempty(pos) || propagateperm!(x, y, p, pos)
    return tracks
end
