
function _permute!(
    x::AbstractArray{T,3},
    p::AbstractVector{<:Integer},
    y::AbstractArray{T,3},
) where {T}
    @views copyto!(y, x[:, :, p])
    copyto!(x, y)
end

function shuffleactive!(
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    nemittersᵥ::Integer,
) where {T}
    p = randperm(nemittersᵥ)
    isequal(p, 1:nemittersᵥ) && return x
    @views _permute!(x[:, :, 1:nemittersᵥ], p, y[:, :, 1:nemittersᵥ])
    return x
end

shuffleactive!(tracks::Tracks{T}, nemitters::NEmitters{T}) where {T} =
    shuffleactive!(tracks.value, tracks.proposal, nemitters.value)

# _randperm!(i::AbstractVector{<:Integer}, M::Integer) = copyto!(i, randperm(M))

# function shuffletracks!(
#     dest::AbstractArray{T,3},
#     src::AbstractArray{T,3},
#     perm::AbstractVector{<:Integer},
# ) where {T}
#     @views copyto!(dest[:, :, 1:length(perm)], src[:, :, perm])
#     return dest
# end

# function shuffleupdater!(
#     x::AbstractArray{T,3},
#     y::AbstractArray{T,3},
#     Δx²::AbstractArray{T,3},
#     Δy²::AbstractArray{T,3},
#     ΔΔx²::AbstractArray{T,3},
#     ΣΔΔx²::AbstractVector{T},
#     D::T,
# ) where {T}
#     perm = randperm(size(x, 3))
#     @views copyto!(y, x[:, :, perm])
#     diff²!(Δx², x)
#     diff²!(Δy², x, y)
#     ΣΔΔx²!(ΣΔΔx², ΔΔx², Δx², Δy², D)
# end
