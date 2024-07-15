_randperm!(i::AbstractVector{<:Integer}, M::Integer) = copyto!(i, randperm(M))

function shuffletracks!(
    dest::AbstractArray{T,3},
    src::AbstractArray{T,3},
    perm::AbstractVector{<:Integer},
) where {T}
    @views copyto!(dest[:, 1:length(perm), :], src[:, perm, :])
    return dest
end