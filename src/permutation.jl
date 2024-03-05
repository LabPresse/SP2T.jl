function get_perms_matrix!(perms::AbstractMatrix{Integer})
    Threads.@threads for n in axes(perms, 2)
        shuffle!(view(perms, :, n))
    end
    return perms
end

function get_perms_matrix(M::Integer, N::Integer)
    perms = Matrix{Int8}(undef, M, N - 1)
    perms .= 1:M
    get_perms_matrix!(perms)
    return perms
end

"""
    local_permute!(x2::AbstractArray{FT,3}, x1::AbstractArray{FT,3}) where {FT<:AbstractFloat}
"""
function local_permute!(
    x2::AbstractArray{FT,3},
    x1::AbstractArray{FT,3},
) where {FT<:AbstractFloat}
    ~, M, N = size(x2)
    perms = get_perms_matrix(M, N)
    x2[:]
    return perms
    # return x
end

function permute_x!(s::ChainStatus, device::Device)
    update_off_x!(s, device)
    update_on_x!(s, device)
    return s
end
