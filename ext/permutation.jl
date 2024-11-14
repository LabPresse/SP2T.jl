function SP2T._permute!(
    x::CuArray{T,3},
    p::AbstractVector{<:Integer},
    y::CuArray{T,3},
) where {T}
    n = size(y, 1) * size(y, 2)
    @views for (i, j) in enumerate(p)
        CUBLAS.swap!(n, x[:, :, j], y[:, :, i])
    end
    return copyto!(x, y)
end