function SP2T._permute!(
    x::CuArray{T,3},
    p::AbstractVector{<:Integer},
    y::CuArray{T,3},
) where {T}
    N = size(x, 1) * size(x, 2)
    @views for (i, j) in enumerate(p)
        CUBLAS.swap!(N, y[:, :, i], x[:, :, j])
    end
    copyto!(x, y)
end