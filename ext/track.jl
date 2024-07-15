function SP2T.addΔlogπ₁!(
    ln𝓇::CuVector{T},
    x::CuArray{T,3},
    y::CuArray{T,3},
    prior::Normal₃,
) where {T}
    CUDA.@allowscalar ln𝓇[1] += SP2T.Δlogπ₁(x, y, prior)
    return ln𝓇
end

SP2T.copyidxto!(x::CuArray{T,N}, y::CuArray{T,N}, i::CuVector{Bool}) where {T,N} =
    @. x = (i * y) + (~i * x)
