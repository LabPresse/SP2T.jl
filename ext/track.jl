function SP2T.addΔlogπ₁!(
    ln𝓇::CuArray{T},
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    prior::Normal₃,
) where {T}
    CUDA.@allowscalar ln𝓇[1] += SP2T.Δlogπ₁(x, y, prior)
    return ln𝓇
end

SP2T.copyidxto!(
    x::AbstractArray{T,N},
    y::AbstractArray{T,N},
    i::CuArray{Bool,N},
) where {T,N} = @. x = (i * y) + (~i * x)
