function SP2T.addΔlogπ₁!(
    ln𝓇::CuVector{T},
    x::CuArray{T,3},
    y::CuArray{T,3},
    prior::DNormal,
) where {T}
    CUDA.@allowscalar ln𝓇[1] += SP2T.Δlogπ₁(x, y, prior)
    return ln𝓇
end

SP2T._copyto!(dest::CuArray{T,N}, src::CuArray{T,N}, i::CuVector{Bool}) where {T,N} =
    @. dest = (i * src) + (~i * dest)

function SP2T.propose!(y::CuArray{T,3}, x::CuArray{T,3}, σ::CuVector{T}) where {T}
    randn!(y)
    y .= y .* transpose(σ) .+ x
end