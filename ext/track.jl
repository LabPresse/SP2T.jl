function SP2T.addΔlogprior₁!(
    ln𝓇::CuVector{T},
    x::CuArray{T,3},
    y::CuArray{T,3},
    prior::DNormal,
) where {T}
    CUDA.@allowscalar ln𝓇[1] += SP2T.Δlogprior₁(x, y, prior)
    return ln𝓇
end