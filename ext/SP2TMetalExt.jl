module SP2TMetalExt

using SP2T, Metal
using Random: AbstractRNG

# Metal supports Float32 but not Float64, so run with `T = Float32`.

SP2T.devicerng(::AbstractRNG, ::MtlArray) = Metal.default_rng()

function SP2T.addΔlogprior₁!(
    ln𝓇::MtlVector{T},
    x::MtlArray{T,3},
    y::MtlArray{T,3},
    prior::DNormal,
) where {T}
    Metal.@allowscalar ln𝓇[1] += SP2T.Δlogprior₁(x, y, prior)
    return ln𝓇
end

end
