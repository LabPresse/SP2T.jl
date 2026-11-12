module SP2TMetalExt

using SP2T, Metal
using Random: AbstractRNG

# Metal supports Float32 but not Float64, so run with `T = Float32`.

SP2T.devicerng(::AbstractRNG, ::MtlArray) = Metal.default_rng()

function SP2T.addincident!(
    intensity::MtlArray{T,3},
    psfcomponents::NTuple{2,<:AbstractArray{T,3}},
    relativepos::NTuple{2,<:AbstractArray{T,3}},
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    bounds::NTuple{2,<:AbstractVector{T}},
    psf::SP2T.GaussianPSF{T},
) where {T<:AbstractFloat}
    if iszero(brightnessᵥ) || iszero(size(tracksᵥ, 3)) || isempty(intensity)
        return intensity
    end
    px, py = SP2T.integratepsf!(psfcomponents, relativepos, tracksᵥ, bounds, psf)
    scale = brightnessᵥ / psf.A
    # Update each calibrated pixel independently, including when a neighboring
    # masked pixel holds a nonfinite background.
    for m in axes(tracksᵥ, 3)
        x = view(px, :, m:m, :)
        y = PermutedDimsArray(view(py, :, m:m, :), (2, 1, 3))
        intensity .+= scale .* x .* y
    end
    return intensity
end

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
