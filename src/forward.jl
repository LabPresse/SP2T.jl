# forward functions for spatial tracks
initialize!(x, prior) = rand(prior, length(x))

"""
    fill!(x, D, T)

Generate Brownian spatial trajectories `x` with diffusion coefficient `D` and
period `T`. Initial positions `x[:, :, 1]` are required. The three indices of
`x` are dimension, particle, and time, respectively. This convention is used to
optimize operations on a time point.
"""
function fill!(x, D, T)
    (~, M, N) = size(x)
    x[:, :, 2:N] = sqrt(2 * D * T) .* randn(eltype(x), 3, M, N - 1)
    cumsum!(x, x, dims = 3)
    return x
end

function simulate!(x, prior, D, T)
    initialize!(view(x, :, :, 1), prior)
    fill!(x, D, T)
    return x
end

# forward functions that calculate contributions from emitters
function get_erf(x, xᵖ, σ)
    U = (xᵖ .- x) ./ σ
    return @views erf.(U[1:end-1, :, :], U[2:end, :, :]) ./ 2
end

function simulate!(g, x, xᵖ, yᵖ, PSF::CircularGaussianLorenzian)
    σ_sqrt2 = PSF.σ_ref_sqrt2 .* sqrt.(1 .+ (view(x, 3:3, :, :) ./ PSF.z_ref) .^ 2)
    U = get_erf(view(x, 1:1, :, :), xᵖ, σ_sqrt2)
    V = get_erf(view(x, 2:2, :, :), yᵖ, σ_sqrt2)
    return batched_mul!(g, U, batched_adjoint(V))
end

# sim_frames(u) = rand(size(u)) .< -expm1.(-u)

# g = sim_img(x, params.pxboundsx, params.pxboundsy, params.PSF)

function simulate(g, F, h, params)
    u = @. F * params.pxareatimesexposure + h * g * params.exposure
    return rand(eltype(u), size(u)) .< -expm1.(-u)
end