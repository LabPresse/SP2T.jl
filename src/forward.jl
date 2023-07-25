# include("samplers.jl")
# include("DataTypes.jl")

# struct PSF
#     type::String
#     z_ref::Float64 # [length] std of PSF along z  (optical axis)
#     v_ref::Float64 # [length] std of PSF along xy (image plane)
# end

# function (PSF::CircularGaussianLorenzian)(x, y, z)
#     v_z = PSF.v_ref * (1 + (z / PSF.z_ref)^2)
#     return 1 / (2 * pi * v_z) * exp(-((x - xp)^2 + (y - yp)^2) / (2 * v_z))
# end

# function init_states!(s::AbstractArray{<:Integer,3}, p::AbstractVector{<:AbstractFloat})
#     return randc!(view(s, 1, :), p)
# end

# """
#     get_photostates!(s::AbstractArray{<:Integer,3}, Π::AbstractMatrix{<:AbstractFloat})

# Generate discrete state trajectories `s` with transition probability matrix `Π`.
# at times `t`. Initial states `s[1, :, 1]` are required. The first index of `s`
# should always the one, the second index is particle, and the third index is
# time. This convention is used to optimize operations on a time point.
# """
# function get_photostates!(s::AbstractArray{<:Integer,3}, Π::AbstractMatrix{<:AbstractFloat})
#     (~, M, N) = size(s)
#     for n = 2:N, m = 1:M
#         s[1, m, n] = randc(view(Π, :, s[1, m, n-1]))
#     end
#     return s
# end

# function init_trajs!(x::AbstractArray{<:AbstractFloat,3}, p::AbstractVector{<:AbstractFloat})
#     return randc!(view(x, 1, :), p)
# end

# """
#     get_tracks!(x::AbstractArray{<:AbstractFloat,3}, D::Real, t::AbstractVector{<:AbstractFloat})

# Generate Brownian spatial trajectories `x` with diffusion coefficient `D` at
# times `t`. Initial positions `x[:, :, 1]` are required. The three indices of `x`
# are dimension, particle, and time, respectively.  This convention is used to
# optimize operations on a time point.
# """
# function get_tracks!(x::AbstractArray{<:AbstractFloat,3}, D::Real, t::AbstractVector{<:AbstractFloat})
#     (~, M, N) = size(x)
#     cumsum!(
#         view(x, :, :, 2:N),
#         sqrt.(2 .* D .* reshape(diff(t), 1, 1, :)) .* randn(eltype(x), 3, M, N - 1),
#         dims = 3,
#     )
#     x[:, :, 2:N] .+= view(x, :, :, 1)
#     return x
# end


# forward functions for state trajectories
init_states!(s, p) = randc!(view(s, 1, :), p)

"""
    fill_photostates!(s, Π)

Generate discrete state trajectories `s` with transition probability matrix `Π`.
at times `t`. Initial states `s[1, :, 1]` are required. The first index of `s`
should always the one, the second index is particle, and the third index is
time. This convention is used to optimize operations on a time point.
"""
function fill_states!(s, Π)
    (~, M, N) = size(s)
    for n = 2:N, m = 1:M
        s[1, m, n] = randc(view(Π, :, s[1, m, n-1]))
    end
    return s
end

function sim_states(
    M,
    N,
    prior::Categorical,
    # ratematrix::AbstractMatrix{<:Real},
)
    s = Array{Integer,3}(undef, 1, M, N)
    fill!(s, 1)
    # s[1, :, 1] .= rand(prior, M)
    # fill_states!(s, stochastic_mat)
    return s
end

# forward functions for spatial tracks
function init_tracks(s, prior)
    (~, M, N) = size(s)
    x = Array{Float64,3}(undef, 3, M, N)
    x[:, :, 1] .= rand(prior, M)
    return x
end

"""
    fill_tracks!(x, s, D, T)

Generate Brownian spatial trajectories `x` with diffusion coefficient `D` at
times `t`. Initial positions `x[:, :, 1]` are required. The three indices of `x`
are dimension, particle, and time, respectively.  This convention is used to
optimize operations on a time point.
"""
function fill_tracks!(x, s, D, T)
    (~, M, N) = size(x)
    x[:, :, 2:N] = repeat(sqrt.(2 .* D[view(s, :, :, 2:N)] .* T), 3)
    x[:, :, 2:N] .*= randn(eltype(x), 3, M, N - 1)
    cumsum!(x, x, dims = 3)
    return x
end

function sim_tracks(s, prior, D, T)
    x = init_tracks(s, prior)
    fill_tracks!(x, s, D, T)
    return x
end

# forward functions that calculate contributions from emitters
function get_erf(x, xᵖ, σ)
    U = (xᵖ .- x) ./ σ
    return @views erf.(U[1:end-1, :, :], U[2:end, :, :]) ./ 2
end

function sim_img(x, xᵖ, yᵖ, PSF::CircularGaussianLorenzian)
    σ_sqrt2 = PSF.σ_ref_sqrt2 .* sqrt.(1 .+ (view(x, 3:3, :, :) ./ PSF.z_ref) .^ 2)
    U = get_erf(view(x, 1:1, :, :), xᵖ, σ_sqrt2)
    V = get_erf(view(x, 2:2, :, :), yᵖ, σ_sqrt2)
    return batched_mul(U, batched_adjoint(V))
end

function sim_img!(g, x, xᵖ, yᵖ, PSF::CircularGaussianLorenzian)
    σ_sqrt2 = PSF.σ_ref_sqrt2 .* sqrt.(1 .+ (view(x, 3:3, :, :) ./ PSF.z_ref) .^ 2)
    U = get_erf(view(x, 1:1, :, :), xᵖ, σ_sqrt2)
    V = get_erf(view(x, 2:2, :, :), yᵖ, σ_sqrt2)
    return batched_mul!(g, U, batched_adjoint(V))
end

# sim_frames(u) = rand(size(u)) .< -expm1.(-u)

# g = sim_img(x, params.pxboundsx, params.pxboundsy, params.PSF)

function sim_frames(g, F, h, params)
    u = @. F * params.pxareatimesexposure + h * g * params.exposure
    return rand(eltype(u), size(u)) .< -expm1.(-u)
end

"""
    gen_pureframe!(g::AbstractArray{AbstractFloat,2}, x::AbstractArray{AbstractFloat,2}, PSF::AbstractPSF)

Generate a frame that only contains one particle's photon contribution 'g', assuming
it's bright. `x` must contain all this particle's positions within the frame's
exposure.
"""
function gen_pureframe!(
    g::AbstractArray{AbstractFloat,2},
    x::AbstractArray{AbstractFloat,2},
    PSF::AbstractPSF,
    params::ExperimentalParameters,
)
    K = size(x, 2)
    g .= PSF(view(x, :, 1), params.pxboundsx, params.pxboundsy) ./ 2
    @simd for k = 2:K-1
        @inbounds g .+= PSF(view(x, :, k), params.pxboundsx, params.pxboundsy)
    end
    g .+= PSF(view(x, :, K), params.pxboundsx, params.pxboundsy) ./ 2
    g .*= params.exposure / (K - 1)
    return g
end

"""
    gen_frame!(w::AbstractArray{AbstractFloat,2}, u::AbstractArray{AbstractFloat,2}, PSF::AbstractPSF)

Generate a frame that only contains one particle's photon contribution, assuming
it's bright. `x` must contain all this particle's positions within the frame's
exposure.
"""
function gen_frame!(
    u::AbstractArray{<:AbstractFloat,3},
    g::AbstractArray{<:AbstractFloat,3},
    b::AbstractArray{Bool,3},
    params::ExperimentalParameters,
)
    K = size(x, 2)
    u .= PSF(view(x, :, 1), params.pxboundsx, params.pxboundsy) ./ 2
    @simd for k = 2:K-1
        @inbounds u .+= PSF(view(x, :, k), params.pxboundsx, params.pxboundsy)
    end
    u .+= PSF(view(x, :, K), params.pxboundsx, params.pxboundsy) ./ 2
    u .*= params.exposure / (K - 1)
    return u
end


function gen_imgs!(
    w::AbstractArray{<:AbstractFloat,3},
    params::ExperimentalParameters,
    G::Float64 = 1.0,
)
    (N, ~, M) = size(x)
    x[2:N, :, :] =
        cumsum(sqrt.(@. 2 * D * diff(t)) .* randn(eltype(x), N - 1, 3, M), dims = 1)
end

# tn_bnd = dt_stp * (0:N);
# tn_min = tn_bnd(1:N) + 0.5 * (dt_stp - dt_exp);
# tn_max = tn_bnd(2:N+1) - 0.5 * (dt_stp - dt_exp);

function forward_init()

end

"""
    get_times!(T::Float64, τ::Float64, K::Int, N::Int)

Generate a vector of time points such that trajectories can be obtained
accordingly. `T` is the frame period, `τ` is the exposure time, `K` is the
number of time points within each exposure, and `N` is the total number of
frames.
"""
function get_times(T::Float64, τ::Float64, K::Int, N::Int)
    return τ / (K - 1) .* repeat(0:K-1, N) .+ (T - τ) .* repeat(0.5:N, inner = K) .+
           τ .* repeat(0:N-1, inner = K)
end

"""
    get_weights!(T::Float64, τ::Float64, K::Int, N::Int)

Generate all weights needed in the trapezoidal integration. `K` is the number of
weights.
"""
function get_weights(K::Int)
    v = ones(K)
    v[1], v[end] = 0.5, 0.5
    return v
end