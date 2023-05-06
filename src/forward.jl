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

# function init_states!(s::AbstractArray{Int8,3}, p::AbstractVector{Float64})
#     return randc!(view(s, 1, :), p)
# end

# """
#     get_photostates!(s::AbstractArray{Int8,3}, Π::AbstractMatrix{Float64})

# Generate discrete state trajectories `s` with transition probability matrix `Π`.
# at times `t`. Initial states `s[1, :, 1]` are required. The first index of `s`
# should always the one, the second index is particle, and the third index is
# time. This convention is used to optimize operations on a time point.
# """
# function get_photostates!(s::AbstractArray{Int8,3}, Π::AbstractMatrix{Float64})
#     (~, M, N) = size(s)
#     for n = 2:N, m = 1:M
#         s[1, m, n] = randc(view(Π, :, s[1, m, n-1]))
#     end
#     return s
# end

# function init_trajs!(x::AbstractArray{Float64,3}, p::AbstractVector{Float64})
#     return randc!(view(x, 1, :), p)
# end

# """
#     get_tracks!(x::AbstractArray{Float64,3}, D::Real, t::AbstractVector{Float64})

# Generate Brownian spatial trajectories `x` with diffusion coefficient `D` at
# times `t`. Initial positions `x[:, :, 1]` are required. The three indices of `x`
# are dimension, particle, and time, respectively.  This convention is used to
# optimize operations on a time point.
# """
# function get_tracks!(x::AbstractArray{Float64,3}, D::Real, t::AbstractVector{Float64})
#     (~, M, N) = size(x)
#     cumsum!(
#         view(x, :, :, 2:N),
#         sqrt.(2 .* D .* reshape(diff(t), 1, 1, :)) .* randn(eltype(x), 3, M, N - 1),
#         dims = 3,
#     )
#     x[:, :, 2:N] .+= view(x, :, :, 1)
#     return x
# end

"""
    gen_pureframe!(g::AbstractArray{Float64,2}, x::AbstractArray{Float64,2}, PSF::AbstractPSF)

Generate a frame that only contains one particle's photon contribution 'g', assuming
it's bright. `x` must contain all this particle's positions within the frame's
exposure.
"""
function gen_pureframe!(
    g::AbstractArray{Float64,2},
    x::AbstractArray{Float64,2},
    PSF::AbstractPSF,
    params::ExperimentalParameters,
)
    K = size(x, 2)
    g .= PSF(view(x, :, 1), params.pixelboundsx, params.pixelboundsy) ./ 2
    @simd for k = 2:K-1
        @inbounds g .+= PSF(view(x, :, k), params.pixelboundsx, params.pixelboundsy)
    end
    g .+= PSF(view(x, :, K), params.pixelboundsx, params.pixelboundsy) ./ 2
    g .*= params.exposure / (K - 1)
    return g
end

"""
    gen_frame!(w::AbstractArray{Float64,2}, u::AbstractArray{Float64,2}, PSF::AbstractPSF)

Generate a frame that only contains one particle's photon contribution, assuming
it's bright. `x` must contain all this particle's positions within the frame's
exposure.
"""
function gen_frame!(
    u::AbstractArray{Float64,3},
    g::AbstractArray{Float64,3},
    b::AbstractArray{Bool,3},
    params::ExperimentalParameters,
)
    K = size(x, 2)
    u .= PSF(view(x, :, 1), params.pixelboundsx, params.pixelboundsy) ./ 2
    @simd for k = 2:K-1
        @inbounds u .+= PSF(view(x, :, k), params.pixelboundsx, params.pixelboundsy)
    end
    u .+= PSF(view(x, :, K), params.pixelboundsx, params.pixelboundsy) ./ 2
    u .*= params.exposure / (K - 1)
    return u
end


function gen_imgs!(
    w::AbstractArray{Float64,3},
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