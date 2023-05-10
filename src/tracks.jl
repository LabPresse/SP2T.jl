function init_trajs!(x::AbstractArray{Float64,3}, p::AbstractVector{Float64})
    return randc!(view(x, 1, :), p)
end

"""
    fill_tracks!(x::AbstractArray{Float64,3}, s::AbstractArray{Float64,3}, D::AbstractVector{Float64}, T::Real)

Generate Brownian spatial trajectories `x` with diffusion coefficient `D` at
times `t`. Initial positions `x[:, :, 1]` are required. The three indices of `x`
are dimension, particle, and time, respectively.  This convention is used to
optimize operations on a time point.
"""
function fill_tracks!(
    x::AbstractArray{Float64,3},
    s::AbstractArray{Int8,3},
    D::AbstractVector{Float64},
    T::Real,
)
    (~, M, N) = size(x)
    x[:, :, 2:N] = repeat(sqrt.(2 .* D[view(s, :, :, 2:N)] .* T), 3)
    x[:, :, 2:N] .*= randn(eltype(x), 3, M, N - 1)
    cumsum!(x, x, dims = 3)
    return x
end

function get_tracks_from_prior(
    s::AbstractArray{Int8,3},
    T::Real,
    initial_location_prior::MvNormal,
    D::AbstractVector{Float64},
)
    (~, M, N) = size(s)
    x = Array{Float64,3}(undef, 3, M, N)
    x[:, :, 1] .= rand(initial_location_prior, M)
    fill_tracks!(x, s, D, T)
    return x
end