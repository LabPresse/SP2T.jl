function init_trajs!(x::AbstractArray{Float64,3}, p::AbstractVector{Float64})
    return randc!(view(x, 1, :), p)
end

"""
    fill_tracks!(x::AbstractArray{Float64,3}, D::Real, t::AbstractVector{Float64})

Generate Brownian spatial trajectories `x` with diffusion coefficient `D` at
times `t`. Initial positions `x[:, :, 1]` are required. The three indices of `x`
are dimension, particle, and time, respectively.  This convention is used to
optimize operations on a time point.
"""
function fill_tracks!(x::AbstractArray{Float64,3}, D::Real, t::AbstractVector{Float64})
    (~, M, N) = size(x)
    cumsum!(
        view(x, :, :, 2:N),
        sqrt.(2 .* D .* reshape(diff(t), 1, 1, :)) .* randn(eltype(x), 3, M, N - 1),
        dims = 3,
    )
    x[:, :, 2:N] .+= view(x, :, :, 1)
    return x
end

function tracks_from_prior(
    particle_num::Int,
    times::AbstractArray{<:Real},
    initial_location_prior::MvNormal,
    diffusion::Real
)
    tracks = Array{Float64,3}(undef, 3, particle_num, length(times))
    tracks[:, :, 1] .= rand(initial_location_prior, particle_num)
    fill_tracks!(tracks, diffusion, times)
    return tracks
end