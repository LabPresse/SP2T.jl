function init_states!(s::AbstractArray{Int8,3}, p::AbstractVector{Float64})
    return randc!(view(s, 1, :), p)
end

"""
    fill_photostates!(s::AbstractArray{Int8,3}, Π::AbstractMatrix{Float64})

Generate discrete state trajectories `s` with transition probability matrix `Π`.
at times `t`. Initial states `s[1, :, 1]` are required. The first index of `s`
should always the one, the second index is particle, and the third index is
time. This convention is used to optimize operations on a time point.
"""
function fill_photostates!(s::AbstractArray{Int8,3}, Π::AbstractMatrix{Float64})
    (~, M, N) = size(s)
    for n = 2:N, m = 1:M
        s[1, m, n] = randc(view(Π, :, s[1, m, n-1]))
    end
    return s
end

function photostates_from_prior(
    particle_num::Int,
    length::Int,
    initial_state_prior::Categorical,
    # ratematrix::AbstractMatrix{<:Real},
)
    photostates = Array{Int8,3}(undef, 1, particle_num, length)
    photostates[1, :, 1] .= rand(initial_state_prior, particle_num)
    # fill_photostates!(photostates, stochastic_mat)
    return photostates
end
