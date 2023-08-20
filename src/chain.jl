struct Prior{FT<:AbstractFloat}
    x::MvNormal{FT}
    D::InverseGamma{FT}
    h::Gamma{FT}
    F::Gamma{FT}
    b::Bernoulli{FT}
    Prior{FT}(;
        μₓ = [0, 0, 0],
        σₓ = [0, 0, 0],
        ϕ_D = 1,
        χ_D = 1,
        ϕ_F = 1,
        ψ_F = 1,
        ϕ_h = 1,
        ψ_h = 1,
        p_b = 0.1,
    ) where {FT} = new{FT}(
        MvNormal(μₓ, Diagonal(σₓ)),
        InverseGamma(ϕ_D, ϕ_D * χ_D), # ϕ_D = α, ϕ_Dχ_D = θ
        Gamma(ϕ_h, ψ_h / ϕ_h), # ϕ_h = α, ϕ_h / ϕ_h = θ
        Gamma(ϕ_F, ψ_F / ϕ_F), # ϕ_F = α, ϕ_F / ϕ_F = θ
        Bernoulli(p_b),
    )
end

mutable struct Acceptances
    x::Int
    Acceptances() = new(0)
end

mutable struct Chain{FT<:AbstractFloat}
    status::FullSample{FT}
    samples::Vector{Sample{FT}}
    annealing::Annealing
    acceptances::Acceptances
    prior::Prior
    stride::Int
    sizelimit::Int
end

chainlength(c::Chain) = length(c.samples)

ftypeof(c::Chain{FT}) where {FT} = FT

isfull(c::Chain) = chainlength(c) > c.sizelimit

get_D(c::Chain) = [s.D for s in c.samples]

get_h(c::Chain) = [s.h for s in c.samples]

"""
    shrink!(chain)

Shrink the chain of samples by only keeping the odd number samples.
"""
shrink!(c::Chain) = deleteat!(c.samples, 2:2:lastindex(c.samples))

"""
    extend!(chain::Chain)

Push the chain's current 'status' (a full sample)  to `samples` and check if the updated chain has reached the `sizelimit`. If so, call `shrink!`.
"""
function extend!(c::Chain)
    push!(c.samples, Sample(c.status))
    isfull(c) && shrink!(c)
    return c
end

# function get_next_sample(old_sample::Sample, data::Video)
#     new_sample = 0
#     return new_sample
# end

# function extend!(status::Chain)
#     status.sample.i += 1
#     return status
# end

function run_MCMC!(c::Chain, v::Video; num_iter = nothing)
    iter::Int64 = 0
    while isnothing(num_iter) || iter < num_iter
        update_𝕩!(c.status, c.prior.x, v.param)
        update_D!(c.status, c.prior.D, v.param)
        extend!(c)
        iter += 1
    end
    return c
end