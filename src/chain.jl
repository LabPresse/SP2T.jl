struct Prior
    x::MvNormal
    D::InverseGamma
    h::Gamma
    F::Gamma
    b::Bernoulli
    Prior(;
        μₓ = [0, 0, 0],
        σₓ = [0, 0, 0],
        ϕ_D = 1,
        χ_D = 1,
        ϕ_F = 1,
        ψ_F = 1,
        ϕ_h = 1,
        ψ_h = 1,
        p_b = 0.1,
    ) = new(
        MvNormal(μₓ, Diagonal(σₓ)),
        InverseGamma(ϕ_D, ϕ_D * χ_D),
        Gamma(ϕ_h, ψ_h / ϕ_h),
        Gamma(ϕ_F, ψ_F / ϕ_F),
        Bernoulli(p_b),
    )
end

Prior(params::ExperimentalParameter) = Prior(
    μₓ = [params.pxnumx * params.pxsize / 2, params.pxnumy * params.pxsize / 2, 0],
    σₓ = [params.pxsize * 2, params.pxsize * 2, 0],
)

mutable struct Acceptances
    x::Int
    Acceptances() = new(0)
end

mutable struct Chain{FT<:AbstractFloat}
    status::FullSample{FT}
    samples::Vector{Sample{FT}}
    acceptances::Acceptances
    prior::Prior
    stride::Int
    sizelimit::Int
    Chain(FT) = new{FT}()
    Chain(
        status::FullSample{FT},
        samples::Vector{Sample{FT}},
        acceptances::Acceptances,
        prior::Prior,
        stride::Int,
        sizelimit::Int,
    ) where {FT<:AbstractFloat} =
        new{FT}(status, samples, acceptances, prior, stride, sizelimit)
end

Chain(;
    initial_guesss::Sample,
    max_emitter_num::Integer,
    prior::Prior,
    sizelimit::Integer,
) = Chain(
    FullSample(initial_guesss, max_emitter_num),
    [initial_guesss],
    Acceptances(),
    prior,
    1,
    sizelimit,
)

chainlength(c::Chain) = length(c.samples)

get_type(c::Chain{FT}) where {FT} = FT

isfull(c::Chain) = chainlength(c) >= c.sizelimit

"""
    shrink!(chain)

Shrink the chain of samples by only keeping the odd number samples.
"""
shrink!(c::Chain) = deleteat!(c.samples, 2:2:lastindex(c.samples))

function initialize!(
    c::Chain;
    initial_guess::Sample,
    max_emitter_num::Integer,
    prior::Prior,
    annealing::Annealing,
)
    M = max_emitter_num
    c.status, c.prior = FullSample(initial_guess, M)
    c.prior = prior
    return c
end

# function initialize!(chain::Chain, initial_sample::Sample)
#     if isnothing
#     return chain
# end

# function initialize(sample::Sample, priors::Priors)
#     chain = Chain
#     return chain
# end

#! change!
"""
    update!(chain::Vector{Sample}, sample::Sample, sizelimit::Integer)

Push `sample`  to `chain` and check if the updated chain has reached the
`sizelimit`. If so, call `shrink!`.
"""
function update!(chain::Vector{Sample}, sample::Sample, sizelimit::Integer)
    push!(chain, sample)
    lastindex(chain) >= sizelimit && shrink!(chain)
    return chain
end

# function get_next_sample(old_sample::Sample, data::Video)
#     new_sample = 0
#     return new_sample
# end

function update!(status::Chain)
    status.sample.i += 1
    return status
end

function run_MCMC(chain::Vector{Sample}, status::Chain)
    update!(status)
    isfull(status) && shrink!(chain, status)
    return (chain, status)
end