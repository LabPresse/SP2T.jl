"""
    shrink!(chain)

Shrink the chain of samples by only keeping the odd number samples.
"""
function shrink!(chain)
    deleteat!(chain.samples, 2:2:lastindex(chain.samples))
    chain.size = ceil(Int, chain.size / 2)
    return chain
end

function initialize!(chain::Chain, M::Integer, priors::Priors)
    chain.status = 
    return chain
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

isfull(status::Chain) = status.size >= status.sizelimit

function update!(status::Chain)
    status.sample.i += 1
    return status
end

function run_MCMC(chain::Vector{Sample}, status::Chain)
    update!(status)
    isfull(status) && shrink!(chain, status)
    return (chain, status)
end