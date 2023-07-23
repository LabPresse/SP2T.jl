"""
    shrink_chain!(chain::Vector{Sample})

Shrink the chain of samples by only keeping the odd number samples.
"""
function shrink_chain!(chain::Vector{Sample})
    deleteat!(chain, 2:2:lastindex(chain))
    return chain
end

"""
    update_chain!(chain::Vector{Sample}, sample::Sample, sizelimit::Integer)

Push `sample`  to `chain` and check if the updated chain has reached the
`sizelimit`. If so, call `shrink_chain!`.
"""
function update_chain!(chain::Vector{Sample}, sample::Sample, sizelimit::Integer)
    push!(chain, sample)
    lastindex(chain) >= sizelimit && shrink_chain!(chain)
    return chain
end

function get_next_sample(old_sample::Sample, data::Video)

    new_sample = 0
    return new_sample
end


