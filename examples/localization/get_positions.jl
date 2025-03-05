using SP2T
using JLD2

dir = "./examples/localization"

positions = Vector{Matrix{Float64}}()

for i = 1:100
    chain = load(joinpath(dir, "chain_$i.jld2"), "chain")
    push!(positions, dropdims(chain.samples[end].tracks, dims = 1))
end
