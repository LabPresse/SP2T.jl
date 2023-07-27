module SpBNPTrack

using Distributions
using SpecialFunctions
using LinearAlgebra
using Flux
using CUDA
using GLMakie
# using StatsBase
# using Octavian
# using LoopVectorization


include("datatypes.jl")
include("psfs.jl")
include("samplers.jl")
# include("priors.jl")
include("forward.jl")
include("plotting.jl")
include("chains.jl")

function forward_main(params, priors; M = 3, D = 0.05, h = 1e1, F = 1e2; T = Float64)
    x = Array{T,3}(undef, 3, M, params.length)
    simulate!(x, priors.x, D, params.period)

    g = Array{T,3}(undef, params.pxnumx, params.pxnumy, params.length)
    simulate!(g, x, params.pxboundsx, params.pxboundsy, params.PSF)

    data = simulate(g, F, h, params)

    return (Video(data, params), Sample(x, D, h, F))
end

end
