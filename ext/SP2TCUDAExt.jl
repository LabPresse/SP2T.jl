module SP2TCUDAExt

using SP2T, CUDA, NNlib, LogExpFunctions, SpecialFunctions, Random, LinearAlgebra

include("data.jl")
include("chain.jl")

include("nemitters.jl")
include("track.jl")

end