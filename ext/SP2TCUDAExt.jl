module SP2TCUDAExt

using SP2T, CUDA, NNlib, LogExpFunctions, SpecialFunctions, Random

include("data.jl")
include("chain.jl")

include("likelihood.jl")
include("track.jl")

end