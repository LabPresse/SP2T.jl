module SP2TCUDAExt

using SP2T, CUDA
using Random: AbstractRNG

SP2T.devicerng(::AbstractRNG, ::CuArray) = CUDA.default_rng()

include("permutation.jl")
include("track.jl")

end
