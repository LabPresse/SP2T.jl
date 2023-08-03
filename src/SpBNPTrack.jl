module SpBNPTrack

using Distributions
using SpecialFunctions
using LinearAlgebra
using Flux
using CUDA
using GLMakie

export ExperimentalParameter
export Video
export Prior
export Sample
export Chain
export visualize_data_3D


# The files in the first "include" block ONLY contains struct definitions, basic constructors, and simple utility functions.
include("psf.jl")
include("data.jl")
include("sample.jl")
include("annealing.jl")
include("chain.jl")

# This file contains the outer constructors (constructors users should call). These constructor methods are placed in a separate file as they take structs as arguments. If these constructors are distributed to the files above, the order of inclusion will be a problem.
include("constructors.jl")

include("samplers.jl")
# include("priors.jl")
include("forward.jl")
include("plotting.jl")

end
