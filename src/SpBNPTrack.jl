module SpBNPTrack

#* This code is optizmized for GPU.

using Distributions
using SpecialFunctions
using LogExpFunctions
using LinearAlgebra
using Random

# GPU 
using Flux
using CUDA

# # IO
# using JLD2

# For visualization
using GLMakie
using ColorSchemes

# Progress Meter
using ProgressMeter

export ExperimentalParameter
export PriorParameter
export Video
export Prior
export Sample
export Chain

export simulate_sample
export visualize
export ftypeof

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

include("likelihood.jl")
include("diffusion.jl")
include("trajectory.jl")
include("brightness.jl")

end
