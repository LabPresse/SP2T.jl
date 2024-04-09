module SP2T

#* This code is optizmized for GPU.

using Distributions
using SpecialFunctions
using LogExpFunctions
using LinearAlgebra
using Random

# GPU 
using Flux: batched_mul, batched_mul!, batched_transpose
using CUDA

# # IO
# using JLD2

#
using StatsBase: counts

# For visualization
using GLMakie
using ColorSchemes

# Progress Meter
using ProgressMeter

# For postprocessing
using Combinatorics: permutations

export ExperimentalParameter
export PriorParameter
export Video
export Prior
export Sample
export Chain

export simulate_sample
export visualize
export ftypeof

export readbin, getframes, getROIindices, viewframes

# The files in the first "include" block ONLY contains struct definitions, basic constructors, and simple utility functions.
include("detection_model.jl")
include("data.jl")
include("sample.jl")
include("annealing.jl")
include("random_variable.jl")
include("chain.jl")

# This file contains the outer constructors (constructors users should call). These constructor methods are placed in a separate file as they take structs as arguments. If these constructors are distributed to the files above, the order of inclusion will be a problem.
include("constructors.jl")

include("samplers.jl")
# include("priors.jl")
include("main_fxns.jl")
include("visualization.jl")

include("likelihood.jl")
include("diffusion.jl")
include("trajectory.jl")
include("number.jl")
include("brightness.jl")
include("permutation.jl")
include("posterior.jl")

include("import.jl")
include("data_viewer.jl")

end
