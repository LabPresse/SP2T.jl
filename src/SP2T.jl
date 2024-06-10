module SP2T

#* This code is optizmized for GPU.

using Distributions
using SpecialFunctions
using LogExpFunctions
using LinearAlgebra
using Random

# GPU 
using NNlib: batched_mul, batched_mul!, batched_transpose

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

# export 
# export PriorParameter
# export Video
# export Prior
# export Sample
# export Chain

export BrownianTracks,
    NEmitters,
    Diffusivity,
    Brightness,
    Chain,
    Sample,
    Data,
    Normal₃,
    CircularGaussianLorentzian

export simulate, runMCMC!, maxcount, runMCMC
export visualize

export readbin, getframes, getROIindices, viewframes, pxsize

# include("type.jl")

# The files in the first "include" block ONLY contains struct definitions, basic constructors, and simple utility functions.

include("data.jl")
# include("detection_model.jl")
# include("sample.jl")
include("annealing.jl")
# include("variable.jl")
include("chain.jl")

# This file contains the outer constructors (constructors users should call). These constructor methods are placed in a separate file as they take structs as arguments. If these constructors are distributed to the files above, the order of inclusion will be a problem.
# include("constructors.jl")

# include("samplers.jl")
include("visualization.jl")

include("likelihood.jl")
include("diffusion.jl")
include("track.jl")
include("nemitters.jl")
include("brightness.jl")
# include("permutation.jl")

# include("utility.jl")
include("simulation.jl")
include("updater.jl")
include("posterior.jl")

include("import.jl")
include("data_viewer.jl")

end
