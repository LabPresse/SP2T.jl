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

export BrownianTracks,
    NEmitters,
    Diffusivity,
    Brightness,
    Chain,
    Sample,
    Data,
    Normal₃,
    CircularGaussianLorentzian

export simulate!, runMCMC!, maxcount, runMCMC
export visualize

export readbin, getframes, getROIindices, viewframes, pxsize

include("data.jl")
include("annealing.jl")
include("chain.jl")

include("likelihood.jl")
include("diffusion.jl")
include("track.jl")
include("nemitters.jl")
include("brightness.jl")
# include("permutation.jl")

include("updater.jl")
include("posterior.jl")

include("import.jl")
include("data_viewer.jl")
include("visualization.jl")

end
