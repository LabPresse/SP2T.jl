module SP2T

#* This code is optizmized for GPU.

using Statistics
using Distributions
using SpecialFunctions: erf
using LogExpFunctions: logexpm1
using LinearAlgebra
using Random

using NNlib: batched_mul!, batched_transpose

using ProgressMeter: @showprogress

export Tracks,
    NEmitters,
    MeanSquaredDisplacement,
    Brightness,
    Chain,
    Sample,
    DNormal,
    CircularGaussianLorentzian,
    ConstantAnnealing,
    PolynomialAnnealing,
    SPAD

export simulate!, bridge!, runMCMC!, runMCMC

include("basetypes.jl")
include("utils.jl")
include("3dpsf.jl")
include("pixeldetector.jl")
# include("data.jl")

include("msd.jl")
include("track.jl")
include("nemitters.jl")
include("brightness.jl")

include("annealing.jl")
include("chain.jl")

# include("likelihood.jl")
include("permutation.jl")
include("posterior.jl")
include("updater.jl")

end
