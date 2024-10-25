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
    Data,
    Normal₃,
    CircularGaussianLorentzian,
    ConstantAnnealing,
    PolynomialAnnealing

export simulate!, bridge!, runMCMC!, runMCMC

include("utils.jl")
include("psf.jl")
include("data.jl")
include("annealing.jl")
include("chain.jl")

include("msd.jl")
include("track.jl")
include("nemitters.jl")
include("brightness.jl")
# include("permutation.jl")

include("likelihood.jl")
include("posterior.jl")
include("updater.jl")

end
