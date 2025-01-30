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
    MeanSquaredDisplacement,
    Brightness,
    Chain,
    Sample,
    DNormal,
    CircularGaussian,
    CircularGaussianLorentzian,
    ConstantAnnealing,
    PolynomialAnnealing,
    SPAD,
    EMCCD

export simulate!, bridge!, runMCMC!, runMCMC

include("base.jl")
include("distributions.jl")
include("gaussianpsf.jl")
include("pixeldetector.jl")

include("spad.jl")
include("emccd.jl")

include("msd.jl")
include("ntracks.jl")
include("brightness.jl")
include("tracks.jl")

include("annealing.jl")
include("permutation.jl")
include("chain.jl")
end
