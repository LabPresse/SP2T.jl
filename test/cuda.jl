# Optional CUDA regression entry point. Run in an environment with the declared
# CUDA weak dependency and test dependencies installed; ordinary CPU CI is unchanged.
using SP2T, Test, Distributions, StableRNGs, Random, CUDA
CUDA.functional() || error("CUDA regression tests require a functional CUDA device")
CUDA.allowscalar(false)
include("detector_regressions.jl")
detector_regression_suite(CuArray)
include("spad_saturation_regressions.jl")
spad_saturation_regression_suite(CuArray)
CUDA.synchronize()
