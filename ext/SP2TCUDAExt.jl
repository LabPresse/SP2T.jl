module SP2TCUDAExt

using SP2T, CUDA, NNlib, LogExpFunctions, SpecialFunctions, Random

# include("type.jl")

# The files in the first "include" block ONLY contains struct definitions, basic constructors, and simple utility functions.

# include("data.jl")
include("detection_model.jl")
# include("sample.jl")
# include("annealing.jl")
# include("variable.jl")
include("chain.jl")

# This file contains the outer constructors (constructors users should call). These constructor methods are placed in a separate file as they take structs as arguments. If these constructors are distributed to the files above, the order of inclusion will be a problem.
# include("constructors.jl")

# include("samplers.jl")
# include("visualization.jl")

include("likelihood.jl")
# include("diffusion.jl")
include("track.jl")
# include("nemitters.jl")
# include("brightness.jl")
# include("permutation.jl")
# include("posterior.jl")

# include("import.jl")
# include("data_viewer.jl")

end