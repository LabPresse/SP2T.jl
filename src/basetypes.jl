abstract type Detector{T} end

# abstract type StreamDetector{T} <: Detector{T} end

abstract type PointSpreadFunction{T} end

abstract type RandomVariable{T} end

# abstract type MSD{T} <: AbstractRandomVariable{T} end

# abstract type AbstractBrightness{T} <: AbstractRandomVariable{T} end