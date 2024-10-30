abstract type Detector{T} end

function Base.getproperty(detector::Detector, s::Symbol)
    if s == :framecenter
        return framecenter(detector)
    else
        return getfield(detector, s)
    end
end

abstract type PixelDetector{T} <: Detector{T} end

Base.size(detector::PixelDetector) = size(detector.darkcounts)

framecenter(detector::PixelDetector) = mean(detector.pxboundsx), mean(detector.pxboundsy)

abstract type StreamDetector{T} <: Detector{T} end

abstract type PointSpreadFunction{T} end

abstract type SimplifiedDistribution{T} end

abstract type AuxiliaryVariables{T} end

abstract type AbstractRandomVariable{T} end

# abstract type MSD{T} <: AbstractRandomVariable{T} end

# abstract type AbstractBrightness{T} <: AbstractRandomVariable{T} end