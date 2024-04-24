abstract type AbstractAnnealing{T} end

abstract type AbstractPSF{T} end

abstract type Device end

struct CPU <: Device end

struct GPU <: Device end

abstract type RandomVariable{T} end

abstract type DirectlySampled{T} <: RandomVariable{T} end

abstract type MHSampled{T} <: RandomVariable{T} end

RealNumberOrArray = Union{Real,AbstractArray{<:Real}}

abstract type Dynamics end

struct Brownian <: Dynamics end

struct IID <: Dynamics end

struct Step <: Dynamics end

abstract type SimplifiedDistribution{T} end

GeneralDistribution{T} = Union{SimplifiedDistribution{T},Distribution}