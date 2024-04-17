abstract type RandomVariable{T} end

abstract type DirectlySampled{T} <: RandomVariable{T} end

abstract type MHSampled{T} <: RandomVariable{T} end

RealNumberOrArray = Union{Real,AbstractArray{<:Real}}

abstract type Dynamics end

struct Brownian <: Dynamics end

struct Step <: Dynamics end

abstract type DistributionParameter{T} end

DistrOrParam = Union{DistributionParameter,Distribution}

struct Normalₙ{T} <: DistributionParameter{T}
    μ::Vector{T}
    σ::Vector{T}
end

mutable struct DSIID{T<:RealNumberOrArray} <: DirectlySampled{T}
    value::T
    prior::DistrOrParam
end

_eltype(rv::DSIID{T}) where {T} = T

mutable struct MHIID{T<:RealNumberOrArray} <: MHSampled{T}
    value::T
    candidate::T
    prior::DistrOrParam
    proposal::DistrOrParam
    counter::Matrix{Int}
    MHIID(value::T, prior::DistrOrParam, proposal::DistrOrParam) where {T} =
        new{T}(value, copy(value), prior, proposal, zeros(Int, 2, 2))
end

_eltype(rv::MHIID{T}) where {T} = T

mutable struct DSTrajectory{T<:AbstractArray{<:Real}} <: DirectlySampled{T}
    value::T
    dynamics::Dynamics
    prior::DistrOrParam
    DSTrajectory(value::T, dynamics::Dynamics, prior::DistrOrParam) where {T} =
        new{T}(value, dynamics, prior)
end

_eltype(rv::DSTrajectory{T}) where {T} = T

mutable struct MHTrajectory{T<:AbstractArray{<:Real}} <: MHSampled{T}
    value::T
    candidate::T
    dynamics::Dynamics
    prior::DistrOrParam
    proposal::DistrOrParam
    counter::Matrix{Int}
    MHTrajectory(
        value::T,
        dynamics::Dynamics,
        prior::DistrOrParam,
        proposal::DistrOrParam,
    ) where {T} = new{T}(value, copy(value), dynamics, prior, proposal, zeros(Int, 2, 2))
end

_eltype(rv::MHTrajectory{T}) where {T} = T

Trajectory = Union{DSTrajectory,MHTrajectory}

IID = Union{DSIID,MHIID}