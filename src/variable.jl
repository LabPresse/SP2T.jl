abstract type RandomVariable{T} end

abstract type DirectlySampled{T} <: RandomVariable{T} end

abstract type MHSampled{T} <: RandomVariable{T} end

RealNumberOrArray = Union{Real,AbstractArray{<:Real}}

abstract type Dynamics end

struct Brownian <: Dynamics end

struct Step <: Dynamics end

mutable struct DSIID{T<:RealNumberOrArray} <: DirectlySampled{T}
    value::T
    prior::Distribution
end

_eltype(rv::DSIID{T}) where {T} = T

mutable struct MHIID{T<:RealNumberOrArray} <: MHSampled{T}
    value::T
    proposedvalue::T
    prior::Distribution
    proposal::Distribution
    counter::Matrix{Int}
    batchsize::Int
    MHIID(value::T, prior::Distribution, proposal::Distribution) where {T} =
        new{T}(value, copy(value), prior, proposal, zeros(Int, 2, 2), 1)
end

_eltype(rv::MHIID{T}) where {T} = T

mutable struct DSTrajectory{T<:AbstractArray{<:Real}} <: DirectlySampled{T}
    value::T
    dynamics::Dynamics
    prior::Distribution
    DSTrajectory(value::T, dynamics::Dynamics, prior::Distribution) where {T} =
        new{T}(value, dynamics, prior)
end

_eltype(rv::DSTrajectory{T}) where {T} = T

mutable struct MHTrajectory{T<:AbstractArray{<:Real}} <: MHSampled{T}
    value::T
    proposedvalue::T
    dynamics::Dynamics
    prior::Distribution
    proposal::Distribution
    counter::Matrix{Int}
    batchsize::Int
    MHTrajectory(value::T, dynamics::Dynamics, 𝒫::Distribution, 𝒬::Distribution) where {T} =
        new{T}(value, copy(value), dynamics, 𝒫, 𝒬, zeros(Int, 2, 2), 1)
end

_eltype(rv::MHTrajectory{T}) where {T} = T

Trajectory = Union{DSTrajectory,MHTrajectory}

IID = Union{DSIID,MHIID}