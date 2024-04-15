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
    prior::Distribution
    proposal::Distribution
    counter::Matrix{Int}
    batchsize::Int
    MHIID(value::T, 𝒫::Distribution, 𝒬::Distribution) where {T} =
        new{T}(value, 𝒫, 𝒬, zeros(Int, 2, 2), 1)
end

_eltype(rv::MHIID{FT}) where {FT} = FT

mutable struct DSTrajectory{AT<:AbstractArray{<:Real}} <: DirectlySampled{AT}
    value::AT
    dynamics::Dynamics
    prior::Distribution
    DSTrajectory(value::T, dynamics::Dynamics, 𝒫::Distribution) where {T} =
        new{T}(value, dynamics, 𝒫)
end

_eltype(rv::DSTrajectory{AT}) where {AT} = AT

mutable struct MHTrajectory{AT<:AbstractArray{<:Real}} <: MHSampled{AT}
    value::AT
    dynamics::Dynamics
    prior::Distribution
    proposal::Distribution
    counter::Matrix{Int}
    batchsize::Int
    MHTrajectory(
        value::AT,
        dynamics::Dynamics,
        𝒫::Distribution,
        𝒬::Distribution,
    ) where {AT} = new{AT}(value, dynamics, 𝒫, 𝒬, zeros(Int, 2, 2), 1)
end

_eltype(rv::MHTrajectory{AT}) where {AT} = AT

Trajectory = Union{DSTrajectory,MHTrajectory}

IID = Union{DSIID,MHIID}