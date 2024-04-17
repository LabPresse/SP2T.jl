struct PriorParameter{T<:AbstractFloat}
    pb::T
    μx::AbstractArray{T}
    σx::AbstractArray{T}
    ϕD::T
    χD::T
    ϕh::T
    ψh::T
    qM::T
    PriorParameter(
        T::DataType;
        pb::Real,
        μx::AbstractArray{<:Real},
        σx::AbstractArray{<:Real},
        ϕD::Real,
        χD::Real,
        ϕh::Real,
        ψh::Real,
        qM::Real,
    ) = new{T}(pb, μx, σx, ϕD, χD, ϕh, ψh, qM)
end

_eltype(p::PriorParameter{T}) where {T} = T

# ChainStatus contains auxiliary variables
mutable struct ChainStatus{T<:AbstractFloat,AT<:AbstractArray{T}}
    tracks::MHTrajectory{AT}
    emittercount::DSIID{Int}
    diffusivity::DSIID{T}
    brightness::MHIID{T}
    𝐔::AbstractArray{T,3}
    𝐔ᵖ::AbstractArray{T,3}
    iteration::Int # iteration
    temperature::T # temperature
    logposterior::T # log posterior
    loglikelihood::T # log likelihood
    ChainStatus(
        x::MHTrajectory{AT},
        M::DSIID{Int},
        D::DSIID{FT},
        h::MHIID{FT},
        𝐔::AbstractArray{FT,3},
        i::Int = 0,
        𝑇::FT = 1.0,
        ln𝒫::FT = NaN,
        lnℒ::FT = NaN,
    ) where {FT<:AbstractFloat,AT<:AbstractArray{FT}} =
        new{FT,AT}(x, M, D, h, 𝐔, copy(𝐔), i, 𝑇, ln𝒫, lnℒ)
end

# get_B(s::ChainStatus) = count(s.b.value)

get_M(s::ChainStatus) = size(s.tracks.value, 2)

_eltype(s::ChainStatus{T}) where {T} = T

view_on_x(s::ChainStatus) = view(s.tracks.value, :, 1:s.emittercount.value, :)

view_off_x(s::ChainStatus) = @view s.tracks.value[:, s.emittercount.value+1:end, :]

viewdiag(M::AbstractMatrix) = view(M, diagind(M))

default_init_pos_prior(param::ExperimentalParameter) = MvNormal(
    [param.pxboundsx[end], param.pxboundsy[end], 0] ./ 2,
    getpxsize(param) .* [2, 2, 0],
)

mutable struct Chain{T<:AbstractFloat}
    status::ChainStatus{T}
    samples::Vector{Sample{T}}
    annealing::Annealing{T}
    stride::Int
    sizelimit::Int
end

chainlength(c::Chain) = length(c.samples)

_eltype(c::Chain{T}) where {T} = T

isfull(c::Chain) = chainlength(c) > c.sizelimit

function get_x(S::AbstractVector{Sample{T}}) where {T}
    M = get_B.(S)
    N = size(S[1].x, 3)
    x = Array{T}(undef, sum(M), N, 3)
    𝒷 = 0
    @views for (s, m) in zip(S, M)
        permutedims!(x[𝒷.+(1:m), :, :], s.x, (2, 3, 1))
        𝒷 += m
    end
    return x
end

get_D(S::AbstractVector{Sample{T}}) where {T} = [s.D for s in S]

get_h(S::AbstractVector{Sample{T}}) where {T} = [s.h for s in S]

"""
    shrink!(chain)

Shrink the chain of samples by only keeping the odd number samples.
"""
function shrink!(c::Chain)
    deleteat!(c.samples, 2:2:lastindex(c.samples))
    c.stride *= 2
end

"""
    extend!(chain::Chain)

Push the chain's current 'status' (a full sample)  to `samples` and check if the updated chain has reached the `sizelimit`. If so, call `shrink!`.
"""
function extend!(c::Chain)
    push!(c.samples, Sample(c.status))
    isfull(c) && shrink!(c)
    return c
end

function to_cpu!(c::Chain)
    s = c.status
    x = MHTrajectory(
        Array(s.tracks.value),
        s.tracks.dynamics,
        s.tracks.prior,
        s.tracks.proposal,
    )
    𝐔 = Array(s.𝐔)
    c.status = ChainStatus(
        x,
        s.emittercount,
        s.diffusivity,
        s.brightness,
        𝐔,
        iszero(s.iteration) ? 1 : s.iteration,
        s.temperature,
        s.logposterior,
        s.loglikelihood,
    )
    return c
end

function to_cpu!(p::ExperimentalParameter)
    p.pxboundsx = Array(p.pxboundsx)
    p.pxboundsy = Array(p.pxboundsy)
    p.darkcounts = Array(p.darkcounts)
    return p
end

function to_cpu!(v::Video)
    to_cpu!(v.param)
    v.frames = Array(v.frames)
    return v
end

function to_gpu!(c::Chain)
    s = c.status
    x = MHTrajectory(
        CuArray(s.tracks.value),
        s.tracks.dynamics,
        s.tracks.prior,
        s.tracks.proposal,
    )
    𝐔 = CuArray(s.𝐔)
    c.status = ChainStatus(
        x,
        s.emittercount,
        s.diffusivity,
        s.brightness,
        𝐔,
        iszero(s.iteration) ? 1 : s.iteration,
        s.temperature,
        s.logposterior,
        s.loglikelihood,
    )
    return c
end

function to_gpu!(v::Video)
    v.param.pxboundsx = CuArray(v.param.pxboundsx)
    v.param.pxboundsy = CuArray(v.param.pxboundsy)
    v.param.darkcounts = CuArray(v.param.darkcounts)
    v.frames = CuArray(v.frames)
    return v
end

function run_MCMC!(
    c::Chain,
    v::Video;
    num_iter::Union{Integer,Nothing} = nothing,
    run_on_gpu::Bool = true,
)
    iter::Int64 = 0
    device::Device = CPU()
    if run_on_gpu && has_cuda_gpu()
        CUDA.allowscalar(false)
        to_gpu!(c)
        to_gpu!(v)
        device = GPU()
    end
    @showprogress 1 "Computing..." for iter = 1:num_iter
        c.status.iteration = iter
        update_x!(c.status, v, device)
        # update_M!(c.status, v, device)
        update_D!(c.status, v.param)
        update_ln𝒫!(c.status, v, device)
        if mod(iter, c.stride) == 0
            extend!(c)
        end
    end
    return c
end
