abstract type RandomVariable{T} end

mutable struct DirectlySampledScalarRV{FT<:AbstractFloat} <: RandomVariable{FT}
    value::FT
    𝒫::Distribution
end

ftypeof(rv::DirectlySampledScalarRV{FT}) where {FT} = FT

mutable struct DirectlySampledVectorRV{AT<:AbstractArray} <: RandomVariable{AT}
    value::AT
    𝒫::Distribution
end

ftypeof(rv::DirectlySampledVectorRV{AT}) where {AT} = AT

mutable struct MetropolisHastingsScalarRV{FT<:AbstractFloat} <: RandomVariable{FT}
    value::FT
    𝒫::Distribution
    𝒬::Distribution
    counter::Matrix{Int}
    batchsize::Int
    MetropolisHastingsScalarRV(value::FT, 𝒫::Distribution, 𝒬::Distribution) where {FT} =
        new{FT}(value, 𝒫, 𝒬, zeros(Int, 2, 2), 1)
end

ftypeof(rv::MetropolisHastingsScalarRV{FT}) where {FT} = FT

mutable struct MetropolisHastingsVectorRV{AT<:AbstractArray} <: RandomVariable{AT}
    value::AT
    𝒫::Distribution
    𝒬::Distribution
    counter::Matrix{Int}
    batchsize::Int
    MetropolisHastingsVectorRV(value::FT, 𝒫::Distribution, 𝒬::Distribution) where {FT} =
        new{FT}(value, 𝒫, 𝒬, zeros(Int, 2, 2), 1)
end

ftypeof(rv::MetropolisHastingsVectorRV{AT}) where {AT} = AT

struct PriorParameter{FT<:AbstractFloat}
    pb::FT
    μx::AbstractArray{FT}
    σx::AbstractArray{FT}
    ϕD::FT
    χD::FT
    ϕh::FT
    ψh::FT
    PriorParameter(
        FT::DataType;
        pb::Real,
        μx::AbstractArray{<:Real},
        σx::AbstractArray{<:Real},
        ϕD::Real,
        χD::Real,
        ϕh::Real,
        ψh::Real,
    ) = new{FT}(pb, μx, σx, ϕD, χD, ϕh, ψh)
end

ftypeof(p::PriorParameter{FT}) where {FT} = FT

# ChainStatus contains auxiliary variables
mutable struct ChainStatus{FT<:AbstractFloat,AT<:AbstractArray{FT}}
    b::DirectlySampledVectorRV
    x::MetropolisHastingsVectorRV{AT}
    D::DirectlySampledScalarRV{FT}
    h::MetropolisHastingsScalarRV{FT}
    G::AbstractArray{FT,3}
    i::Int # iteration
    𝑇::FT # temperature
    ln𝒫::FT # log posterior
    ChainStatus(
        b::DirectlySampledVectorRV{<:AbstractVector{Bool}},
        x::MetropolisHastingsVectorRV{AT},
        D::DirectlySampledScalarRV{FT},
        h::MetropolisHastingsScalarRV{FT},
        G::AbstractArray{FT,3},
        i::Int = 0,
        𝑇::FT = 1.0,
        ln𝒫::FT = NaN,
    ) where {FT<:AbstractFloat,AT<:AbstractArray{FT}} = new{FT,AT}(b, x, D, h, G, i, 𝑇, ln𝒫)
end

get_B(s::ChainStatus) = count(s.b.value)

get_M(s::ChainStatus) = size(s.x.value, 2)

ftypeof(s::ChainStatus{FT}) where {FT} = FT

view_on_x(s::ChainStatus) = @view s.x.value[:, 1:get_B(s), :]

view_off_x(s::ChainStatus) = @view s.x.value[:, get_B(s)+1:end, :]

viewdiag(M::AbstractMatrix) = view(M, diagind(M))

function default_init_pos_prior(param::ExperimentalParameter)
    Nx, Ny, a = param.pxnumx, param.pxnumy, param.pxsize
    μₓ = [Nx * a / 2, Ny * a / 2, 0]
    σₓ = [a * 2, a * 2, 0]
    return MvNormal(μₓ, σₓ)
end

# struct Prior{FT<:AbstractFloat}
#     x::MvNormal{FT}
#     D::InverseGamma{FT}
#     h::Gamma{FT}
#     b::Bernoulli{FT}
#     Prior{FT}(;
#         μₓ = [0, 0, 0],
#         σₓ = [0, 0, 0],
#         ϕᴰ = 1,
#         χᴰ = 1,
#         ϕₕ = 1,
#         ψₕ = 1,
#         pᵇ = 0.1,
#     ) where {FT} = new{FT}(
#         MvNormal(μₓ, Diagonal(σₓ)),
#         InverseGamma(ϕᴰ, ϕᴰ * χᴰ), # ϕᴰ = α, ϕᴰχᴰ = θ
#         Gamma(ϕₕ, ψₕ / ϕₕ), # ϕₕ = α, ϕₕ / ϕₕ = θ
#         Bernoulli(pᵇ),
#     )
# end

# 𝐱

# Prior(
#     FloatType;
#     μₓ = [0, 0, 0],
#     σₓ = [0, 0, 0],
#     ϕᴰ = 1,
#     χᴰ = 1,
#     ϕ_F = 1,
#     ψ_F = 1,
#     ϕₕ = 1,
#     ψₕ = 1,
#     pᵇ = 0.1,
# ) = new{FT}(
#     MvNormal(μₓ, Diagonal(σₓ)),
#     InverseGamma(ϕᴰ, ϕᴰ * χᴰ), # ϕᴰ = α, ϕᴰχᴰ = θ
#     Gamma(ϕₕ, ψₕ / ϕₕ), # ϕₕ = α, ϕₕ / ϕₕ = θ
#     Bernoulli(pᵇ),
# )

# mutable struct MetropolisHastings
#     𝒬::Distribution
#     accep_count::Matrix{Int}
# stepsize::Real
#     stepsize::Float64
#     MetropolisHastings(𝒬::Distribution) = new(𝒬, zeros(Int, 2, 2), 1)
# end

# mutable struct Acceptance
#     x::Int
#     Acceptance() = new(0)
# end

# mutable struct Proposal
#     accep_count::Vector{Int}
#     distritbution::Distribution
#     Proposal(ℚ::Distribution) = new([0, 0], ℚ)
# end

# struct Proposals
#     h::Proposal
# end

mutable struct Chain{FT<:AbstractFloat}
    status::ChainStatus{FT}
    samples::Vector{Sample{FT}}
    annealing::Annealing{FT}
    stride::Int
    sizelimit::Int
end

chainlength(c::Chain) = length(c.samples)

ftypeof(c::Chain{FT}) where {FT} = FT

isfull(c::Chain) = chainlength(c) > c.sizelimit

# function get_x(s::Sample{FT}) where {FT}
#     ℬ = get_B(s)
#     N = size(s.x, 3)
#     x = Matrix{FT}(undef, ℬ, N)
#     y = Matrix{FT}(undef, ℬ, N)
#     z = Matrix{FT}(undef, ℬ, N)
#     𝒷 = 1
#     for s in S, m in get_B(s)
#         x[𝒷, :] = s.x[1, m, :]
#         y[𝒷, :] = s.x[2, m, :]
#         z[𝒷, :] = s.x[3, m, :]
#         𝒷 += 1
#     end
#     return x, y, z
# end

function get_x(S::AbstractVector{Sample{FT}}) where {FT}
    ℬ = sum(get_B.(S))
    N = size(S[1].x, 3)
    x = Matrix{FT}(undef, ℬ, N)
    y = Matrix{FT}(undef, ℬ, N)
    z = Matrix{FT}(undef, ℬ, N)
    𝒷 = 1
    for s in S, m in get_B(s)
        x[𝒷, :] = s.x[1, m, :]
        y[𝒷, :] = s.x[2, m, :]
        z[𝒷, :] = s.x[3, m, :]
        𝒷 += 1
    end
    return x, y, z
end

get_D(S::AbstractVector{Sample{FT}}) where {FT} = [s.D for s in S]

get_h(S::AbstractVector{Sample{FT}}) where {FT} = [s.h for s in S]

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

# function get_next_sample(old_sample::Sample, data::Video)
#     new_sample = 0
#     return new_sample
# end

# function extend!(status::Chain)
#     status.sample.i += 1
#     return status
# end

function to_cpu!(c::Chain)
    s = c.status
    b = DirectlySampledVectorRV(Array(s.b.value), s.b.𝒫)
    x = MetropolisHastingsVectorRV(Array(s.x.value), s.x.𝒫, s.x.𝒬)
    G = Array(s.G)
    c.status = ChainStatus(b, x, s.D, s.h, G, iszero(s.i) ? 1 : s.i, s.𝑇, s.ln𝒫)
    return c
end

function to_cpu!(v::Video)
    v.param.pxboundsx = Array(v.param.pxboundsx)
    v.param.pxboundsy = Array(v.param.pxboundsy)
    v.param.darkcounts = Array(v.param.darkcounts)
    v.data = Array(v.data)
    return v
end

function to_gpu!(c::Chain)
    s = c.status
    b = DirectlySampledVectorRV(CuArray(s.b.value), s.b.𝒫)
    x = MetropolisHastingsVectorRV(CuArray(s.x.value), s.x.𝒫, s.x.𝒬)
    G = CuArray(s.G)
    c.status = ChainStatus(b, x, s.D, s.h, G, iszero(s.i) ? 1 : s.i, s.𝑇, s.ln𝒫)
    return c
end

function to_gpu!(v::Video)
    v.param.pxboundsx = CuArray(v.param.pxboundsx)
    v.param.pxboundsy = CuArray(v.param.pxboundsy)
    v.param.darkcounts = CuArray(v.param.darkcounts)
    v.data = CuArray(v.data)
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
    # while isnothing(num_iter) || iter < num_iter
    #     iter += 1
    #     # update_off_x!(c.status, c.prior.x, v.param)
    #     # update_D!(c.status, v.param)
    #     update_on_x!(c.status, v.data, v.param)
    #     c.status.i += 1
    #     extend!(c)
    # end

    @showprogress 1 "Computing..." for iter = 1:num_iter
        c.status.i = iter
        # update_D!(c.status, v.param)
        update_x!(c.status, v.data, v.param, device)
        if mod(iter, c.stride) == 0
            extend!(c)
        end
    end
    return c
end
