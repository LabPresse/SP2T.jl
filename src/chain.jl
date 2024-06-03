# struct PriorParameter{T<:AbstractFloat}
#     pb::T
#     μx::Vector{T}
#     σx::Vector{T}
#     ϕD::T
#     χD::T
#     ϕh::T
#     ψh::T
#     qM::T
#     PriorParameter(
#         T::DataType;
#         pb::Real,
#         μx::Vector{<:Real},
#         σx::Vector{<:Real},
#         ϕD::Real,
#         χD::Real,
#         ϕh::Real,
#         ψh::Real,
#         qM::Real,
#     ) = new{T}(pb, μx, σx, ϕD, χD, ϕh, ψh, qM)
# end

# _eltype(::PriorParameter{T}) where {T} = T
struct Sample{Ts,Ta}
    tracks::Ta
    diffusivity::Ts
    brightness::Ts
    iteration::Int # iteration
    temperature::Union{Ts,Int} # temperature
    log𝒫::Ts # log posterior
    logℒ::Ts # log likelihood
end

Sample(x::Array, M, D, h, i, T, log𝒫, logℒ) = Sample(x[:, 1:M, :], D, h, i, T, log𝒫, logℒ)

Sample(x::AbstractArray{T,3}, M::Integer, D::T, h::T) where {T<:AbstractFloat} =
    Sample(x, M, D, h, 0, oneunit(T), convert(T, NaN), convert(T, NaN))

get_B(v::AbstractVector{Sample}) = [size(s.tracks, 2) for s in v]

get_D(v::AbstractVector{Sample}) = [s.D for s in v]

get_h(v::AbstractVector{Sample}) = [s.h for s in v]

struct Chain{Ts,Ta}
    samples::Vector{Ts}
    # 𝐔::T1
    # 𝐔ᵖ::T1
    # Δ𝐔::T1 # a scratch space
    # iteration::Int # iteration
    sizelimit::Int
    # stride::Int
    # temperature::S
    # logℒ::S # log likelihood
    # log𝒫::S # log posterior
    annealing::Ta # annealing
end

# function Chain(
#     T::DataType;
#     # x,
#     # frames,
#     sizelimit::Integer,
#     iteration = 1,
#     stride = 1,
#     annealing = nothing,
# )
#     # T = eltype(x)
#     # 𝐔 = similar(x, size(frames)...)
#     # 𝐔ᵖ = similar(𝐔)
#     # temp = similar(𝐔)
#     isnothing(annealing) && (annealing = PolynomialAnnealing{T}())
#     temperature = get_temperature(annealing, 1)
#     return Chain(
#         # 𝐔,
#         # 𝐔ᵖ,
#         # temp,
#         iteration,
#         sizelimit,
#         stride,
#         temperature,
#         convert(T, NaN),
#         convert(T, NaN),
#         annealing,
#     )
# end

isfull(chain::Chain) = length(chain.samples) == chain.sizelimit

function shrink!(chain::Chain)
    deleteat!(chain.samples, 2:2:lastindex(chain.samples))
    return chain
end

temperature(chain::Chain, i) = temperature(chain.annealing, i)

saveperiod(chain::Chain) =
    length(chain.samples) == 1 ? 1 : chain.samples[2].iteration - chain.samples[1].iteration

struct AuxiliaryVariables{T}
    Δx²::T
    Δxᵖ²::T
    U::T
    Uᵖ::T
    ΔU::T # a scratch space
end

function AuxiliaryVariables(x, xbnds, ybnds)
    Δx² = similar(x, size(x, 1), size(x, 2), size(x, 3) - 1)
    U = similar(x, length(xbnds) - 1, length(ybnds) - 1, size(x, 3))
    return AuxiliaryVariables(Δx², similar(Δx²), U, similar(U), similar(U))
end

function diff²!(aux::AuxiliaryVariables, x)
    diff²!(aux.Δx², x)
    return aux
end

# function set_temperature!(c::Chain)
#     c.temperature = get_temperature(c.annealing, c.iteration)
#     return c
# end

# get_temperature(c::Chain) = get_temperature(c.annealing, c.iteration)

# get_temperature(c::ChainParameters) = get_temperature(c.annealing, c.iteration)

# function update!(c::Chain, iter)
#     c.iteration = iter
#     set_temperature!(c)
#     return c
# end

# ChainStatus contains auxiliary variables
# mutable struct ChainStatus{T<:Real,AT<:AbstractArray{T}}
#     tracks::MHArray{AT}
#     inactivetracks::DSArray{AT}
#     emittercount::DSScalar{Int}
#     diffusivity::DSScalar{T}
#     brightness::MHScalar{T}
#     alltracks::AbstractArray{T,3}
#     𝐔::AbstractArray{T,3}
#     𝐔ᵖ::AbstractArray{T,3}
#     iteration::Int # iteration
#     temperature::T # temperature
#     logposterior::T # log posterior
#     loglikelihood::T # log likelihood
#     function ChainStatus(
#         tracks::MHArray{AT},
#         emittercount::DSScalar{Int},
#         diffusivity::DSScalar{T},
#         brightness::MHScalar{T},
#         𝐔::AbstractArray{T,3},
#         iteration::Int = 0,
#         temperature::T = 1.0,
#         logposterior::T = NaN,
#         loglikelihood::T = NaN,
#     ) where {T<:Real,AT<:AbstractArray{T}}
#         inactive_x = similar(tracks.value)
#         return new{T,AT}(
#             tracks,
#             emittercount,
#             diffusivity,
#             brightness,
#             𝐔,
#             copy(𝐔),
#             iteration,
#             temperature,
#             logposterior,
#             loglikelihood,
#         )
#     end
# end

# # get_B(s::ChainStatus) = count(s.b.value)

# get_M(s::ChainStatus) = size(s.tracks.value, 2)

# _eltype(::ChainStatus{T}) where {T} = T

# view_on_x(s::ChainStatus) = (
#     view(s.tracks.value, :, 1:s.emittercount.value, :),
#     view(s.tracks.candidate, :, 1:s.emittercount.value, :),
# )

# view_off_x(s::ChainStatus) = @view s.tracks.value[:, s.emittercount.value+1:end, :]

# viewdiag(M::AbstractMatrix) = view(M, diagind(M))

# default_init_pos_prior(param::ExperimentalParameters) = MvNormal(
#     [param.pxboundsx[end], param.pxboundsy[end], 0] ./ 2,
#     getpxsize(param) .* [2, 2, 0],
# )

# mutable struct Chain{T<:AbstractFloat}
#     status::ChainStatus{T}
#     samples::Vector{Sample{T}}
#     annealing::AbstractAnnealing{T}
#     stride::Int
#     sizelimit::Int
# end

# chainlength(c::Chain) = length(c.samples)

# _eltype(::Chain{T}) where {T} = T

# isfull(c::Chain) = chainlength(c) > c.sizelimit

# function get_x(S::AbstractVector{Sample{T}}) where {T}
#     M = get_B.(S)
#     N = size(S[1].x, 3)
#     x = Array{T}(undef, sum(M), N, 3)
#     𝒷 = 0
#     @views for (s, m) in zip(S, M)
#         permutedims!(x[𝒷.+(1:m), :, :], s.x, (2, 3, 1))
#         𝒷 += m
#     end
#     return x
# end

"""
    shrink!(chain)

Shrink the chain of samples by only keeping the odd number samples.
"""
# function shrink!(c::Chain)
#     deleteat!(c.samples, 2:2:lastindex(c.samples))
#     c.stride *= 2
# end

"""
    extend!(chain::Chain)

Push the chain's current 'status' (a full sample)  to `samples` and check if the updated chain has reached the `sizelimit`. If so, call `shrink!`.
"""
# function extend!(c::Chain)
#     push!(c.samples, Sample(c.status))
#     isfull(c) && shrink!(c)
#     return c
# end

# function to_cpu!(c::Chain)
#     s = c.status
#     x = MHArray(Array(s.tracks.value), s.tracks.dynamics, s.tracks.prior, s.tracks.proposal)
#     𝐔 = Array(s.𝐔)
#     c.status = ChainStatus(
#         x,
#         s.emittercount,
#         s.diffusivity,
#         s.brightness,
#         𝐔,
#         iszero(s.iteration) ? 1 : s.iteration,
#         s.temperature,
#         s.logposterior,
#         s.loglikelihood,
#     )
#     return c
# end

# function to_cpu!(v::Video)
#     to_cpu!(v.param)
#     v.frames = Array(v.frames)
#     return v
# end

# function to_gpu!(c::Chain)
#     s = c.status
#     x = MHArray(
#         CuArray(s.tracks.value),
#         s.tracks.dynamics,
#         s.tracks.prior,
#         s.tracks.proposal,
#     )
#     𝐔 = CuArray(s.𝐔)
#     c.status = ChainStatus(
#         x,
#         s.emittercount,
#         s.diffusivity,
#         s.brightness,
#         𝐔,
#         iszero(s.iteration) ? 1 : s.iteration,
#         s.temperature,
#         s.logposterior,
#         s.loglikelihood,
#     )
#     return c
# end

# function run_MCMC!(
#     c::Chain,
#     v::Video;
#     num_iter::Union{Integer,Nothing} = nothing,
#     run_on_gpu::Bool = true,
# )
#     iter::Int = 0
#     if run_on_gpu && has_cuda_gpu()
#         CUDA.allowscalar(false)
#         to_gpu!(c)
#         to_gpu!(v)
#     end
#     @showprogress 1 "Computing..." for iter = 1:num_iter
#         c.status.iteration = iter
#         update_tracks!(c.status, v)
#         # update_M!(c.status, v)
#         update_diffusivity!(c.status, v.param)
#         update_ln𝒫!(c.status, v)
#         if mod(iter, c.stride) == 0
#             extend!(c)
#         end
#     end
#     return c
# end
