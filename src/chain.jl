struct Sample{T<:AbstractFloat,A<:AbstractArray{T,3}}
    tracks::A
    diffusivity::T
    brightness::T
    iteration::Int # iteration
    𝑇::T # temperature
    log𝒫::T # log posterior
    logℒ::T # log likelihood
end

Sample(
    x::AbstractArray{T,3},
    M::Integer,
    msd::T,
    h::T,
    i::Integer,
    𝑇::T,
    log𝒫::T,
    logℒ::T,
) where {T} = Sample(collect(view(x, :, :, 1:M)), msd, h, i, 𝑇, log𝒫, logℒ)

Sample(x::AbstractArray{T,3}, M::Integer, msd::T, h::T) where {T<:AbstractFloat} =
    Sample(x, M, msd, h, 0, oneunit(T), convert(T, NaN), convert(T, NaN))

get_B(v::AbstractVector{Sample}) = [size(s.tracks, 2) for s in v]

# get_D(v::AbstractVector{Sample}) = [s.D for s in v]

# get_h(v::AbstractVector{Sample}) = [s.h for s in v]

mutable struct Chain{T<:AbstractFloat,VofS<:Vector{<:Sample{T}},A<:AbstractAnnealing{T}}
    samples::VofS
    sizelimit::Int
    annealing::A
end

function Base.getproperty(c::Chain, s::Symbol)
    if s === :msd
        return [sample.diffusivity for sample in getfield(c, :samples)]
    elseif s === :nemitters
        return [size(sample.tracks, 3) for sample in getfield(c, :samples)]
    elseif s === :logposterior
        return [sample.log𝒫 for sample in getfield(c, :samples)]
    elseif s === :loglikelihood
        return [sample.logℒ for sample in getfield(c, :samples)]
    elseif s === :stride
        return length(getfield(c, :samples)) == 1 ? 1 :
               getfield(c, :samples)[2].iteration - getfield(c, :samples)[1].iteration
    else
        return getfield(c, s)
    end
end

Base.length(c::Chain) = length(c.samples)

isfull(chain::Chain) = length(chain.samples) == chain.sizelimit

function shrink!(chain::Chain)
    deleteat!(chain.samples, 2:2:lastindex(chain.samples))
    return chain
end

temperature(chain::Chain, i::Real) = temperature(chain.annealing, i)

# saveperiod(chain::Chain) =
#     length(chain.samples) == 1 ? 1 : chain.samples[2].iteration - chain.samples[1].iteration

struct AuxiliaryVariables{T<:AbstractFloat,A<:AbstractArray{T,3},V<:AbstractVector{T}}
    Δ𝐱²::A
    Δ𝐲²::A
    ΔΔ𝐱²::A
    ΣΔΔ𝐱²::V
    Sᵥ::V # scratch vector
    U::A
    V::A
    Sₐ::A # scratch array
end

AuxiliaryVariables(
    x::AbstractArray{T,3},
    dims::Tuple{<:Integer,<:Integer,<:Integer},
) where {T} = AuxiliaryVariables(
    similar(x, dims[3] - 1, size(x, 2), size(x, 3)),
    similar(x, dims[3] - 1, size(x, 2), size(x, 3)),
    similar(x, dims[3] - 1, size(x, 2), size(x, 3)),
    similar(x, dims[3] - 1),
    similar(x, dims[3]),
    similar(x, dims[1], dims[2], dims[3]),
    similar(x, dims[1], dims[2], dims[3]),
    similar(x, dims[1], dims[2], dims[3]),
)

displacements(aux::AuxiliaryVariables, M::Integer) =
    @views aux.Δ𝐱²[:, :, 1:M], aux.Δ𝐲²[:, :, 1:M], aux.ΔΔ𝐱²[:, :, 1:M]