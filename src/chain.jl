struct Sample{Ts,Ta}
    tracks::Ta
    diffusivity::Ts
    brightness::Ts
    iteration::Int # iteration
    𝑇::Ts # temperature
    log𝒫::Ts # log posterior
    logℒ::Ts # log likelihood
end

Sample(
    x::AbstractArray{T,3},
    M::Integer,
    D::T,
    h::T,
    i::Integer,
    𝑇::T,
    log𝒫::T,
    logℒ::T,
) where {T} = Sample(collect(view(x, :, :, 1:M)), D, h, i, 𝑇, log𝒫, logℒ)

Sample(x::AbstractArray{T,3}, M::Integer, D::T, h::T) where {T<:AbstractFloat} =
    Sample(x, M, D, h, 0, oneunit(T), convert(T, NaN), convert(T, NaN))

get_B(v::AbstractVector{Sample}) = [size(s.tracks, 2) for s in v]

get_D(v::AbstractVector{Sample}) = [s.D for s in v]

get_h(v::AbstractVector{Sample}) = [s.h for s in v]

struct Chain{Ts,Ta}
    samples::Vector{Ts}
    sizelimit::Int
    annealing::Ta # annealing
end

isfull(chain::Chain) = length(chain.samples) == chain.sizelimit

function shrink!(chain::Chain)
    deleteat!(chain.samples, 2:2:lastindex(chain.samples))
    return chain
end

temperature(chain::Chain, i::Real) = temperature(chain.annealing, i)

saveperiod(chain::Chain) =
    length(chain.samples) == 1 ? 1 : chain.samples[2].iteration - chain.samples[1].iteration

struct AuxiliaryVariables{T}
    Δ𝐱²::AbstractArray{T,3}
    Δ𝐲²::AbstractArray{T,3}
    ΔΔ𝐱²::AbstractArray{T,3}
    ΣΔΔ𝐱²::AbstractVector{T}
    Sᵥ::AbstractVector{T} # scratch vector
    U::AbstractArray{T,3}
    V::AbstractArray{T,3}
    Sₐ::AbstractArray{T,3} # scratch array
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