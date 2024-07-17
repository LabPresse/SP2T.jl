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
    x::Array{T,3},
    M::Integer,
    D::T,
    h::T,
    i::Integer,
    𝑇::T,
    log𝒫::T,
    logℒ::T,
) where {T} = Sample(x[:, :, 1:M], D, h, i, 𝑇, log𝒫, logℒ)

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

struct AuxiliaryVariables{Ta,Tv}
    Δx²::Ta
    Δy²::Ta
    ΔΔx²::Ta
    ΣΔΔx²::Tv
    ΔlogP::Tv
    U::Ta
    V::Ta
    Sᵤ::Ta # S for scratch
end

function AuxiliaryVariables(
    x::AbstractArray{T,3},
    xbnds::AbstractVector{T},
    ybnds::AbstractVector{T},
    F::AbstractMatrix{T},
) where {T}
    N = size(x, 1)
    Δx² = similar(x, N - 1, size(x, 2), size(x, 3))
    U = similar(x, length(xbnds) - 1, length(ybnds) - 1, N)
    V = similar(U)
    V .= F
    ΣΔΔx² = similar(x, N - 1)
    ΔlogP = similar(x, N)
    return AuxiliaryVariables(
        Δx²,
        similar(Δx²),
        similar(Δx²),
        ΣΔΔx²,
        ΔlogP,
        U,
        V,
        similar(U),
    )
end

displacements(aux::AuxiliaryVariables, M::Integer) =
    @views aux.Δx²[:, :, 1:M], aux.Δy²[:, :, 1:M], aux.ΔΔx²[:, :, 1:M]