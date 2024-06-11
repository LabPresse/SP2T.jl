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
    sizelimit::Int
    annealing::Ta # annealing
end

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
    Δy²::T
    ΔΔx²::T
    ΣΔΔx²::T
    ΔlogP::T
    U::T
    V::T
    ΔU::T
end

function AuxiliaryVariables(
    x::AbstractArray{T,3},
    xbnds::AbstractVector,
    ybnds::AbstractVector,
    F::AbstractMatrix{T},
) where {T}
    Δx² = similar(x, size(x, 1), size(x, 2), size(x, 3) - 1)
    U = similar(x, length(xbnds) - 1, length(ybnds) - 1, size(x, 3))
    V = similar(U)
    V .= F
    ΣΔΔx² = similar(x, 1, 1, size(x, 3) - 1)
    ΔlogP = similar(x, 1, 1, size(x, 3))
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
    @views aux.Δx²[:, 1:M, :], aux.Δy²[:, 1:M, :], aux.ΔΔx²[:, 1:M, :]

function diff²!(aux::AuxiliaryVariables, x)
    diff²!(aux.Δx², x)
    return aux
end
