abstract type SimplifiedDistribution{T} end

struct Normal₃{T} <: SimplifiedDistribution{T}
    μ::T
    σ::T
end

Distributions.params(n::Normal₃) = n.μ, n.σ
struct Tracks{T,Ta,Tv,B}
    value::Ta
    valueᵖ::Ta
    prior::Normal₃{Tv}
    perturbsize::Tv
    logratio::Tv
    accepted::B
    counter::Matrix{Int}
end

# function Tracks{T}(
#     x::AbstractArray{<:Real,3},
#     xᵖ::AbstractArray{<:Real,3},
#     prior::Normal₃{<:AbstractVector{<:Real}},
#     perturbsize::AbstractVector{<:Real},
# ) where {T<:AbstractFloat}
#     logratio = similar(x, T, axes(x, 1))
#     accepted = fill!(similar(logratio, Bool), false)
#     return Tracks{T,typeof(x),typeof(logratio),typeof(accepted)}(
#         convert.(T, x),
#         xᵖ,
#         prior,
#         perturbsize,
#         logratio,
#         accepted,
#         zeros(Int, 2, 2),
#     )
# end

function Tracks{T}(;
    value::AbstractArray{<:Real,3},
    prior::Normal₃{<:AbstractVector{<:Real}},
    perturbsize::AbstractVector{<:Real},
) where {T<:AbstractFloat}
    value = convert.(T, value)
    valueᵖ = similar(value)
    logratio = similar(value, T, axes(value, 1))
    accepted = similar(logratio, Bool)
    return Tracks{T,typeof(value),typeof(logratio),typeof(accepted)}(
        value,
        valueᵖ,
        prior,
        perturbsize,
        logratio,
        accepted,
        zeros(Int, 2, 2),
    )
end

function Tracks(;
    value::AbstractArray{<:Real,3},
    prior::Normal₃{<:AbstractVector{<:Real}},
    perturbsize::AbstractVector{<:Real},
)
    valueᵖ = similar(value)
    logratio = similar(value, axes(value, 1))
    accepted = similar(logratio, Bool)
    return Tracks{eltype(value),typeof(value),typeof(logratio),typeof(accepted)}(
        value,
        valueᵖ,
        prior,
        perturbsize,
        logratio,
        accepted,
        zeros(Int, 2, 2),
    )
end

ontracks(x::Tracks, M::Integer) = @views x.value[:, :, 1:M], x.valueᵖ[:, :, 1:M]

logrand!(x::AbstractArray) = x .= log.(rand!(x))

neglogrand!(x::AbstractArray) = x .= .-log.(rand!(x))

function _randn!(x::AbstractArray{T}, σ::Union{T,AbstractMatrix{T}}) where {T}
    randn!(x)
    x .*= σ
end

function _randn!(x::AbstractArray{T,3}, σ::T, σ₀::AbstractVecOrMat{T}) where {T}
    _randn!(x, σ)
    @views @. x[1, :, :] *= σ₀ / σ
end

function simulate!(
    x::AbstractArray{T,3},
    μ::AbstractVector{T},
    σ::AbstractVector{T},
    msd::T,
    y::AbstractArray{T,3},
) where {T}
    _randn!(y, √msd, σ)
    cumsum!(x, y, dims = 1)
    x .+= reshape(μ, 1, :)
end

function simulate!(
    x::AbstractArray{T,3},
    μ::AbstractArray{T},
    σ::AbstractVector{T},
    msd::T,
    dims::Integer = 3,
) where {T}
    _randn!(x, √msd, σ)
    cumsum!(x, x, dims = 1)
    x .+= reshape(μ, 1, dims, :)
end

# function simulate!(
#     x::AbstractArray{T,3},
#     μ::AbstractArray{T,3},
#     σ::AbstractVector{T},
#     msd::T,
# ) where {T}
#     _randn!(x, √msd, σ)
#     cumsum!(x, x, dims = 1)
#     x .+= μ
# end

function bridge!(x::AbstractArray{T,3}, msd::T, xend::AbstractArray{T,3}) where {T}
    σ = fill!(similar(x, size(x, 2)), 0)
    simulate!(x, -diff(xend, dims = 1), σ, msd)
    N = size(x, 1) - 1
    @views @. x = x - (0:N) / N * x[end:end, :, :] + xend[2:2, :, :]
end

function MHinit!(x::Tracks)
    neglogrand!(x.logratio)
    fill!(x.accepted, false)
    return x
end

propose!(y::AbstractArray{T,3}, x::AbstractArray{T,3}, σ::AbstractVector{T}) where {T} =
    y .= x .+ transpose(σ) .* randn(T, size(y))

# propose!(y::AbstractArray{T}, x::AbstractArray{T}, t::BrownianTracks) where {T} =
#     propose!(y, x, t.perturbsize)

Δlogπ₁(
    x₁::AbstractMatrix{T},
    y₁::AbstractMatrix{T},
    μ::AbstractVector{T},
    σ::AbstractVector{T},
) where {T} = sum(vec(@. ((x₁ - μ)^2 - (y₁ - μ)^2) / (2 * σ^2)))

Δlogπ₁(x::AbstractArray{T,3}, y::AbstractArray{T,3}, prior::Normal₃) where {T} =
    @views Δlogπ₁(x[1, :, :], y[1, :, :], prior.μ, prior.σ)

function addΔlogπ₁!(
    ln𝓇::AbstractVector{T},
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    prior::Normal₃,
) where {T}
    ln𝓇[1] += Δlogπ₁(x, y, prior)
    return ln𝓇
end

diff²!(Δx²::AbstractArray{T,3}, x::AbstractArray{T,3}, y::AbstractArray{T,3}) where {T} =
    @views Δx² .= (y[2:end, :, :] .- x[1:end-1, :, :]) .^ 2

diff²!(Δx²::AbstractArray{T,3}, x::AbstractArray{T,3}) where {T} = diff²!(Δx², x, x)

function staggered_diff²!(
    Δx²::AbstractArray{T,3},
    x1::AbstractArray{T,3},
    x2::AbstractArray{T,3},
) where {T}
    @views begin
        Δx²[1:2:end, :, :] .= (x1[2:2:end, :, :] .- x2[1:2:end-1, :, :]) .^ 2
        Δx²[2:2:end, :, :] .= (x2[3:2:end, :, :] .- x1[2:2:end-1, :, :]) .^ 2
    end
    return Δx²
end

function ΣΔΔx²!(
    ΣΔΔx²::AbstractVector{T},
    ΔΔx²::AbstractArray{T,3},
    Δx²::AbstractArray{T,3},
    Δy²::AbstractArray{T,3},
    msd::T,
) where {T}
    ΔΔx² .= Δx² .- Δy²
    sum!(ΣΔΔx², ΔΔx²)
    ΣΔΔx² ./= 2 * msd
end

function counter!(x::Tracks)
    @views x.counter[:, 2] .+= count(x.accepted), length(x.accepted)
    return x
end

function copyidxto!(
    dest::AbstractArray{T,3},
    src::AbstractArray{T,3},
    i::Vector{Bool},
) where {T}
    @views dest[i, :, :] .= src[i, :, :]
    return dest
end

function Δlogπ!(
    logr::AbstractVector{T},
    idx1::StepRange,
    idx2::StepRange,
    ΣΔΔx²::AbstractVector{T},
) where {T}
    @views copyto!(logr[idx1], ΣΔΔx²[idx1])
    @views logr[idx2.+1] .+= ΣΔΔx²[idx2]
    return logr
end

oddΔlogπ!(logr::AbstractVector{T}, ΣΔΔx²::AbstractVector{T}) where {T} =
    Δlogπ!(logr, 1:2:length(ΣΔΔx²), 2:2:length(ΣΔΔx²), ΣΔΔx²)

evenΔlogπ!(logr::AbstractVector{T}, ΣΔΔx²::AbstractVector{T}) where {T} =
    Δlogπ!(logr, 2:2:length(ΣΔΔx²), 1:2:length(ΣΔΔx²), ΣΔΔx²)

function oddΔlogπ!(
    Δlogπ::AbstractVector{T},
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    Δx²::AbstractArray{T,3},
    Δy²::AbstractArray{T,3},
    msd::T,
    ΔΔx²::AbstractArray{T,3},
    ΣΔΔx²::AbstractVector{T},
) where {T}
    diff²!(Δx², x)
    staggered_diff²!(Δy², x, y)
    ΣΔΔx²!(ΣΔΔx², ΔΔx², Δx², Δy², msd)
    oddΔlogπ!(Δlogπ, ΣΔΔx²)
end

function evenΔlogπ!(
    Δlogπ::AbstractVector{T},
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    Δx²::AbstractArray{T,3},
    Δy²::AbstractArray{T,3},
    msd::T,
    ΔΔx²::AbstractArray{T,3},
    ΣΔΔx²::AbstractVector{T},
) where {T}
    diff²!(Δx², x)
    staggered_diff²!(Δy², y, x)
    ΣΔΔx²!(ΣΔΔx², ΔΔx², Δx², Δy², msd)
    evenΔlogπ!(Δlogπ, ΣΔΔx²)
end