abstract type SimplifiedDistribution{T} end

struct Normal₃{T} <: SimplifiedDistribution{T}
    μ::T
    σ::T
end

_params(n::Normal₃) = n.μ, n.σ
struct BrownianTracks{Ta,Tv,B}
    value::Ta
    valueᵖ::Ta
    prior::Normal₃{Tv}
    perturbsize::Tv
    logratio::Ta
    logrands::Ta
    accepted::B
    counter::Matrix{Int}
end

function BrownianTracks(
    x::AbstractArray{T,3},
    xᵖ::AbstractArray{T,3},
    prior::Normal₃{<:AbstractVector{T}},
    perturbsize::AbstractVector{T},
) where {T}
    logratio = similar(x, 1, 1, axes(x, 3))
    acceptance = fill!(similar(logratio, Bool), false)
    return BrownianTracks(
        x,
        xᵖ,
        prior,
        perturbsize,
        logratio,
        similar(logratio),
        acceptance,
        zeros(Int, 2, 2),
    )
end

function BrownianTracks(; value, prior, perturbsize)
    valueᵖ = similar(value)
    logacceptance = similar(value, 1, 1, axes(value, 3))
    acceptance = similar(logacceptance, Bool)
    return BrownianTracks(
        value,
        valueᵖ,
        prior,
        perturbsize,
        logacceptance,
        similar(logacceptance),
        acceptance,
        zeros(Int, 2, 2),
    )
end

# maxnemitters(tracks::BrownianTracks) = size(tracks.x, 2)

# viewtracks(tracks::BrownianTracks, M) = view(tracks.x, :, 1:M, :)

ontracks(x::BrownianTracks, M) = @views x.value[:, 1:M, :], x.valueᵖ[:, 1:M, :]

# candidates(tracks::BrownianTracks, M) = view(tracks.xᵖ, :, 1:M, :)

# displacements(tracks::BrownianTracks, M) =
#     view(tracks.Δx², :, 1:M, :), view(tracks.Δxᵖ², :, 1:M, :)

logrand!(x::AbstractArray) = x .= log.(rand!(x))

neglogrand!(x::AbstractArray) = x .= .-log.(rand!(x))

# function logrand!(tracks::BrownianTracks)
#     rand!(tracks.lograndnums)
#     tracks.lograndnums .= log.(tracks.lograndnums)
#     return tracks
# end

# diff2!(Δx², x) = @views Δx² .= (x[:, :, 2:end] .- x[:, :, 1:end-1]) .^ 2

# function setΔx²!(tracks::BrownianTracks)
#     diff2!(tracks.Δx², tracks.x)
#     return tracks
# end

function simulate!(x, μ, σ, D)
    randn!(x)
    @views begin
        x[:, :, 1] .= x[:, :, 1] .* σ .+ μ
        x[:, :, 2:end] .*= √(2 * D)
    end
    return cumsum!(x, x, dims = 3)
end

# function simulate!(x, Δx², μ, σ, D)
#     simulate!(x, μ, σ, D)
#     @views Δx² .= x[:, :, 2:end] .^ 2
#     return x
# end

function MHinit!(x::BrownianTracks)
    neglogrand!(x.logratio)
    fill!(x.accepted, false)
    # logrand!(x.logrands)
    return x
end

function propose!(y::AbstractArray{T}, x::AbstractArray{T}, σ::AbstractArray{T}) where {T}
    randn!(y)
    y .= y .* σ .+ x
end

propose!(y::AbstractArray{T}, x::AbstractArray{T}, t::BrownianTracks) where {T} =
    propose!(y, x, t.perturbsize)

Δlogπ₁(x₁::AbstractMatrix{T}, y₁::AbstractMatrix{T}, prior::Normal₃) where {T} =
    sum(((x₁ .- prior.μ) .^ 2 - (y₁ .- prior.μ) .^ 2) ./ (2 .* prior.σ .^ 2))

Δlogπ₁(x::AbstractArray{T,3}, y::AbstractArray{T,3}, prior::Normal₃) where {T} =
    @views Δlogπ₁(x[:, :, 1], y[:, :, 1], prior)

function addΔlogπ₁!(
    ln𝓇::Array{T},
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    prior::Normal₃,
) where {T}
    ln𝓇[1] += Δlogπ₁(x, y, prior)
    return ln𝓇
end

# function add_Δlog𝒫!(x::BrownianTracks, M::Integer)
#     @views add_Δlog𝒫!(x.logratio[:, :, 1], x.value[:, 1:M, 1], x.valueᵖ[:, 1:M, 1], x.prior)
#     return x
# end

# function add_Δln𝒫_x₁!(tracks::BrownianTracks, x, xᵖ)
#     add_Δln𝒫_x₁!(tracks.logacceptance, x, xᵖ, tracks.prior)
#     return tracks
# end

# copyidxto!(x, xᵖ, tracks::BrownianTracks) = copyidxto!(x, xᵖ, vec(tracks.accepted))

diff²!(Δx²::AbstractArray{T,3}, x::AbstractArray{T,3}) where {T} =
    @views Δx² .= (x[:, :, 2:end] .- x[:, :, 1:end-1]) .^ 2

function diff²!(
    Δx²::AbstractArray{T,3},
    x1::AbstractArray{T,3},
    x2::AbstractArray{T,3},
) where {T}
    @views begin
        Δx²[:, :, 1:2:end] .= (x1[:, :, 2:2:end] .- x2[:, :, 1:2:end-1]) .^ 2
        Δx²[:, :, 2:2:end] .= (x2[:, :, 3:2:end] .- x1[:, :, 2:2:end-1]) .^ 2
    end
    return Δx²
end

function ΣΔΔx²!(
    ΣΔΔx²::AbstractArray{T,3},
    ΔΔx²::AbstractArray{T,3},
    Δx²::AbstractArray{T,3},
    Δy²::AbstractArray{T,3},
    D::T,
) where {T}
    ΔΔx² .= Δx² .- Δy²
    sum!(ΣΔΔx², ΔΔx²)
    ΣΔΔx² ./= 4 * D
end

# function setΔx²!(tracks::BrownianTracks)
#     setΔx²!(tracks.Δx², tracks.x)
#     return tracks
# end

# function add_ΔΔx²!(
#     logr::AbstractArray{T},
#     Δx²::AbstractArray{T,3},
#     Δy²::AbstractArray{T,3},
#     idx1::StepRange,
#     idx2::StepRange,
#     D::T,
# ) where {T}
#     @views begin
#         logr[:, :, idx1] .+=
#             sum(Δx²[:, :, idx1] .- Δy²[:, :, idx1], dims = (1, 2)) ./ (4 * D)
#         logr[:, :, idx2.+1] .+=
#             sum(Δx²[:, :, idx2] .- Δy²[:, :, idx2], dims = (1, 2)) ./ (4 * D)
#     end
#     return logr
# end

# add_odd_ΔΔx²!(
#     logr::AbstractArray{T},
#     Δx²::AbstractArray{T,3},
#     Δy²::AbstractArray{T,3},
#     D::T,
# ) where {T} = add_ΔΔx²!(logr, Δx², Δy², 1:2:size(Δx², 3), 2:2:size(Δx², 3), D)

# add_even_ΔΔx²!(
#     logr::AbstractArray{T},
#     Δx²::AbstractArray{T,3},
#     Δy²::AbstractArray{T,3},
#     D::T,
# ) where {T} = add_ΔΔx²!(logr, Δx², Δy², 2:2:size(Δx², 3), 1:2:size(Δx², 3), D)

function counter!(x::BrownianTracks)
    @views x.counter[:, 2] .+= count(x.accepted), length(x.accepted)
    return x
end

function copyidxto!(x::AbstractArray{T}, y::AbstractArray{T}, i::Array{<:Integer}) where {T}
    j = vec(i)
    @views x[:, :, j] .= y[:, :, j]
    return x
end

function Δlogπ!(
    logr::AbstractArray{T},
    idx1::StepRange,
    idx2::StepRange,
    ΣΔΔx²::AbstractArray{T},
) where {T}
    @views copyto!(logr[idx1], ΣΔΔx²[idx1])
    @views logr[idx2.+1] .+= ΣΔΔx²[idx2]
    return logr
end

oddΔlogπ!(logr::AbstractArray{T}, ΣΔΔx²::AbstractArray{T}) where {T} =
    Δlogπ!(logr, 1:2:length(ΣΔΔx²), 2:2:length(ΣΔΔx²), ΣΔΔx²)

evenΔlogπ!(logr::AbstractArray{T}, ΣΔΔx²::AbstractArray{T}) where {T} =
    Δlogπ!(logr, 2:2:length(ΣΔΔx²), 1:2:length(ΣΔΔx²), ΣΔΔx²)

# function accept!(
#     x::AbstractArray{T,3},
#     y::AbstractArray{T,3},
#     accept::AbstractArray{Bool},
#     logratio::AbstractArray{T},
# ) where {T}
#     accept .= logratio .> 0
#     copyidxto!(x, y, accept)
# end

# accept_odd!(
#     x::AbstractArray{T,3},
#     y::AbstractArray{T,3},
#     accept::AbstractArray{Bool,3},
#     logratio::AbstractArray{T,3},
# ) where {T} =
#     @views accept!(x[:, :, 1:2:end], y[:, :, 1:2:end], accept[1:2:end], logratio[1:2:end])

# accept_even!(
#     x::AbstractArray{T,3},
#     y::AbstractArray{T,3},
#     accept::AbstractArray{Bool,3},
#     logratio::AbstractArray{T,3},
# ) where {T} =
#     @views accept!(x[:, :, 2:2:end], y[:, :, 2:2:end], accept[2:2:end], logratio[2:2:end])

function oddΔlogπ!(
    Δlogπ::AbstractArray{T,3},
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    Δx²::AbstractArray{T,3},
    Δy²::AbstractArray{T,3},
    D::T,
    ΔΔx²::AbstractArray{T,3},
    ΣΔΔx²::AbstractArray{T,3},
) where {T}
    diff²!(Δx², x)
    diff²!(Δy², x, y)
    ΣΔΔx²!(ΣΔΔx², ΔΔx², Δx², Δy², D)
    oddΔlogπ!(Δlogπ, ΣΔΔx²)
end

function evenΔlogπ!(
    Δlogπ::AbstractArray{T,3},
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    Δx²::AbstractArray{T,3},
    Δy²::AbstractArray{T,3},
    D::T,
    ΔΔx²::AbstractArray{T,3},
    ΣΔΔx²::AbstractArray{T,3},
) where {T}
    diff²!(Δx², x)
    diff²!(Δy², y, x)
    ΣΔΔx²!(ΣΔΔx², ΔΔx², Δx², Δy², D)
    evenΔlogπ!(Δlogπ, ΣΔΔx²)
end

# function update_odd!(
#     x::AbstractArray{T,3},
#     y::AbstractArray{T,3},
#     Δx²::AbstractArray{T,3},
#     Δy²::AbstractArray{T,3},
#     D::T,
#     logr::AbstractArray{T,3},
#     accept::AbstractArray{Bool,3},
#     ΔΔx²::AbstractArray{T,3},
#     ΣΔΔx²::AbstractArray{T,3},
#     Δlogπ::AbstractArray{T,3},
# ) where {T}
#     oddΔlogπ!(Δlogπ, x, y, Δx², Δy², D, ΔΔx², ΣΔΔx²)
#     @views begin
#         logr[1:2:end] .+= Δlogπ[1:2:end]
#         # accept_odd!(x, y, accept, logr)
#         accept[1:2:end] .= logr[1:2:end] .> 0
#     end
#     copyidxto!(x, y, accept)
# end

# function update_even!(
#     x::AbstractArray{T,3},
#     y::AbstractArray{T,3},
#     Δx²::AbstractArray{T,3},
#     Δy²::AbstractArray{T,3},
#     D::T,
#     logr::AbstractArray{T,3},
#     accept::AbstractArray{Bool,3},
#     ΔΔx²::AbstractArray{T,3},
#     ΣΔΔx²::AbstractArray{T,3},
#     Δlogπ::AbstractArray{T,3},
# ) where {T}
#     evenΔlogπ!(Δlogπ, x, y, Δx², Δy², D, ΔΔx², ΣΔΔx²)
#     @views begin
#         logr[2:2:end] .+= Δlogπ[2:2:end]
#         # accept_even!(x, y, accept, logr)
#         accept[2:2:end] .= logr[2:2:end] .> 0
#     end
#     copyidxto!(x, y, accept)
# end