struct Tracks{
    T,
    A<:AbstractArray{T,4},
    P<:SP2TDistribution{T},
    V<:AbstractVector{T},
    B<:AbstractVector{Bool},
} <: RandomVariable{T}
    fullvalue::A
    prior::P
    fulldisplacement²::A
    ΣΔdisplacement²::V
    perturbsize::V
    logratio::V
    accepted::B
    counter::Matrix{Int}
end

function Tracks{T}(;
    value::AbstractArray{<:Real,3},
    prior::SP2TDistribution{<:Real},
    perturbsize::AbstractVector{<:Real},
) where {T<:AbstractFloat}
    fullvalue = similar(value, T, size(value)..., 2)
    nframes, ndims, nemitters = size(value)
    fulldisplacement² = similar(fullvalue, nframes - 1, ndims, nemitters, 2)
    logratio = similar(fullvalue, nframes)
    accepted = similar(fullvalue, Bool, nframes)
    counter = zeros(Int, 2, 2)
    return Tracks(
        fullvalue,
        prior,
        fulldisplacement²,
        similar(logratio, nframes - 1),
        perturbsize,
        logratio,
        accepted,
        counter,
    )
end

function Base.getproperty(tracks::Tracks, s::Symbol)
    if s === :value
        return selectdim(getfield(tracks, :fullvalue), 4, 1)
    elseif s === :proposal
        return selectdim(getfield(tracks, :fullvalue), 4, 2)
    elseif s === :displacement²
        return selectdim(getfield(tracks, :fulldisplacement²), 4, 1)
    elseif s === :proposaldisplacement²
        return selectdim(getfield(tracks, :fulldisplacement²), 4, 2)
    else
        return getfield(tracks, s)
    end
end

viewactive(tracks::Tracks, ntracksᵥ::Integer) =
    @views tracks.fullvalue[:, :, 1:ntracksᵥ, 1],
    tracks.fullvalue[:, :, 1:ntracksᵥ, 2],
    tracks.fulldisplacement²[:, :, 1:ntracksᵥ, 1],
    tracks.fulldisplacement²[:, :, 1:ntracksᵥ, 2]

function logprior(tracks::Tracks{T}, ntracksᵥ::Integer, msdᵥ::T) where {T}
    xᵒⁿ, ~, Δxᵒⁿ² = viewactive(tracks, ntracksᵥ)
    diff²!(Δxᵒⁿ², xᵒⁿ)
    return -(log(msdᵥ) * length(Δxᵒⁿ²) + sum(vec(Δxᵒⁿ²)) / msdᵥ) / 2 -
           _logπ(tracks.prior, view(xᵒⁿ, 1, :, :))
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
    μ::AbstractVector{T},
    σ::AbstractVector{T},
    msd::T,
) where {T}
    _randn!(x, √msd, σ)
    cumsum!(x, x, dims = 1)
    x .+= reshape(μ, 1, size(x, 2), :)
end

function simulate!(
    x::AbstractArray{T,3},
    μ::AbstractArray{T,3},
    σ::AbstractVector{T},
    msd::T,
) where {T}
    _randn!(x, √msd, σ)
    cumsum!(x, x, dims = 1)
    x .+= μ
end

function bridge!(x::AbstractArray{T,3}, msd::T, xend::AbstractArray{T,3}) where {T}
    σ = fill!(similar(x, size(x, 2)), 0)
    simulate!(x, -diff(xend, dims = 1), σ, msd)
    N = size(x, 1) - 1
    @views @. x = x - (0:N) / N * x[end:end, :, :] + xend[2:2, :, :]
end

function initacceptance!(tracks::Tracks)
    neglogrand!(tracks.logratio)
    fill!(tracks.accepted, false)
    return tracks
end

propose!(y::AbstractArray{T,3}, x::AbstractArray{T,3}, σ::AbstractVector{T}) where {T} =
    y .= x .+ transpose(σ) .* randn(T, size(y))

Δlogπ₁(
    x₁::AbstractMatrix{T},
    y₁::AbstractMatrix{T},
    μ::AbstractVector{T},
    σ::AbstractVector{T},
) where {T} = sum(vec(@. ((x₁ - μ)^2 - (y₁ - μ)^2) / (2 * σ^2)))

Δlogπ₁(x::AbstractArray{T,3}, y::AbstractArray{T,3}, prior::DNormal) where {T} =
    @views Δlogπ₁(x[1, :, :], y[1, :, :], prior.μ, prior.σ)

function addΔlogπ₁!(
    ln𝓇::AbstractVector{T},
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    prior::DNormal,
) where {T}
    ln𝓇[1] += Δlogπ₁(x, y, prior)
    return ln𝓇
end

function staggered_diff²!(
    Δx²::AbstractArray{T,3},
    even::AbstractArray{T,3},
    odd::AbstractArray{T,3},
) where {T}
    @views begin
        Δx²[1:2:end, :, :] .= (even[2:2:end, :, :] .- odd[1:2:end-1, :, :]) .^ 2
        Δx²[2:2:end, :, :] .= (odd[3:2:end, :, :] .- even[2:2:end-1, :, :]) .^ 2
    end
    return Δx²
end

function counter!(tracks::Tracks)
    @views tracks.counter[:, 2] .+= count(tracks.accepted), length(tracks.accepted)
    return tracks
end

function boolcopyto!(
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

function Δlogπ!(
    Δlogπ::AbstractVector{T},
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    Δx²::AbstractArray{T,3},
    Δy²::AbstractArray{T,3},
    msdᵥ::T,
    ΣΔΔx²::AbstractVector{T},
    i::Integer,
) where {T}
    diff²!(Δx², x)
    i == 1 ? staggered_diff²!(Δy², x, y) : staggered_diff²!(Δy², y, x)
    sum!(ΣΔΔx², Δx² .-= Δy²) ./= 2 * msdᵥ
    nframem1 = length(ΣΔΔx²)
    Δlogπ!(Δlogπ, i:2:nframem1, mod1(i + 1, 2):2:nframem1, ΣΔΔx²)
end

function update!(
    𝐱::AbstractArray{T,3},
    𝐲::AbstractArray{T,3},
    Δ𝐱²::AbstractArray{T,3},
    Δ𝐲²::AbstractArray{T,3},
    msdᵥ::T,
    logr::AbstractVector{T},
    accept::AbstractVector{Bool},
    ΣΔΔ𝐱²::AbstractVector{T},
    Δlogℒ::AbstractVector{T},
    i::Integer,
) where {T}
    Δlogπ!(Δlogℒ, 𝐱, 𝐲, Δ𝐱², Δ𝐲², msdᵥ, ΣΔΔ𝐱², i)
    @views logr[i:2:end] .+= Δlogℒ[i:2:end]
    logaccept!(accept, logr, start = i, step = 2)
    boolcopyto!(𝐱, 𝐲, accept)
end

function update_ontracks!(
    tracks::Tracks{T},
    ntracksᵥ::Integer,
    msdᵥ::T,
    brightnessᵥ::T,
    measurements::AbstractArray{<:Union{T,UInt16}},
    detector::PixelDetector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    initacceptance!(tracks)
    x, y, Δx², Δy² = viewactive(tracks, ntracksᵥ)
    propose!(y, x, tracks.perturbsize)
    pxcounts!(detector, x, y, brightnessᵥ, psf)
    Δlogℒ!(detector, measurements)
    tracks.logratio .+= anneal!(detector.framelogℒ, 𝑇)
    addΔlogπ₁!(tracks.logratio, x, y, tracks.prior)
    update!(
        x,
        y,
        Δx²,
        Δy²,
        msdᵥ,
        tracks.logratio,
        tracks.accepted,
        tracks.ΣΔdisplacement²,
        detector.framelogℒ,
        1,
    )
    update!(
        x,
        y,
        Δx²,
        Δy²,
        msdᵥ,
        tracks.logratio,
        tracks.accepted,
        tracks.ΣΔdisplacement²,
        detector.framelogℒ,
        2,
    )
    counter!(tracks)
    return tracks
end

function update_offtracks!(tracks::Tracks{T}, ntracksᵥ::Integer, msdᵥ::T) where {T}
    x = @view tracks.fullvalue[:, :, ntracksᵥ+1:end, 1]
    μ, σ = params(tracks.prior)
    simulate!(x, μ, σ, msdᵥ)
    return tracks
end