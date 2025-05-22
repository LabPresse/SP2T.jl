"""
    AbstractTrackChunk{T}

An abstract type representing a generic track part. The type parameter `T` can be used to specify the type of data.
"""
abstract type AbstractTrackChunk{T} end

Base.length(t::AbstractTrackChunk) = size(t.value, 1)

function setdisplacement²!(t::AbstractTrackChunk{T}) where {T}
    diff²!(t.displacement², t.value)
    return t
end

"""
    TrackChunk{T<:AbstractFloat, A<:AbstractArray{T}, P}

A struct that represents a track chunk. 'value::A' stores the particle locations in this chunk, `active::A` shares the same shape as `value` and denotes whether a particle is present (bright). `displacement²::A` is an auxiliary variable which stores squared displacements. `effvalue::A` is also an auxiliary variable which is often set to `value ./ active`.
"""
struct TrackChunk{T<:AbstractFloat,A<:AbstractArray{T},P} <: AbstractTrackChunk{T}
    value::A
    active::A
    displacement²::A
    effvalue::A
    prior::P
end

"""
    MHTrackChunk{T<:AbstractFloat, A<:AbstractArray{T}, V<:AbstractVector{T}}

A struct that represents a track chunk used in the Metropolis-Hastings algorithm. Besides the sames fields in TrackChunk, 'ΣΔdisplacement²::V' is the total difference (sum over particles) between two sets of squared displacements. `scaling::V` for the scaling constant for the additive random walk. (See Pressé, Data Modeling for the Sciences, 2023, p180.) `logacceptance::V`, log acceptance ratio. 'accepted::V', whether to accept the proposals at each frame. `counter::Matrix{Int}`, a matrix recording the number of proposals and the number of acceptances.
"""
struct MHTrackChunk{T<:AbstractFloat,A<:AbstractArray{T},V<:AbstractVector{T}} <:
       AbstractTrackChunk{T}
    value::A
    active::A
    displacement²::A
    effvalue::A
    ΣΔdisplacement²::V
    scaling::V
    logacceptance::V
    accepted::V
    counter::Matrix{Int}
end

function MHTrackChunk(value, active, displacement², effvalue, scaling)
    nframes = size(value, 1)
    return MHTrackChunk(
        value,
        active,
        displacement²,
        effvalue,
        similar(value, nframes - 1),
        scaling,
        fill!(similar(value, nframes), -Inf),
        fill!(similar(value, nframes), 0.0),
        zeros(Int, 2, 2),
    )
end

setacceptance!(t::MHTrackChunk; start::Integer, step::Integer) =
    logaccept!(t.accepted, t.logacceptance, start = start, step = step)

"""
    seteffvalue!(tracks::AbstractTrackChunk{T})

Set the effective value of the track chunk. The effective value is calculated as `value / active`. Note that `effvalue == value` when `active` is 1, and `effvalue == Inf` (a particle is infinitely far away so cannot contribute any photons) when `active` is 0. Currently, `active` should ONLY be used in parametric runs.
"""
function seteffvalue!(tracks::AbstractTrackChunk{T}) where {T}
    @. tracks.effvalue = tracks.value / tracks.active
    return tracks
end

"""
    Tracks{T<:AbstractFloat, A<:AbstractArray{T,3}, NT<:NEmitters{T}, TR<:TrackChunk{T}, MH<:MHTrackChunk{T}}

A mutable struct that encapsulates the number of emitting particles, track chunks, and full values. Note that the `value` of a `TrackChunk` should be a pointer to the `value` of `Tracks`, and the same for `active`, `displacement²`, and `effvalue`. The `onchunk` and `offchunk` fields are used to store the track chunks for the on and off parts of the tracks, respectively. The `proposals` field is used to store the proposals for the Metropolis-Hastings algorithm.
"""
mutable struct Tracks{
    T<:AbstractFloat,
    A<:AbstractArray{T,3},
    NT<:EmitterCount{T},
    TR<:TrackChunk{T},
    MH<:MHTrackChunk{T},
}
    value::NTuple{2,A}
    active::NTuple{2,A}
    displacement²::NTuple{2,A}
    effvalue::NTuple{2,A}
    nemitters::NT
    onchunk::TR
    offchunk::TR
    proposals::MH
end

"""
    trackchunks(value::AbstractArray{T,3}, presence::AbstractArray{T,3}, displacement²::AbstractArray{T,3}, effvalue::AbstractArray{T,3}, nemitters::Integer, prior::P)

Construct two `TrackChunk` objects from the provided arrays. The first `TrackChunk` contains the first `nemitters` pages of the input arrays for the on (emitting) particles, while the second `TrackChunk` contains the remaining pages.
"""
trackchunks(
    value::AbstractArray{T,3},
    presence::AbstractArray{T,3},
    displacement²::AbstractArray{T,3},
    effvalue::AbstractArray{T,3},
    nemitters::Integer,
    prior::P,
) where {T,P} = @views TrackChunk(
    value[:, :, 1:nemitters],
    presence[:, :, 1:nemitters],
    displacement²[:, :, 1:nemitters],
    effvalue[:, :, 1:nemitters],
    prior,
),
TrackChunk(
    value[:, :, nemitters+1:end],
    presence[:, :, nemitters+1:end],
    displacement²[:, :, nemitters+1:end],
    effvalue[:, :, nemitters+1:end],
    prior,
)

function logprior(t::AbstractTrackChunk{T}, msdᵥ::T) where {T}
    setdisplacement²!(t)
    # diff²!(t.displacement², t.value)
    return -(log(msdᵥ) * length(t.displacement²) + sum(vec(t.displacement²)) / msdᵥ) / 2 -
           logprior(t.prior, view(t.value, 1, :, :))
end

function Tracks{T}(;
    guess::AbstractArray{<:Real,3},
    prior::P,
    max_ntracks::Integer,
    scaling::AbstractVector{<:Real},
    logonprob::Real,
    active::Union{Nothing,AbstractArray{<:Real,3}} = nothing,
) where {T,P}
    guess = elconvert(T, guess)
    nframes, ndims, nguesses = size(guess)
    value = copyto!(similar(guess, nframes, ndims, max_ntracks), guess)
    values = (value, similar(value))

    active = if !isnothing(active)
        dimsmatch(guess, active, dims = 1) || throw(
            DimensionMismatch(
                "size of guess dose not match size of presence in the first dimension",
            ),
        )
        _resize3(active, max_ntracks)
    else
        fill!(similar(value, nframes, ndims, max_ntracks), true)
    end
    actives = (active, fill!(similar(active), true))
    Δx² = similar(value, nframes - 1, ndims, max_ntracks)
    Δx²s = (Δx², similar(Δx²))
    effs = (similar(value), similar(value))
    count = EmitterCount{T}(nguesses, max_ntracks, logonprob)
    @views proposals = MHTrackChunk(
        values[2][:, :, 1:nguesses],
        actives[2][:, :, 1:nguesses],
        Δx²s[2][:, :, 1:nguesses],
        effs[2][:, :, 1:nguesses],
        scaling,
    )
    return Tracks(
        values,
        actives,
        Δx²s,
        effs,
        count,
        trackchunks(value, actives[1], Δx², effs[1], nguesses, prior)...,
        proposals,
    )
end

Base.any(tracks::Tracks) = any(tracks.nemitters)

function setdisplacement²!(tracks::Tracks, i::Integer = 1)
    diff²!(tracks.displacement²[i], tracks.value[i])
    return tracks
end

function seteffvalue!(tracks::Tracks, i::Integer = 1)
    @. tracks.effvalue[i] = tracks.value[i] / tracks.active[i]
    return tracks
end

function reassign_track_parts!(tracks::Tracks{T}) where {T}
    tracks.onchunk, tracks.offchunk = trackchunks(
        tracks.value[1],
        tracks.active[1],
        tracks.displacement²[1],
        tracks.effvalue[1],
        tracks.nemitters.value,
        tracks.onchunk.prior,
    )
    return tracks
end

function reassign_proposals!(tracks::Tracks{T}) where {T}
    tracks.proposals = @views MHTrackChunk(
        tracks.value[2][:, :, 1:tracks.nemitters.value],
        tracks.active[2][:, :, 1:tracks.nemitters.value],
        tracks.displacement²[2][:, :, 1:tracks.nemitters.value],
        tracks.effvalue[2][:, :, 1:tracks.nemitters.value],
        tracks.proposals.ΣΔdisplacement²,
        tracks.proposals.scaling,
        tracks.proposals.logacceptance,
        tracks.proposals.accepted,
        tracks.proposals.counter,
    )
    return tracks
end

function reassign!(tracks::Tracks)
    reassign_track_parts!(tracks)
    reassign_proposals!(tracks)
    return tracks
end

function simulate!(x::AbstractArray{T,3}, msd::T) where {T}
    @views _randn!(x[2:end, :, :], √msd)
    cumsum!(x, x, dims = 1)
end

function simulate!(x::AbstractArray{T,3}, msd::T, prior) where {T}
    @views rand!(x[1, :, :], prior)
    simulate!(x, msd)
end

# function simulate!(
#     x::AbstractArray{T,3},
#     μ::AbstractVector{T},
#     σ::AbstractVector{T},
#     msd::T,
# ) where {T}
#     x[1, :, :] .+= reshape(μ, 1, size(x, 2), :)
#     simulate!(x, msd)
# end

# function simulate!(
#     x::AbstractArray{T,3},
#     μ::AbstractArray{T,3},
#     σ::AbstractVector{T},
#     msd::T,
# ) where {T}
#     x[1, :, :] .= μ
#     simulate!(x, msd)
# end

function simulate!(tracks::TrackChunk{T}, msdᵥ::T) where {T}
    simulate!(tracks.value, msdᵥ, tracks.prior)
    return tracks
end

function bridge!(x::AbstractArray{T,3}, msd::T, xend::AbstractArray{T,3}) where {T}
    # σ = fill!(similar(x, size(x, 2)), 0)
    @views copyto!(x[1, :, :], -diff(xend, dims = 1)[1, :, :])
    simulate!(x, msd)
    N = size(x, 1) - 1
    @views @. x = x - (0:N) / N * x[end:end, :, :] + xend[2:2, :, :]
end

function initmh!(tracks::MHTrackChunk)
    neglogrand!(tracks.logacceptance)
    fill!(tracks.accepted, false)
    return tracks
end

function propose!(
    y::AbstractArray{T,3},
    x::AbstractArray{T,3},
    σ::AbstractVector{T},
) where {T}
    randn!(y)
    y .= x .+ transpose(σ) .* y
end

function propose!(proposals::MHTrackChunk{T}, tracks::TrackChunk{T}) where {T}
    propose!(proposals.value, tracks.value, proposals.scaling)
    return proposals
end

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

function addΔlogπ₁!(tracksₚ::MHTrackChunk{T,A}, tracksₒ::TrackChunk{T,A}) where {T,A}
    addΔlogπ₁!(tracksₚ.logacceptance, tracksₒ.value, tracksₚ.value, tracksₒ.prior)
    return tracksₚ
end

function staggered_diff²!(
    Δx²::AbstractArray{T,3};
    even::AbstractArray{T,3},
    odd::AbstractArray{T,3},
) where {T}
    @views begin
        Δx²[1:2:end, :, :] .= (even[2:2:end, :, :] .- odd[1:2:end-1, :, :]) .^ 2
        Δx²[2:2:end, :, :] .= (odd[3:2:end, :, :] .- even[2:2:end-1, :, :]) .^ 2
    end
    return Δx²
end

function countacceptance!(tracks::MHTrackChunk)
    @views tracks.counter[:, 2] .+= count(>(0), tracks.accepted), length(tracks.accepted)
    return tracks
end

function boolcopyto!(
    dest::AbstractArray{T,3},
    src::AbstractArray{T,3},
    i::AbstractVector{T},
) where {T}
    @. dest .+= i .* (src .- dest)
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

function sumΔdisplacement²!(
    tracksₚ::MHTrackChunk{T},
    tracksₒ::TrackChunk{T},
    msdᵥ::T,
) where {T}
    tracksₚ.displacement² .-= tracksₒ.displacement²
    sum!(tracksₚ.ΣΔdisplacement², tracksₚ.displacement²)
    tracksₚ.ΣΔdisplacement² ./= -2 * msdᵥ
    return tracksₚ
end

function Δlogπ!(
    Δlogπ::AbstractVector{T},
    tracksₒ::TrackChunk{T},
    tracksₚ::MHTrackChunk{T},
    msdᵥ::T,
    i::Integer,
) where {T}
    setdisplacement²!(tracksₒ)
    if i == 1
        staggered_diff²!(tracksₚ.displacement², even = tracksₒ.value, odd = tracksₚ.value)
    else
        staggered_diff²!(tracksₚ.displacement², even = tracksₚ.value, odd = tracksₒ.value)
    end
    sumΔdisplacement²!(tracksₚ, tracksₒ, msdᵥ)
    nsteps = length(tracksₚ.ΣΔdisplacement²)
    Δlogπ!(Δlogπ, i:2:nsteps, mod1(i + 1, 2):2:nsteps, tracksₚ.ΣΔdisplacement²)
end

function update!(
    tracksₒ::TrackChunk{T},
    tracksₚ::MHTrackChunk{T},
    msdᵥ::T,
    Δlogℒ::AbstractVector{T},
    i::Integer,
) where {T}
    Δlogπ!(Δlogℒ, tracksₒ, tracksₚ, msdᵥ, i)
    @views tracksₚ.logacceptance[i:2:end] .+= Δlogℒ[i:2:end]
    setacceptance!(tracksₚ, start = i, step = 2)
    boolcopyto!(tracksₒ.value, tracksₚ.value, tracksₚ.accepted)
end

function update_onchunk!(
    tracks::Tracks{T},
    msdᵥ::T,
    brightnessᵥ::T,
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    tracksₒ = tracks.onchunk
    tracksₚ = tracks.proposals
    initmh!(tracksₚ)
    propose!(tracksₚ, tracksₒ)
    seteffvalue!(tracksₒ)
    seteffvalue!(tracksₚ)
    set_poisson_means!(
        llarray,
        detector,
        tracksₒ.effvalue,
        tracksₚ.effvalue,
        brightnessᵥ,
        psf,
    )
    set_frame_Δloglikelihood!(llarray, detector)
    tracksₚ.logacceptance .+= anneal!(llarray.frame, 𝑇)
    addΔlogπ₁!(tracksₚ, tracksₒ)
    update!(tracksₒ, tracksₚ, msdᵥ, llarray.frame, 1)
    update!(tracksₒ, tracksₚ, msdᵥ, llarray.frame, 2)
    countacceptance!(tracksₚ)
    return tracksₒ
end
