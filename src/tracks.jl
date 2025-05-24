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
struct MHTrackChunk{
    T<:AbstractFloat,
    A<:AbstractArray{T,3},
    A2<:AbstractArray{T,3},
    V<:AbstractVector{T},
} <: AbstractTrackChunk{T}
    value::A
    active::A
    displacement²::A
    effvalue::A
    ΣΔdisplacement²::V
    scaling::A2
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

function initmh!(t::MHTrackChunk)
    neglogrand!(t.logacceptance)
    fill!(t.accepted, false)
    return t
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
    chunks(value::AbstractArray{T,3}, presence::AbstractArray{T,3}, displacement²::AbstractArray{T,3}, effvalue::AbstractArray{T,3}, nemitters::Integer, prior::P)

Construct two `TrackChunk` objects from the provided arrays. The first `TrackChunk` contains the first `nemitters` pages of the input arrays for the on (emitting) particles, while the second `TrackChunk` contains the remaining pages.
"""
chunks(
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
    return -(log(msdᵥ) * length(t.displacement²) + sum(vec(t.displacement²)) / msdᵥ) / 2 -
           logprior(t.prior, view(t.value, 1, :, :))
end

function Tracks{T}(;
    guess::AbstractArray{<:Real,3},
    prior::P,
    max_ntracks::Integer,
    scaling::Union{<:Real,AbstractArray{<:Real,3}},
    logonprob::Real,
    active::Union{Nothing,AbstractArray{<:Real,3}} = nothing,
) where {T,P}
    guess = elconvert(T, guess)
    nframes, ndims, nguesses = size(guess)
    value = copyto!(similar(guess, nframes, ndims, max_ntracks), guess)
    values = (value, similar(value))

    scaling isa Real && (scaling = fill!(similar(value), scaling))

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
        chunks(value, actives[1], Δx², effs[1], nguesses, prior)...,
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

"""
    reassign!(t::Tracks{T})

Reassigns the `TrackChunk`s in `t` according the current number of emitting particles.
"""
function reassign!(t::Tracks)
    t.onchunk, t.offchunk = chunks(
        t.value[1],
        t.active[1],
        t.displacement²[1],
        t.effvalue[1],
        t.nemitters.value,
        t.onchunk.prior,
    )
    t.proposals = @views MHTrackChunk(
        t.value[2][:, :, 1:t.nemitters.value],
        t.active[2][:, :, 1:t.nemitters.value],
        t.displacement²[2][:, :, 1:t.nemitters.value],
        t.effvalue[2][:, :, 1:t.nemitters.value],
        t.proposals.ΣΔdisplacement²,
        t.proposals.scaling,
        t.proposals.logacceptance,
        t.proposals.accepted,
        t.proposals.counter,
    )
    return t
end

"""
    simulate!(x::AbstractArray{T,3}, msd::T)

Simulates particle tracks in `x` given the `msd`. `x` should already contains the initial particle positions.
"""
function simulate!(x::AbstractArray{T,3}, msd::T) where {T}
    @views _randn!(x[2:end, :, :], √msd)
    cumsum!(x, x, dims = 1)
end

"""
    simulate!(x::AbstractArray{T,3}, msd::T, prior)

Simulates particle tracks in `x` given the `msd` and the `prior` for the initial positions.
"""
function simulate!(x::AbstractArray{T,3}, msd::T, prior) where {T}
    @views rand!(x[1, :, :], prior)
    simulate!(x, msd)
end

function simulate!(tracks::TrackChunk{T}, msdᵥ::T) where {T}
    simulate!(tracks.value, msdᵥ, tracks.prior)
    return tracks
end

function bridge!(x::AbstractArray{T,3}, msd::T, xend::AbstractArray{T,3}) where {T}
    @views copyto!(x[1, :, :], -diff(xend, dims = 1)[1, :, :])
    simulate!(x, msd)
    N = size(x, 1) - 1
    @views @. x = x - (0:N) / N * x[end:end, :, :] + xend[2:2, :, :]
end

function propose!(proposals::MHTrackChunk{T}, tracks::TrackChunk{T}) where {T}
    arw_propose!(
        proposals.value,
        tracks.value,
        view(proposals.scaling, :, :, 1:size(tracks.value, 3)),
    )
    return proposals
end

Δlogprior₁(
    x₁::AbstractMatrix{T},
    y₁::AbstractMatrix{T},
    μ::AbstractVector{T},
    σ::AbstractVector{T},
) where {T} = sum(vec(@. ((x₁ - μ)^2 - (y₁ - μ)^2) / (2 * σ^2)))

Δlogprior₁(x::AbstractArray{T,3}, y::AbstractArray{T,3}, prior::DNormal{T}) where {T} =
    @views Δlogprior₁(x[1, :, :], y[1, :, :], prior.μ, prior.σ)

function addΔlogprior₁!(
    logacceptance::AbstractVector{T},
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    prior,
) where {T}
    logacceptance[1] += Δlogprior₁(x, y, prior)
    return logacceptance
end

function addΔlogprior₁!(tracksᵖ::MHTrackChunk{T,A}, tracksᵒ::TrackChunk{T,A}) where {T,A}
    addΔlogprior₁!(tracksᵖ.logacceptance, tracksᵒ.value, tracksᵖ.value, tracksᵒ.prior)
    return tracksᵖ
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

function countacceptance!(t::MHTrackChunk)
    @views t.counter[:, 2] .+= count(>(0), t.accepted), length(t)
    return t
end

function Δlogmotion!(
    Δ::AbstractVector{T},
    idx1::StepRange,
    idx2::StepRange,
    ΣΔΔx²::AbstractVector{T},
) where {T}
    @views copyto!(Δ[idx1], ΣΔΔx²[idx1])
    @views Δ[idx2.+1] .+= ΣΔΔx²[idx2]
    return Δ
end

function sumΔdisplacement²!(
    tracksᵒ::MHTrackChunk{T},
    tracksᵖ::TrackChunk{T},
    msdᵥ::T,
) where {T}
    tracksᵒ.displacement² .-= tracksᵖ.displacement²
    sum!(tracksᵒ.ΣΔdisplacement², tracksᵒ.displacement²)
    tracksᵒ.ΣΔdisplacement² ./= -2 * msdᵥ
    return tracksᵒ
end

function Δlogmotion!(
    Δ::AbstractVector{T},
    tracksᵒ::TrackChunk{T},
    tracksᵖ::MHTrackChunk{T},
    msdᵥ::T,
    i::Integer,
) where {T}
    setdisplacement²!(tracksᵒ)
    if i == 1
        staggered_diff²!(tracksᵖ.displacement², even = tracksᵒ.value, odd = tracksᵖ.value)
    else
        staggered_diff²!(tracksᵖ.displacement², even = tracksᵖ.value, odd = tracksᵒ.value)
    end
    sumΔdisplacement²!(tracksᵖ, tracksᵒ, msdᵥ)
    nsteps = length(tracksᵖ) - 1
    Δlogmotion!(Δ, i:2:nsteps, mod1(i + 1, 2):2:nsteps, tracksᵖ.ΣΔdisplacement²)
end

function update!(
    tracksᵒ::TrackChunk{T},
    tracksᵖ::MHTrackChunk{T},
    msdᵥ::T,
    Δloglikelihood::AbstractVector{T},
    i::Integer,
) where {T}
    Δlogmotion!(Δloglikelihood, tracksᵒ, tracksᵖ, msdᵥ, i) # add motion model contribution to Δloglikelihood
    @views tracksᵖ.logacceptance[i:2:end] .+= Δloglikelihood[i:2:end] # add Δlogposterior to log acceptance ratio
    setacceptance!(tracksᵖ, start = i, step = 2) # set acceptance flag for given frame indices
    boolcopyto!(tracksᵒ.value, tracksᵖ.value, tracksᵖ.accepted) # copy accepted values to the chunk
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
    tracksᵒ, tracksᵖ = tracks.onchunk, tracks.proposals # rename for clarity
    initmh!(tracksᵖ)
    propose!(tracksᵖ, tracksᵒ)
    seteffvalue!(tracksᵒ)
    seteffvalue!(tracksᵖ)
    set_poisson_means!(
        llarray,
        detector,
        tracksᵒ.effvalue,
        tracksᵖ.effvalue,
        brightnessᵥ,
        psf,
    )
    set_frame_Δloglikelihood!(llarray, detector)
    tracksᵖ.logacceptance .+= anneal!(llarray.frame, 𝑇)
    addΔlogprior₁!(tracksᵖ, tracksᵒ)
    update!(tracksᵒ, tracksᵖ, msdᵥ, llarray.frame, 1) # update particle positions at odd frame indices
    update!(tracksᵒ, tracksᵖ, msdᵥ, llarray.frame, 2) # update particle positions at even frame indices
    countacceptance!(tracksᵖ)
    return tracksᵒ
end
