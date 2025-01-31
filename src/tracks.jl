Base.length(tracks::AbstractTrackParts) = size(tracks.value, 1)

function setdisplacement²!(tracks::AbstractTrackParts{T}) where {T}
    diff²!(tracks.displacement², tracks.value)
    return tracks
end

function seteffvalue!(tracks::AbstractTrackParts{T}) where {T}
    @. tracks.effvalue = tracks.value / tracks.presence
    return tracks
end

get_track_parts(
    value::AbstractArray{T,3},
    presence::AbstractArray{T,3},
    displacement²::AbstractArray{T,3},
    effvalue::AbstractArray{T,3},
    ntracks::Integer,
    onprior,
    offprior,
) where {T} = @views TrackParts(
    value[:, :, 1:ntracks],
    presence[:, :, 1:ntracks],
    displacement²[:, :, 1:ntracks],
    effvalue[:, :, 1:ntracks],
    onprior,
),
TrackParts(
    value[:, :, ntracks+1:end],
    presence[:, :, ntracks+1:end],
    displacement²[:, :, ntracks+1:end],
    effvalue[:, :, ntracks+1:end],
    offprior,
)

setacceptance!(tracks::MHTrackParts; start::Integer, step::Integer) =
    logaccept!(tracks.acceptance, tracks.logacceptance, start = start, step = step)

function logprior(tracks::TrackParts{T}, msdᵥ::T) where {T}
    diff²!(tracks.displacement², tracks.value)
    return -(
        log(msdᵥ) * length(tracks.displacement²) + sum(vec(tracks.displacement²)) / msdᵥ
    ) / 2 - _logπ(tracks.prior, view(tracks.value, 1, :, :))
end

function Tracks{T}(;
    guess::AbstractArray{<:Real,3},
    prior,
    max_ntracks::Integer,
    perturbsize::AbstractVector{<:Real},
    logonprob::Real,
    presence::Union{Nothing,AbstractArray{<:Real,3}} = nothing,
) where {T}
    guess = elconvert(T, guess)
    nframes, ndims, nguesses = size(guess)
    value = copyto!(similar(guess, nframes, ndims, max_ntracks), guess)
    value2 = similar(value)

    fullpresence = fill!(similar(value, nframes, 1, max_ntracks), true)
    if !isnothing(presence)
        dimsmatch(guess, presence, dims = 1) || throw(
            DimensionMismatch(
                "size of guess dose not match size of presence in the first dimension",
            ),
        )
        copyto!(fullpresence, presence)
    end
    fullpresence2 = fill!(similar(fullpresence), true)
    displacement² = similar(value, nframes - 1, ndims, max_ntracks)
    displacement²2 = similar(displacement²)
    effvalue, effvalue2 = similar(value), similar(value)
    ntracks = NTracks{T}(nguesses, max_ntracks, logonprob)
    @views proposals = MHTrackParts(
        value2[:, :, 1:nguesses],
        fullpresence2[:, :, 1:nguesses],
        displacement²2[:, :, 1:nguesses],
        effvalue2[:, :, 1:nguesses],
        similar(value, nframes - 1),
        perturbsize,
        similar(value, nframes),
        similar(value, nframes),
        zeros(Int, 2, 2),
    )
    return Tracks(
        (value, value2),
        (fullpresence, fullpresence2),
        (displacement², displacement²2),
        (effvalue, effvalue2),
        ntracks,
        get_track_parts(
            value,
            fullpresence,
            displacement²,
            effvalue,
            nguesses,
            prior,
            prior,
        )...,
        proposals,
    )
end

Base.any(tracks::Tracks) = any(tracks.ntracks)

function setdisplacement²!(tracks::Tracks, i::Integer = 1)
    diff²!(tracks.displacement²s[i], tracks.values[i])
    return tracks
end

function seteffvalue!(tracks::Tracks, i::Integer = 1)
    @. tracks.effvalues[i] = tracks.values[i] / tracks.presences[i]
    return tracks
end

function reassign_track_parts!(tracks::Tracks{T}) where {T}
    tracks.onpart, tracks.offpart = get_track_parts(
        tracks.values[1],
        tracks.presences[1],
        tracks.displacement²s[1],
        tracks.effvalues[1],
        tracks.ntracks.value,
        tracks.onpart.prior,
        tracks.offpart.prior,
    )
    return tracks
end

function reassign_proposals!(tracks::Tracks{T}) where {T}
    tracks.proposals = @views MHTrackParts(
        tracks.values[2][:, :, 1:tracks.ntracks.value],
        tracks.presences[2][:, :, 1:tracks.ntracks.value],
        tracks.displacement²s[2][:, :, 1:tracks.ntracks.value],
        tracks.effvalues[2][:, :, 1:tracks.ntracks.value],
        tracks.proposals.ΣΔdisplacement²,
        tracks.proposals.perturbsize,
        tracks.proposals.logacceptance,
        tracks.proposals.acceptance,
        tracks.proposals.counter,
    )
    return tracks
end

function reassign!(tracks::Tracks)
    reassign_track_parts!(tracks)
    reassign_proposals!(tracks)
    return tracks
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

function simulate!(tracks::TrackParts{T}, msdᵥ::T) where {T}
    simulate!(tracks.value, params(tracks.prior)..., msdᵥ)
    return tracks
end

function bridge!(x::AbstractArray{T,3}, msd::T, xend::AbstractArray{T,3}) where {T}
    σ = fill!(similar(x, size(x, 2)), 0)
    simulate!(x, -diff(xend, dims = 1), σ, msd)
    N = size(x, 1) - 1
    @views @. x = x - (0:N) / N * x[end:end, :, :] + xend[2:2, :, :]
end

function initmh!(tracks::MHTrackParts)
    neglogrand!(tracks.logacceptance)
    fill!(tracks.acceptance, false)
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

function propose!(proposals::MHTrackParts{T}, tracks::TrackParts{T}) where {T}
    propose!(proposals.value, tracks.value, proposals.perturbsize)
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

function addΔlogπ₁!(tracksₚ::MHTrackParts{T,A}, tracksₒ::TrackParts{T,A}) where {T,A}
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

function countacceptance!(tracks::MHTrackParts)
    @views tracks.counter[:, 2] .+=
        count(>(0), tracks.acceptance), length(tracks.acceptance)
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
    tracksₚ::MHTrackParts{T},
    tracksₒ::TrackParts{T},
    msdᵥ::T,
) where {T}
    tracksₚ.displacement² .-= tracksₒ.displacement²
    sum!(tracksₚ.ΣΔdisplacement², tracksₚ.displacement²)
    tracksₚ.ΣΔdisplacement² ./= -2 * msdᵥ
    return tracksₚ
end

function Δlogπ!(
    Δlogπ::AbstractVector{T},
    tracksₒ::TrackParts{T},
    tracksₚ::MHTrackParts{T},
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
    tracksₒ::TrackParts{T},
    tracksₚ::MHTrackParts{T},
    msdᵥ::T,
    Δlogℒ::AbstractVector{T},
    i::Integer,
) where {T}
    Δlogπ!(Δlogℒ, tracksₒ, tracksₚ, msdᵥ, i)
    @views tracksₚ.logacceptance[i:2:end] .+= Δlogℒ[i:2:end]
    setacceptance!(tracksₚ, start = i, step = 2)
    boolcopyto!(tracksₒ.value, tracksₚ.value, tracksₚ.acceptance)
end

function update_onpart!(
    tracks::Tracks{T},
    msdᵥ::T,
    brightnessᵥ::T,
    llarray::LogLikelihoodArray{T},
    detector::PixelDetector{T},
    psf::PointSpreadFunction{T},
    𝑇::T,
) where {T}
    tracksₒ = tracks.onpart
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
