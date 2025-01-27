Base.length(tracks::AbstractTrackParts) = size(tracks.value, 1)

function setdisplacement²!(tracks::AbstractTrackParts{T}) where {T}
    diff²!(tracks.displacement², tracks.value)
    return tracks
end

function seteffvalue!(tracks::AbstractTrackParts{T}) where {T}
    @. tracks.effvalue = tracks.value / tracks.presence
    return tracks
end

function TrackParts(part::TrackParts{T}, tracks::Tracks{T}; ison::Bool = true) where {T}
    M = tracks.ntracks.value
    @views if ison
        return TrackParts(
            tracks.values[1][:, :, 1:M],
            tracks.presences[1][:, :, 1:M],
            tracks.displacement²s[1][:, :, 1:M],
            tracks.effvalues[1][:, :, 1:M],
            part.prior,
        )
    else
        return TrackParts(
            tracks.values[1][:, :, M+1:end],
            tracks.presences[1][:, :, M+1:end],
            tracks.displacement²s[1][:, :, M+1:end],
            tracks.effvalues[1][:, :, M+1:end],
            part.prior,
        )
    end
end

MHTrackParts(mhpart::MHTrackParts{T}, tracks::Tracks{T}) where {T} = @views MHTrackParts(
    tracks.values[2][:, :, 1:tracks.ntracks.value],
    tracks.presences[2][:, :, 1:tracks.ntracks.value],
    tracks.displacement²s[2][:, :, 1:tracks.ntracks.value],
    tracks.effvalues[2][:, :, 1:tracks.ntracks.value],
    mhpart.ΣΔdisplacement²,
    mhpart.perturbsize,
    mhpart.logacceptance,
    mhpart.acceptance,
    mhpart.counter,
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
) where {T}
    if eltype(guess) !== T
        guess2 = similar(guess, T)
        copyto!(guess2, guess)
        guess = guess2
    end
    nframes, ndims, nguess = size(guess)
    value = copyto!(similar(guess, nframes, ndims, max_ntracks), guess)
    value2 = similar(value)
    presence = fill!(similar(value, nframes, 1, max_ntracks), true)
    presence2 = similar(presence)
    displacement² = similar(value, nframes - 1, ndims, max_ntracks)
    displacement²2 = similar(displacement²)
    effvalue = similar(value)
    effvalue2 = similar(effvalue)
    ntracks = NTracks{T}(nguess, max_ntracks, logonprob)
    @views begin
        onpart = TrackParts(
            value[:, :, 1:nguess],
            presence[:, :, 1:nguess],
            displacement²[:, :, 1:nguess],
            effvalue[:, :, 1:nguess],
            prior,
        )
        offpart = TrackParts(
            value[:, :, nguess+1:end],
            presence[:, :, nguess+1:end],
            displacement²[:, :, nguess+1:end],
            effvalue[:, :, nguess+1:end],
            prior,
        )
        proposals = MHTrackParts(
            value2[:, :, 1:nguess],
            presence2[:, :, 1:nguess],
            displacement²2[:, :, 1:nguess],
            effvalue2[:, :, 1:nguess],
            similar(value, nframes - 1),
            perturbsize,
            similar(value, nframes),
            similar(value, nframes),
            zeros(Int, 2, 2),
        )
    end
    return Tracks(
        (value, value2),
        (presence, presence2),
        (displacement², displacement²2),
        (effvalue, effvalue2),
        ntracks,
        onpart,
        offpart,
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

function reassign!(tracks::Tracks)
    tracks.onpart = TrackParts(tracks.onpart, tracks)
    tracks.offpart = TrackParts(tracks.offpart, tracks, ison = false)
    tracks.proposals = MHTrackParts(tracks.proposals, tracks)
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

function addΔlogπ₁!(
    tracksₚ::MHTrackParts{T,A},
    tracksₒ::TrackParts{T,A},
) where {T,A<:AbstractArray}
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
    set_poisson_mean!(
        llarray,
        detector,
        tracksₒ.effvalue,
        tracksₚ.effvalue,
        brightnessᵥ,
        psf,
    )
    set_Δloglikelihood!(llarray, detector)
    tracksₚ.logacceptance .+= anneal!(llarray.frame, 𝑇)
    addΔlogπ₁!(tracksₚ, tracksₒ)
    update!(tracksₒ, tracksₚ, msdᵥ, llarray.frame, 1)
    update!(tracksₒ, tracksₚ, msdᵥ, llarray.frame, 2)
    countacceptance!(tracksₚ)
    return tracksₒ
end
