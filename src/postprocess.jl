"""
    sampledtracks(chain::Chain; burn_in::Integer = 0)

The tracks of every emitter of every sample after the first `burn_in`, in order, as one
`(nframes, ndims, nemitters)` array, whatever the number of emitters in each sample.
"""
function sampledtracks(chain::Chain{T}; burn_in::Integer = 0) where {T<:Real}
    N = sum(chain.emittercounts[burn_in+1:end])
    t = chain.samples[1].tracks
    x = Array{T}(undef, size(t, 1), size(t, 2), N)
    i = 1
    for s in chain.samples[burn_in+1:end]
        n = size(s.tracks, 3)
        copyto!(view(x, :, :, i:i+n-1), s.tracks)
        i += n
    end
    return x
end

"""
    localization_error(chain::Chain; burn_in::Integer = 0)

Localization uncertainty of a single emitter: the posterior standard deviation of its
position per coordinate, as the root mean square over frames and coordinates,
`sqrt(mean(var))`, the variance taken over the samples after the first `burn_in`. This is
the σ in the offset `2σ²` of the per-coordinate mean squared displacement curve. It
measures the spread of the posterior, not the distance to a true position. Every sample
after `burn_in` must have exactly one emitter.
"""
function localization_error(chain::Chain; burn_in::Integer = 0)
    counts = chain.emittercounts[burn_in+1:end]
    !isempty(counts) && all(==(1), counts) || throw(
        ArgumentError(
            "localization_error needs exactly one emitter in every sample after burn_in; " *
            "the emitter counts there are $(sort(unique(counts))).",
        ),
    )
    x = sampledtracks(chain; burn_in = burn_in)
    return sqrt(mean(var(x; dims = 3, corrected = false)))
end

"""
    credible1D(samples, [i], xedges, yedges; factor = 1, xshift = 0, yshift = 0)

The marginal posterior of the positions, frame by frame: the fraction of the sampled
positions in each bin of `xedges` (x coordinates) and of `yedges` (y coordinates), as two
`nframes × nbins` matrices. With `i`, only emitter `i` of each sample; without, the
emitters of all samples pooled. Positions are shifted, then scaled, before binning:
`(x + xshift)·factor`. The edges are equally spaced, and each bin holds the positions in
`(lower, upper]`.
"""
credible1D(samples::AbstractVector{<:Sample}, i::Integer, xedges::AbstractRange,
    yedges::AbstractRange; kwargs...) = _credible1D(samples, i:i, xedges, yedges; kwargs...)
credible1D(samples::AbstractVector{<:Sample}, xedges::AbstractRange, yedges::AbstractRange;
    kwargs...) = _credible1D(samples, :, xedges, yedges; kwargs...)

"""
    credible2D(samples, [i], xedges, yedges; factor = 1, xshift = 0, yshift = 0)

As `credible1D`, the joint posterior of each frame's position: the fraction of the sampled
positions in each bin of `xedges` × `yedges`, as an `nxbins × nybins × nframes` array.
"""
credible2D(samples::AbstractVector{<:Sample}, i::Integer, xedges::AbstractRange,
    yedges::AbstractRange; kwargs...) = _credible2D(samples, i:i, xedges, yedges; kwargs...)
credible2D(samples::AbstractVector{<:Sample}, xedges::AbstractRange, yedges::AbstractRange;
    kwargs...) = _credible2D(samples, :, xedges, yedges; kwargs...)

function _credible1D(samples, emitters, xedges, yedges; factor = 1, xshift = 0, yshift = 0)
    nframes = size(samples[1].tracks, 1)
    xcounts = zeros(Float64, nframes, length(xedges) - 1)
    ycounts = zeros(Float64, nframes, length(yedges) - 1)
    npositions = 0 # per frame
    for s in samples
        x = view(s.tracks, :, :, emitters)
        npositions += size(x, 3)
        for k in axes(x, 3), n in 1:nframes
            i = binindex((x[n, 1, k] + xshift) * factor, xedges)
            i > 0 && (xcounts[n, i] += 1)
            j = binindex((x[n, 2, k] + yshift) * factor, yedges)
            j > 0 && (ycounts[n, j] += 1)
        end
    end
    xcounts ./= npositions
    ycounts ./= npositions
    return xcounts, ycounts
end

function _credible2D(samples, emitters, xedges, yedges; factor = 1, xshift = 0, yshift = 0)
    nframes = size(samples[1].tracks, 1)
    counts = zeros(Float64, length(xedges) - 1, length(yedges) - 1, nframes)
    npositions = 0 # per frame
    for s in samples
        x = view(s.tracks, :, :, emitters)
        npositions += size(x, 3)
        for k in axes(x, 3), n in 1:nframes
            i = binindex((x[n, 1, k] + xshift) * factor, xedges)
            j = binindex((x[n, 2, k] + yshift) * factor, yedges)
            i > 0 && j > 0 && (counts[i, j, n] += 1)
        end
    end
    return counts ./= npositions
end

# The bin of x among the equal bins of edges, each holding (lower, upper], or 0 outside them.
function binindex(x, edges::AbstractRange)
    nbins = length(edges) - 1
    lower, upper = extrema(edges)
    b = (x - lower) * (nbins / (upper - lower))
    return 0 < b <= nbins ? ceil(Int, b) : 0
end
