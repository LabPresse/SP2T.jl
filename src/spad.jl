struct SPAD{
    T<:AbstractFloat,
    V<:AbstractVector{T},
    M<:AbstractMatrix{T},
    R<:AbstractArray{UInt16,3},
} <: PixelDetector{T}
    period::T
    pxsize::T
    pxbounds::NTuple{2,V}
    darkcounts::M
    filter::M
    readouts::R
    batchsize::UInt16
end

function SPAD{T}(;
    period::Real,
    pixel_size::Real,
    darkcounts::AbstractMatrix{<:Real},
    cutoffs::Tuple{<:Real,<:Real},
    readouts::AbstractArray{UInt16,3},
    batchsize::Integer = one(UInt16),
) where {T<:AbstractFloat}
    check_pixel_dimsmatch(darkcounts, readouts)
    pxbounds, darkcounts, filter =
        _init_pixel_detector_params(T, pixel_size, darkcounts, cutoffs)
    return SPAD{T,typeof(pxbounds[1]),typeof(darkcounts),typeof(readouts)}(
        period,
        pixel_size,
        pxbounds,
        darkcounts,
        filter,
        readouts,
        batchsize,
    )
end

# log(1-exp(-λ)): use expm1 at small λ and preserve the exponentially small
# negative tail at saturation. Never obtain that tail by subtracting λ from λ.
_spad_log_hit(λ::T) where {T} =
    λ <= log(T(2)) ? log(-expm1(-λ)) : log1p(-exp(-λ))

function _spad_Δlog_hit(old::T, new::T) where {T}
    old == new && old > zero(T) && return zero(T)
    lo, hi = min(old, new), max(old, new)
    lo <= zero(T) && return _spad_log_hit(new) - _spad_log_hit(old)
    # p(hi)-p(lo) = exp(-lo) * (1-exp(-(hi-lo))). This also resolves
    # close proposals for which subtraction of two log-hit scores loses bits.
    ratio = exp(-lo) * (-expm1(-(hi - lo))) / (-expm1(-lo))
    d = isinf(ratio) ? _spad_log_hit(hi) - _spad_log_hit(lo) : log1p(ratio)
    return new > old ? d : -d
end

_spad_included_full(f::T, r, B, λ::T) where {T} =
    iszero(f) ? zero(T) : _spad_loglikelihood(r, B, λ)
_spad_included_delta(f::T, r, B, old::T, new::T) where {T} =
    iszero(f) ? zero(T) : _spad_Δloglikelihood(r, B, old, new)

function _spad_loglikelihood(r::Integer, B::Integer, λ::T) where {T}
    hits = iszero(r) ? zero(T) : T(r) * _spad_log_hit(λ)
    misses = r == B ? zero(T) : (T(B) - T(r)) * λ
    return hits - misses
end

function _spad_Δloglikelihood(r::Integer, B::Integer, old::T, new::T) where {T}
    hits = iszero(r) ? zero(T) : T(r) * _spad_Δlog_hit(old, new)
    misses = r == B ? zero(T) : (T(B) - T(r)) * (new - old)
    return hits - misses
end

function set_pixel_loglikelihood!(
    llarray::LogLikelihoodArray{T},
    detector::SPAD{T},
) where {T}
    @. llarray.pixel = _spad_included_full(
        detector.filter, detector.readouts, detector.batchsize, llarray.means[1],
    )
    return llarray
end

function set_pixel_Δloglikelihood!(
    Δllarray::LogLikelihoodArray{T},
    detector::SPAD{T},
) where {T}
    @. Δllarray.pixel = _spad_included_delta(
        detector.filter, detector.readouts, detector.batchsize,
        Δllarray.means[1], Δllarray.means[2],
    )
    return Δllarray
end

"""
    SPADLogLikelihood{T} <: LogLikelihoodAux{T}

The SPAD log-likelihood, storing photon means only at nonzero, unfiltered readouts.
At those pixels use the stable hit/miss form `r·log(1-exp(-λ)) − (B-r)·λ`.
Zero-readout pixels contribute only `−B·λ`: `darksum[k]` and `linear[i][k]`
are the nonnegative filtered background and emitter signal sums **only over those
zero-readout pixels**. This avoids cancellation between two large terms at saturated
full hits. `zeroruns` identifies frames with no such pixels, all-zero readouts, or
a mixture. The generic reduction uses bounded spatial/time tiles. CUDA sums each
frame directly in one block while updating the clicked photon means.

The clicks are padded to one column per frame, with as many rows as the busiest frame has clicks, so per-frame sums are column sums on CPUs and GPUs alike. Padding entries have readout 0, so they contribute nothing, and point at pixel (1, 1). `xindex` and `yindex` index a PSF component of size `(nx, 1, nframes)` and `(ny, 1, nframes)`. `filter` is `nothing` when no pixel is filtered out.
"""
struct SPADLogLikelihood{
    T<:AbstractFloat,
    A<:AbstractMatrix{T},
    V<:AbstractVector{T},
    I<:AbstractMatrix{Int},
    R<:AbstractMatrix{UInt16},
    F<:Union{Nothing,AbstractMatrix{T}},
} <: LogLikelihoodAux{T}
    readouts::R
    weights::A # filter at the clicks
    darkcounts::A # dark counts at the clicks
    xindex::I
    yindex::I
    filter::F
    darksum::V # filtered background mass at zero-readout pixels, per frame
    zeroruns::Vector{Tuple{UnitRange{Int},Symbol}}
    means::NTuple{2,A}
    linear::NTuple{2,V} # emitter signal mass at zero-readout pixels, per frame
    pixel::A
    frame::V
end

# Walk the included pixels once per frame, in column-major order. Only the
# spatial index of each click is needed to pack the padded workspace below.
function _spad_click_metadata(readouts, included, filter, darkcounts::Matrix{T}) where {T}
    npixels, nframes = length(filter), size(readouts, 3)
    background = zeros(Float64, npixels)
    for p in included
        background[p] = Float64(filter[p]) * Float64(darkcounts[p])
    end
    clickpixels = Int[]
    counts = Vector{Int}(undef, nframes)
    darkzero = Vector{T}(undef, nframes)
    zeroruns = Tuple{UnitRange{Int},Symbol}[]
    for k in 1:nframes
        offset = (k - 1) * npixels
        start = length(clickpixels)
        total = 0.0
        @inbounds for p in included
            if iszero(readouts[offset + p])
                total += background[p]
            else
                push!(clickpixels, p)
            end
        end
        n = length(clickpixels) - start
        counts[k], darkzero[k] = n, T(total)
        kind = n == length(included) ? :none : iszero(n) ? :all : :mixed
        if !isempty(zeroruns) && last(zeroruns)[2] == kind
            zeroruns[end] = (first(last(zeroruns)[1]):k, kind)
        else
            push!(zeroruns, (k:k, kind))
        end
    end
    return (; clickpixels, counts, darkzero, zeroruns)
end

function SPADLogLikelihood(detector::SPAD{T}) where {T}
    readouts, filter = Array(detector.readouts), Array(detector.filter)
    darkcounts = Array(detector.darkcounts)
    nx, ny, nframes = size(readouts)
    included = all(!iszero, filter) ? eachindex(filter) : findall(!iszero, vec(filter))
    metadata = _spad_click_metadata(readouts, included, filter, darkcounts)
    nrows = max(1, maximum(metadata.counts)) # GPU kernels need a nonempty row dimension
    x, y = Matrix{Int}(undef, nrows, nframes), Matrix{Int}(undef, nrows, nframes)
    r, w = zeros(UInt16, nrows, nframes), zeros(T, nrows, nframes)
    d = fill(darkcounts[1], nrows, nframes)
    coords = CartesianIndices(filter)
    firstclick = 1
    @inbounds for k in 1:nframes
        xoffset, yoffset = (k - 1) * nx, (k - 1) * ny
        for j in 1:metadata.counts[k]
            p = metadata.clickpixels[firstclick + j - 1]
            ix, iy = Tuple(coords[p])
            x[j, k], y[j, k] = xoffset + ix, yoffset + iy
            r[j, k] = readouts[(k - 1) * length(filter) + p]
            w[j, k], d[j, k] = filter[p], darkcounts[p]
        end
        for j in (metadata.counts[k] + 1):nrows
            x[j, k], y[j, k] = xoffset + 1, yoffset + 1
        end
        firstclick += metadata.counts[k]
    end
    todevice(a) = copyto!(similar(detector.darkcounts, eltype(a), size(a)), a)
    return SPADLogLikelihood(
        todevice(r),
        todevice(w),
        todevice(d),
        todevice(x),
        todevice(y),
        all(isone, filter) ? nothing : detector.filter,
        todevice(metadata.darkzero),
        metadata.zeroruns,
        (todevice(d), todevice(d)),
        (similar(detector.darkcounts, nframes), similar(detector.darkcounts, nframes)),
        todevice(w),
        similar(detector.darkcounts, nframes),
    )
end

"""
    loglikelihoodaux(detector::PixelDetector)

The log-likelihood workspace for `detector`: `SPADLogLikelihood` for a `SPAD`, `LogLikelihoodArray` otherwise.
"""
loglikelihoodaux(detector::PixelDetector{T}) where {T} =
    LogLikelihoodArray{T}(detector.readouts)
loglikelihoodaux(detector::SPAD) = SPADLogLikelihood(detector)

function reset!(llarray::SPADLogLikelihood{T}, ::SPAD{T}, i::Integer) where {T}
    llarray.means[i] .= llarray.darkcounts
    fill!(llarray.linear[i], zero(T))
    return llarray
end

_spad_zero_mass(r, f::T, x::T, y::T) where {T} =
    iszero(f) || !iszero(r) ? zero(T) : f * x * y
_spad_scale_mass(scale::T, mass::T) where {T} =
    iszero(mass) ? zero(T) : scale * mass

function add_zero_framesums!(linear, px, py, detector, llarray, scale)
    nx, ny = size(detector)
    for (frames, kind) in llarray.zeroruns
        kind == :none && continue
        if kind == :all && isnothing(llarray.filter)
            @views mass = vec(sum(px[:, :, frames], dims = 1)) .*
                          vec(sum(py[:, :, frames], dims = 1))
            @views linear[frames] .+= _spad_scale_mass.(scale, mass)
            continue
        end
        # <= 128*8*16 scalar entries per product tile, independent of movie size.
        # Unlike subtracting click signal from total signal, all terms are
        # nonnegative and absent pixels are branched away before multiplication.
        for k in first(frames):16:last(frames), j in 1:8:ny, i in 1:128:nx
            K = k:min(k + 15, last(frames)); J = j:min(j + 7, ny); I = i:min(i + 127, nx)
            @views reads = detector.readouts[I, J, K]
            @views filter = detector.filter[I, J]
            @views x = px[I, :, K]
            @views y = reshape(py[J, :, K], 1, length(J), length(K))
            tile = _spad_zero_mass.(reads, filter, x, y)
            mass = vec(sum(tile, dims = (1, 2)))
            @views linear[K] .+= _spad_scale_mass.(scale, mass)
        end
    end
    return linear
end

function add_spad_signal!(llarray, detector, px, py, scale, i)
    llarray.means[i] .+= scale .* px[llarray.xindex] .* py[llarray.yindex]
    add_zero_framesums!(llarray.linear[i], px, py, detector, llarray, scale)
    return llarray
end

function add_poisson_mean!(
    llarray::SPADLogLikelihood{T},
    detector::SPAD{T},
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    psf::PointSpreadFunction{T},
    i::Integer,
) where {T}
    iszero(brightnessᵥ) && return llarray
    nframes = size(tracksᵥ, 1)
    psfcomponents = (
        similar(tracksᵥ, size(detector, 1), 1, nframes),
        similar(tracksᵥ, size(detector, 2), 1, nframes),
    )
    relativepos = (
        similar(tracksᵥ, size(detector, 1) + 1, 1, nframes),
        similar(tracksᵥ, size(detector, 2) + 1, 1, nframes),
    )
    scale = brightnessᵥ / psf.A
    for m in axes(tracksᵥ, 3)
        px, py = integratepsf!(
            psfcomponents,
            relativepos,
            view(tracksᵥ, :, :, m:m),
            detector.pxbounds,
            psf,
        )
        add_spad_signal!(llarray, detector, px, py, scale, i)
    end
    return llarray
end

function set_poisson_means!(
    llarray::SPADLogLikelihood{T},
    detector::SPAD{T},
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ₁::T,
    brightnessᵥ₂::T,
    psf::PointSpreadFunction{T},
) where {T}
    set_poisson_mean!(llarray, detector, tracksᵥ, brightnessᵥ₁, psf, 1)
    if iszero(brightnessᵥ₁)
        return set_poisson_mean!(llarray, detector, tracksᵥ, brightnessᵥ₂, psf, 2)
    end
    iszero(brightnessᵥ₂) && return reset!(llarray, detector, 2)
    ratio = brightnessᵥ₂ / brightnessᵥ₁
    llarray.means[2] .=
        llarray.darkcounts .+ (llarray.means[1] .- llarray.darkcounts) .* ratio
    llarray.linear[2] .= _spad_scale_mass.(ratio, llarray.linear[1])
    return llarray
end

_spad_weighted_full(w::T, r, B, λ::T) where {T} =
    iszero(w) ? zero(T) : w * _spad_loglikelihood(r, B, λ)
_spad_weighted_delta(w::T, r, B, old::T, new::T) where {T} =
    iszero(w) ? zero(T) : w * _spad_Δloglikelihood(r, B, old, new)

function get_loglikelihood!(llarray::SPADLogLikelihood{T}, detector::SPAD{T}) where {T}
    @. llarray.pixel = _spad_weighted_full(llarray.weights, llarray.readouts, detector.batchsize, llarray.means[1])
    sum!(reshape(llarray.frame, 1, :), llarray.pixel)
    if !iszero(detector.batchsize)
        llarray.frame .-= detector.batchsize .* (llarray.darksum .+ llarray.linear[1])
    end
    return sum(llarray.frame)
end

function set_frame_Δloglikelihood!(
    llarray::SPADLogLikelihood{T},
    detector::SPAD{T},
) where {T}
    @. llarray.pixel = _spad_weighted_delta(
        llarray.weights, llarray.readouts, detector.batchsize, llarray.means[1], llarray.means[2],
    )
    sum!(reshape(llarray.frame, 1, :), llarray.pixel)
    if !iszero(detector.batchsize)
        llarray.frame .-= detector.batchsize .* (llarray.linear[2] .- llarray.linear[1])
    end
    return llarray
end

get_Δloglikelihood!(llarray::SPADLogLikelihood{T}, detector::SPAD{T}) where {T} =
    sum(set_frame_Δloglikelihood!(llarray, detector).frame)

function simulate_readouts!(
    rng::AbstractRNG,
    detector::SPAD{T},
    means::AbstractArray{T,3},
) where {T}
    if detector.batchsize == 1
        @. detector.readouts = $rand!($devicerng(rng, means), $similar(means)) < -expm1(-means)
    else
        detector.readouts .= rand.(Ref(rng), Binomial.(detector.batchsize, -expm1.(-means)))
    end
    return detector
end
