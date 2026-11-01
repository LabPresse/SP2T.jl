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

# Most SPAD readouts are 0, so `logexpm1` is evaluated only where it contributes: a real
# branch (not `ifelse`) is 5× faster on CPU and up to 3.5× on Metal.
clickterm(r::Integer, λ::T) where {T} = iszero(r) ? zero(T) : r * logexpm1(λ)
Δclickterm(r::Integer, λ₁::T, λ₂::T) where {T} =
    iszero(r) ? zero(T) : r * (logexpm1(λ₂) - logexpm1(λ₁))

function set_pixel_loglikelihood!(
    llarray::LogLikelihoodArray{T},
    detector::SPAD{T},
) where {T}
    @. llarray.pixel =
        clickterm(detector.readouts, llarray.means[1]) -
        detector.batchsize * llarray.means[1]
    return llarray
end

function set_pixel_Δloglikelihood!(
    Δllarray::LogLikelihoodArray{T},
    detector::SPAD{T},
) where {T}
    @. Δllarray.pixel =
        Δclickterm(detector.readouts, Δllarray.means[1], Δllarray.means[2]) -
        detector.batchsize * (Δllarray.means[2] - Δllarray.means[1])
    return Δllarray
end

"""
    SPADLogLikelihood{T} <: LogLikelihoodAux{T}

The SPAD log-likelihood, evaluated only where it is nonlinear. Per pixel it is `r·logexpm1(λ) − B·λ`. The first term vanishes where the readout `r` is 0, and the second, summed over a frame's pixels, separates: `Σ f·λ = Σ f·darkcounts + (b/A) Σₘ pxₘᵀ F pyₘ`, with `pxₘ`, `pyₘ` emitter `m`'s pixel-integrated PSF along x and y and `F` the filter. So `means[i]` holds the expected counts only at the clicks (nonzero, unfiltered readouts), and `linear[i]` each frame's `Σ f·(λ − darkcounts)`. The results equal those of `LogLikelihoodArray` up to rounding.

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
    darksum::T # Σ f·darkcounts over a frame
    means::NTuple{2,A}
    linear::NTuple{2,V}
    pixel::A
    frame::V
end

function SPADLogLikelihood(detector::SPAD{T}) where {T}
    # indexed on the CPU, once per run
    readouts, filter = Array(detector.readouts), Array(detector.filter)
    darkcounts = Array(detector.darkcounts)
    nx, ny, nframes = size(readouts)
    clicks = [findall((readouts[:, :, k] .> 0) .& (filter .!= 0)) for k = 1:nframes]
    nrows = max(1, maximum(length, clicks)) # Metal cannot launch kernels over empty arrays
    x, y = ones(Int, nrows, nframes), ones(Int, nrows, nframes)
    r, w = zeros(UInt16, nrows, nframes), zeros(T, nrows, nframes)
    for k = 1:nframes, (j, c) in enumerate(clicks[k])
        x[j, k], y[j, k] = Tuple(c)
        r[j, k], w[j, k] = readouts[c, k], filter[c]
    end
    d = darkcounts[CartesianIndex.(x, y)]
    frameoffsets = (0:nframes-1)'
    todevice(a) = copyto!(similar(detector.darkcounts, eltype(a), size(a)), a)
    return SPADLogLikelihood(
        todevice(r),
        todevice(w),
        todevice(d),
        todevice(x .+ nx .* frameoffsets),
        todevice(y .+ ny .* frameoffsets),
        all(isone, filter) ? nothing : detector.filter,
        convert(T, sum(filter .* darkcounts)),
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

# Add `scale · pxᵀ F py` of each frame to `linear`; with no pixel filtered out, F is all
# ones and this is `scale · Σpx · Σpy`.
addframesums!(
    linear::AbstractVector{T},
    px::AbstractArray{T,3},
    py::AbstractArray{T,3},
    ::Nothing,
    scale::T,
) where {T} = linear .+= scale .* vec(sum(px, dims = 1)) .* vec(sum(py, dims = 1))

function addframesums!(
    linear::AbstractVector{T},
    px::AbstractArray{T,3},
    py::AbstractArray{T,3},
    filter::AbstractMatrix{T},
    scale::T,
) where {T}
    Px, Py = reshape(px, size(px, 1), :), reshape(py, size(py, 1), :)
    # zeroed first: Metal's mul! keeps a NaN or Inf already in the output
    FPy = mul!(fill!(similar(Px), zero(T)), filter, Py)
    return linear .+= scale .* vec(sum(Px .* FPy, dims = 1))
end

function add_poisson_mean!(
    llarray::SPADLogLikelihood{T},
    detector::SPAD{T},
    tracksᵥ::AbstractArray{T,3},
    brightnessᵥ::T,
    psf::PointSpreadFunction{T},
    i::Integer,
) where {T}
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
        llarray.means[i] .+= scale .* px[llarray.xindex] .* py[llarray.yindex]
        addframesums!(llarray.linear[i], px, py, llarray.filter, scale)
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
    ratio = brightnessᵥ₂ / brightnessᵥ₁
    llarray.means[2] .=
        llarray.darkcounts .+ (llarray.means[1] .- llarray.darkcounts) .* ratio
    llarray.linear[2] .= llarray.linear[1] .* ratio
    return llarray
end

function get_loglikelihood!(llarray::SPADLogLikelihood{T}, detector::SPAD{T}) where {T}
    @. llarray.pixel = llarray.weights * clickterm(llarray.readouts, llarray.means[1])
    sum!(reshape(llarray.frame, 1, :), llarray.pixel)
    llarray.frame .-= detector.batchsize .* (llarray.darksum .+ llarray.linear[1])
    return sum(llarray.frame)
end

function set_frame_Δloglikelihood!(
    llarray::SPADLogLikelihood{T},
    detector::SPAD{T},
) where {T}
    @. llarray.pixel =
        llarray.weights *
        Δclickterm(llarray.readouts, llarray.means[1], llarray.means[2])
    sum!(reshape(llarray.frame, 1, :), llarray.pixel)
    llarray.frame .-= detector.batchsize .* (llarray.linear[2] .- llarray.linear[1])
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