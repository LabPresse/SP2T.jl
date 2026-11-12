# One block owns one frame. Updating clicked means and summing the intensity
# at zero readouts together avoids temporary gathers and per-tile launches.
const SPAD_WARPS_PER_BLOCK = 4

function SP2T.add_spad_signal!(
    llarray::SP2T.SPADLogLikelihood{T}, detector::SPAD{T},
    px::CuArray{T,3}, py::CuArray{T,3}, scale::T, i::Integer,
) where {T<:Union{Float32,Float64}}
    no_zeros = length(llarray.zeroruns) == 1 && only(llarray.zeroruns)[2] === :none
    threads = 32 * SPAD_WARPS_PER_BLOCK
    CUDA.@cuda threads=threads blocks=size(px, 3) add_spad_signal_kernel!(
        llarray.means[i], llarray.linear[i], px, py, scale,
        llarray.xindex, llarray.yindex, llarray.weights, detector.readouts,
        llarray.filter, no_zeros,
    )
    return llarray
end

@inline function warp_sum(value)
    for offset in (16, 8, 4, 2, 1)
        value += CUDA.shfl_down_sync(0xffffffff, value, offset)
    end
    return value
end

function add_spad_signal_kernel!(
    means, linear, px, py, scale::T, xindex, yindex, weights, readouts, filter, no_zeros,
) where {T}
    k = CUDA.blockIdx().x
    thread = CUDA.threadIdx().x
    width = CUDA.warpsize()
    lane = (thread - Int32(1)) % width + Int32(1)
    warp = (thread - Int32(1)) ÷ width + Int32(1)
    @inbounds for h in thread:CUDA.blockDim().x:size(means, 1)
        means[h, k] += scale * px[xindex[h, k]] * py[yindex[h, k]]
    end
    no_zeros && return

    nx, ny, _ = size(readouts)
    if isnothing(filter) && iszero(@inbounds weights[1, k])
        # A frame with no clicks and no filter has separable zero-readout mass.
        if warp == 1
            xsum, ysum = zero(T), zero(T)
            @inbounds for x in lane:width:nx
                xsum += px[x, 1, k]
            end
            @inbounds for y in lane:width:ny
                ysum += py[y, 1, k]
            end
            mass = warp_sum(xsum) * warp_sum(ysum)
            if lane == 1
                @inbounds linear[k] += SP2T._spad_scale_mass(scale, mass)
            end
        end
        return
    end

    # Sum only included zero readouts. In particular, subtracting the signal
    # at clicks from the total could erase a small positive saturated tail.
    mass = zero(T)
    nwarps = CUDA.blockDim().x ÷ width
    @inbounds for y in warp:nwarps:ny, x in lane:width:nx
        if iszero(readouts[x, y, k])
            weight = isnothing(filter) ? one(T) : filter[x, y]
            if !iszero(weight)
                mass += weight * px[x, 1, k] * py[y, 1, k]
            end
        end
    end
    mass = warp_sum(mass)
    partial = CUDA.CuStaticSharedArray(T, SPAD_WARPS_PER_BLOCK)
    if lane == 1
        @inbounds partial[warp] = mass
    end
    CUDA.sync_threads()
    if thread == 1
        total = zero(T)
        @inbounds for w in 1:SPAD_WARPS_PER_BLOCK
            total += partial[w]
        end
        @inbounds linear[k] += SP2T._spad_scale_mass(scale, total)
    end
    return
end
