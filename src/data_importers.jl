# The binary files exclusively contain a list of pixel indices that have been activated. Ranging from 0 to 512×512-1, these indices are represented as 32-bit integers. Frames are separated by elements whose values are 512×512 in the list.

# Both signed and unsigned integers would work, as the number of pixels is much fewer than the upper limit of either type.

function readbin(
    path::AbstractString;
    framewidth::Integer = 512,
    frameheight::Integer = 512,
)
    files = checkpath(path)
    println("Found the following binary file(s):")
    for f in files
        println(f)
    end
    return _readbin(files, framewidth, frameheight)
end

# function readdir(path::AbstractString; framewidth::Integer, frameheight::Integer)
#     files = isdir(path) ? listbins(path::AbstractString) : throw(SystemError(path))
#     println("Found the following binary files in $path:\n$files")
#     return readbins(files, framewidth, frameheight)
# end

checkpath(path) =
    if isbinfile(path)
        [path]
    elseif isbindir(path)
        listbins(path::AbstractString)
    else
        throw(ErrorException("Cannot find any binary file in $path."))
    end

isbinfile(path) = isfile(path) && endswith(path, ".bin")

isbindir(path) = isdir(path) && any(f -> endswith(f, ".bin"), readdir(path))

listbins(dir) = filter(f -> endswith(f, ".bin"), readdir(dir, join = true))

function _readbin(files::AbstractVector{String}, width::Integer, height::Integer)
    number_of_signals = countsignals(files)
    signals = Vector{UInt32}(undef, sum(number_of_signals))
    readsignals!(files, signals, number_of_signals)
    return signal2linear!(signals, width * height)
end

function countsignals(files::AbstractVector{String})
    sizes = filesize.(files) # sizes in bytes
    corrupted = iscorrupted.(sizes)
    any(corrupted) && throw(
        DomainError(
            sizes[corrupted],
            "File size is not divisible by 4 in $(files[corrupted])",
        ),
    )
    return sizes .÷ 4
end

iscorrupted(sizeinbyte) = sizeinbyte % 4 != 0

function readsignals!(
    files::AbstractVector{String},
    signals::Vector{<:Integer},
    number_of_signals::AbstractVector{<:Integer},
)
    start = 1
    @inbounds for (file, number) in zip(files, number_of_signals)
        read!(file, view(signals, range(start, length = number)))
        start += number
    end
    return signals .+= 1
end

function signal2linear!(signals::Vector{<:Integer}, framesize::Integer)
    frameindices = popdelimiters!(signals, framesize)
    indices = convert(Vector{Int}, signals)
    addframeshift!(indices, frameindices, framesize)
    return indices
end

function popdelimiters!(signals::Vector{<:Integer}, framesize::Integer)
    indices = findall(>(framesize), signals)
    indices[end] != length(signals) && @warn "The last signal is not a delimiter."
    deleteat!(signals, indices)
    indices .-= 1:length(indices)
    return indices
end

function addframeshift!(
    indices::Vector{<:Integer},
    delimiters::Vector{<:Integer},
    framesize::Integer,
)
    @inbounds for i = 1:length(delimiters)-1
        indices[delimiters[i]+1:delimiters[i+1]] .+= i * framesize
    end
    return indices
end

indices2video(
    indices::AbstractVector{<:Integer};
    framewidth::Integer = 512,
    frameheight::Integer = 512,
    bits::Integer = 8,
) = indices2video(
    indices,
    framewidth = framewidth,
    frameheight = frameheight,
    batchsize = getbatchsize(bits),
)

getbatchsize(bits::Integer)::Integer = 2^bits - 1

function indices2video(
    indices::AbstractVector{<:Integer};
    framewidth::Integer = 512,
    frameheight::Integer = 512,
    batchsize::Integer = 255,
)
    framesize = framewidth * frameheight
    framecount, framesleft = countframes(indices[end], framesize, batchsize)
    @show framesleft
    framesleft != 0 &&
        @warn "The number of binary frames is not divisible by the merge count provided."
    frames = mergebinframes(indices, framesize, batchsize, framecount)
    return reshape(frames, framewidth, frameheight, framecount)
end

countframes(lastindex::Integer, framesize::Integer, batchsize::Integer) =
    divrem(fld1(lastindex, framesize), batchsize)

mergebinframes(
    indices::AbstractVector{<:Integer},
    framesize::Integer,
    batchsize::Integer,
    framecount::Integer,
) = counts(batchindices(indices, framesize, batchsize), 1:framesize*framecount) # See StatsBase.counts

batchindices(indices::AbstractVector{<:Integer}, framesize::Integer, batchsize::Integer) =
    @. (indices - 1) ÷ (framesize * batchsize) * framesize + mod1(indices, framesize)

get_cartesian_indices(
    linear::AbstractVector{<:Integer},
    height::Integer,
    framecount::Integer,
) = fld1.(linear, height), mod1.(linear, height), fld1.(linear, framecount)

function extract_ROI(
    full_indices::AbstractMatrix{<:Integer},
    ROIbounds::AbstractMatrix{<:Integer},
)
    inROI = trues(size(full_indices, 1))
    @views begin
        applybounds!(inROI, full_indices[:, 2], ROIbounds[:, 1])
        applybounds!(inROI, full_indices[:, 3], ROIbounds[:, 2])
        applybounds!(inROI, full_indices[:, 4], ROIbounds[:, 3])
    end
    return full_indices[inROI, :]
end

function applybounds!(
    inROI::AbstractVector{Bool},
    indices::AbstractVector{<:Integer},
    bounds::AbstractVector{<:Integer},
)
    @. inROI &= bounds[1] <= indices <= bounds[end]
    indices .-= (bounds[1] - 1)
    return inROI
end

function form_frames!(
    frames::AbstractArray{Bool,3},
    signal_indices::AbstractMatrix{<:Integer},
)
    height, width, ~ = size(frames)
    @views @. signal_indices[:, 1] =
        (signal_indices[:, 4] - 1) * height * width +
        (signal_indices[:, 3] - 1) * height +
        signal_indices[:, 2]
    pixel2frame!(frames, view(signal_indices, :, 1))
    return frames
end

function pixel2frame!(frame::AbstractArray{Bool,3}, pixels::AbstractVector{<:Integer})
    frame[pixels] .= true
    return frame
end