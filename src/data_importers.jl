# The binary files exclusively contain a list of pixel indices that have been activated. Ranging from 0 to 512×512-1, these indices are represented as 32-bit integers. Frames are separated by elements whose values are 512×512 in the list.

# Both signed and unsigned integers would work, as the number of pixels is much fewer than the upper limit of either type.

function readbin(
    path::AbstractString;
    framewidth::Integer = 512,
    frameheight::Integer = 512,
)
    files = checkpath(path)
    println("Found the following binary file(s):\n$files")
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
    return div.(sizes, 4)
end

iscorrupted(sizeinbyte) = rem(sizeinbyte, 4) != 0

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

get_cartesian_indices(
    linear::AbstractVector{<:Integer},
    height::Integer,
    frame_count::Integer,
) = fld1.(linear, height), mod1.(linear, height), fld1.(linear, frame_count)

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

# function form_frams1(i::AbstractArray{<:Integer})
#     f = falses(512, 512, 255)
#     f[i] .= true
#     return sum(f, dims = 3)
# end

# function form_frams2(i::AbstractArray{<:Integer})
#     framesize = 512^2
#     j = mod1.(i, framesize)
#     f = counts(j, 1:framesize)
#     return reshape(f, 512, 512)
# end

function indices2video(
    linear_indices::AbstractVector{<:Integer};
    framewidth::Integer,
    frameheight::Integer,
    bits::Integer,
)
    checkbits(bits)
    return indices2video(
        linear_indices,
        framewidth = framewidth,
        frameheight = frameheight,
        merge_count = bits2number(bits),
    )
end

function checkbits(bits::Integer)
    bits <= 0 && throw(DomainError(bits, "The number of bits must be positive."))
    bits > Sys.WORD_SIZE &&
        @warn "The number of bits is greater than the system word size " *
              string(Sys.WORD_SIZE) *
              " overflow may happen."
end

bits2number(bits::Integer) = 2^bits - 1

function indices2video(
    linear_indices_1bit::AbstractVector{<:Integer};
    framewidth::Integer,
    frameheight::Integer,
    merge_count::Integer,
)
    px_per_frame = framewidth * frameheight
    frames, frame_count = get_merged_frames(linear_indices_1bit, px_per_frame, merge_count)
    frames = reshape(frames, framewidth, frameheight, frame_count)
end

function get_merged_frames(
    linear_indices_1bit::AbstractVector{<:Integer},
    px_per_frame::Integer,
    merge_count::Integer,
)
    number_of_1bit_frames = get_1bit_frame_count(linear_indices_1bit[end], px_per_frame)
    frame_count = get_merged_frame_count(number_of_1bit_frames, merge_count)
    linear_indices = rearrange_indices(linear_indices_1bit, px_per_frame, merge_count)
    frames = counts(linear_indices, 1:px_per_frame*frame_count)
    return frames, frame_count
end

get_1bit_frame_count(last_signal_index::Integer, px_per_frame::Integer) =
    fld1(last_signal_index, px_per_frame)

function get_merged_frame_count(frame_count_1bit::Integer, merge_count::Integer)
    iszero(frame_count_1bit % merge_count) &&
        @warn "The number of binary frames is not divisible by the merge count provided."
    return frame_count_1bit ÷ merge_count
end

rearrange_indices(
    indices::AbstractVector{<:Integer},
    px_per_frame::Integer,
    merge_count::Integer,
) = @. (indices - 1) ÷ (px_per_frame * merge_count) * px_per_frame +
   mod1(indices, px_per_frame)