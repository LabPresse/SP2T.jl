# The binary files exclusively contain a list of pixel indices that have been activated. Ranging from 0 to 512×512-1, these indices are represented as 32-bit integers. Frames are separated by elements whose values are 512×512 in the list.

# Both signed and unsigned integers would work, as the number of pixels is much fewer than the upper limit of either type.

function read_file(path::String; framewidth::Integer, frameheight::Integer)
    file = isfile(path) ? [path] : throw(SystemError(path))
    linear_indices = read_binary_files(file, framewidth, frameheight)
    return linear_indices
end

# full_indices2 = extract_ROI(full_indices, ROIbounds)
# frames = falses((diff(ROIbounds, dims = 1) .+ 1)...)
# form_frames!(frames, full_indices2)

function read_dir(path::String, type::String; framewidth::Integer, frameheight::Integer)
    files =
        isdir(path) ? get_file_list(path::String, type::String) : throw(SystemError(path))
    print("Found the following $type files in $path:\n$files")
    linear_indices = read_binary_files(files, framewidth, frameheight)
    return linear_indices
end

# full_indices2 = extract_ROI(full_indices, ROIbounds)
# frames = falses((diff(ROIbounds, dims = 1) .+ 1)...)
# form_frames!(frames, full_indices2)

function read_files(path::String, type::String; framewidth::Integer, frameheight::Integer)
    files = get_file_list(path, type)
    read_binary_files(files, framewidth, frameheight)
    full_indices2 = extract_ROI(full_indices, ROIbounds)
    frames = falses((diff(ROIbounds, dims = 1) .+ 1)...)
    form_frames!(frames, full_indices2)
    return frames
end

# function read_files(
#     path::String,
#     type::String;
#     framewidth::Integer,
#     frameheight::Integer,
#     ROIbounds::AbstractMatrix{T} = [1 1 1; typemax(Int) typemax(Int) typemax(Int)],
# ) where {T<:Integer}
#     files = get_file_list(path, type)
#     full_indices = if type == "bin"
#         read_binary_files(files; width = framewidth, height = frameheight)
#     end
#     ROIbounds[2, 1] = min(ROIbounds[2, 1], framewidth)
#     ROIbounds[2, 2] = min(ROIbounds[2, 2], frameheight)
#     ROIbounds[2, 3] = min(ROIbounds[2, 3], full_indices[end-1, 4])
#     full_indices2 = extract_ROI(full_indices, ROIbounds)
#     frames = falses((diff(ROIbounds, dims = 1) .+ 1)...)
#     form_frames!(frames, full_indices2)
#     return frames
# end

# function get_file_list(path::String, type::String)
#     files = if isfile(path)
#         get_file_from_file(path, type)
#     elseif isdir(path)
#         get_file_from_dir(path, type)
#     else
#         error("No binary (.$type) files in the input path")
#     end
#     return files
# end

# get_file_from_file(path::String, type::String) =
#     if endswith(path, "." * type)
#         [path]
#     else
#         error("No .$type files in $path")
#     end

function get_file_list(dir::String, type::String)
    files = filter(f -> endswith(f, "." * type), readdir(dir))
    if isempty(files)
        error_prefix = dir * (endswith(dir, "/") ? "*.$type" : "/*.$type")
        throw(SystemError(error_prefix))
    end
    return files
end

function read_binary_files(files::AbstractVector{String}, width::Integer, height::Integer)
    number_of_signals = count_signals(files)
    signals = Vector{UInt32}(undef, sum(number_of_signals))
    read_signals!(files, signals, number_of_signals)
    linear_indices = signal2linear!(signals, width * height)
    return linear_indices
end

# width_indices = fld1.(linear_indices, height)
# height_indices = mod1.(linear_indices, height)
# frame_indices = fld1.(linear_indices, framesize)
# , width_indices, height_indices, frame_indices

function count_signals(files::AbstractVector{String})
    sizes = filesize.(files) # sizes in bytes
    number_of_signals = Vector{Int}(undef, length(files))
    number_of_signals .= div.(sizes, 4)
    checksize(files, sizes)
    return number_of_signals
end

function checksize(files::AbstractVector{String}, sizes::AbstractVector{<:Integer})
    corrupted = findfirst(!=(0), rem.(sizes, 4))
    isnothing(corrupted) || throw(
        DomainError(
            sizes[corrupted],
            "File size is not divisible by 4 in $files[$corrupted]",
        ),
    )
end

function read_signals!(
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
    delimiters = pop_delimiters!(signals, framesize)
    linear = convert(Vector{Int}, signals)
    fill_linear!(linear, delimiters, framesize)
    return linear
end

function pop_delimiters!(signals::Vector{<:Integer}, framesize::Integer)
    delimiters = findall(>(framesize), signals)
    delimiters[end] != length(signals) && @warn "The last signal is not a delimiter."
    deleteat!(signals, delimiters)
    delimiters .-= 1:length(delimiters)
    return delimiters
end

function fill_linear!(
    linear::Vector{<:Integer},
    delimiters::Vector{<:Integer},
    framesize::Integer,
)
    @inbounds for i = 1:length(delimiters)-1
        linear[delimiters[i]+1:delimiters[i+1]] .+= i * framesize
    end
    return linear
end

get_cartesian_indices(
    linear::AbstractVector{<:Integer},
    height::Integer,
    frame_count::Integer,
) = fld1.(linear, height), mod1.(linear, height), fld1.(linear, frame_count)

# get_width_indices(linear::AbstractVector{<:Integer}, height::Integer) =
#     fld1.(linear, height)

# get_height_indices(linear::AbstractVector{<:Integer}, height::Integer) =
#     mod1.(linear, height)

# function get_frame_indices(
#     linear::AbstractVector{<:Integer},
#     delimiters::AbstractVector{<:Integer},
# )
#     frame_indices = similar(linear)
#     @inbounds for i = 1:length(delimiters)-1
#         frame_indices[delimiters[i]+1:delimiters[i+1]] .= i
#     end
#     return frame_indices
# end

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