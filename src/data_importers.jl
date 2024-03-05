# The binary files exclusively contain a list of pixel indices that have been activated. Ranging from 0 to 512×512-1, these indices are represented as 32-bit integers. Frames are separated by elements whose values are 512×512 in the list.

# Both signed and unsigned integers would work, as the number of pixels is much fewer than the upper limit of either type.

function read_files(
    path::String,
    type::String;
    framewidth::Integer,
    frameheight::Integer,
    ROIbounds::AbstractMatrix{T} = [1 1 1; typemax(Int) typemax(Int) typemax(Int)],
) where {T<:Integer}
    files = get_file_list(path, type)
    full_indices = if type == "bin"
        read_binary_files(files; width = framewidth, height = frameheight)
    end
    ROIbounds[2, 1] = min(ROIbounds[2, 1], framewidth)
    ROIbounds[2, 2] = min(ROIbounds[2, 2], frameheight)
    ROIbounds[2, 3] = min(ROIbounds[2, 3], full_indices[end-1, 4])
    full_indices2 = extract_ROI(full_indices, ROIbounds)
    frames = falses((diff(ROIbounds, dims = 1) .+ 1)...)
    form_frames!(frames, full_indices2)
    return frames
end

function read_files(
    path::String,
    type::String;
    framewidth::Integer,
    frameheight::Integer,
    ROIbounds::AbstractMatrix{T} = [1 1 1; typemax(Int) typemax(Int) typemax(Int)],
) where {T<:Integer}
    files = get_file_list(path, type)
    full_indices = if type == "bin"
        read_binary_files(files; width = framewidth, height = frameheight)
    end
    ROIbounds[2, 1] = min(ROIbounds[2, 1], framewidth)
    ROIbounds[2, 2] = min(ROIbounds[2, 2], frameheight)
    ROIbounds[2, 3] = min(ROIbounds[2, 3], full_indices[end-1, 4])
    full_indices2 = extract_ROI(full_indices, ROIbounds)
    frames = falses((diff(ROIbounds, dims = 1) .+ 1)...)
    form_frames!(frames, full_indices2)
    return frames
end

function get_file_list(path::String, type::String)
    files = if isfile(path)
        get_file_from_file(path, type)
    elseif isdir(path)
        get_file_from_dir(path, type)
    else
        error("No binary (.$type) files in the input path")
    end
    return files
end

get_file_from_file(path::String, type::String) =
    if endswith(path, "." * type)
        [path]
    else
        error("No .$type files in $path")
    end

function get_file_from_dir(path::String, type::String)
    files = filter(files -> endswith(files, "." * type), readdir(path))
    isempty(files) && error("No .$type files in $path")
    return files
end

function read_binary_files(files::AbstractVector{String}; width::Integer, height::Integer)
    number_of_signals = count_signals(files)
    signal_indices = Matrix{UInt32}(undef, sum(number_of_signals), 4)
    fill_signal_indices!(signal_indices, files, number_of_signals)
    linear2cartesian!(signal_indices; width, height)
end

function count_signals(files::AbstractVector{String})
    number_of_bytes = filesize.(files)
    number_of_signals = Vector{Int}(undef, length(files))
    number_of_signals .= div.(number_of_bytes, 4)
    corrupted = findfirst(!=(0), rem.(number_of_bytes, 4))
    isnothing(corrupted) || throw(
        DomainError(
            number_of_signals[corrupted],
            "File size is not divisible by 4 in $files[$corrupted]",
        ),
    )
    return number_of_signals
end

function fill_signal_indices!(
    signal_indices::AbstractMatrix{<:Integer},
    files::AbstractVector{String},
    number_of_signals::AbstractVector{<:Integer},
)
    start = 1
    @inbounds for (file, number) in zip(files, number_of_signals)
        read!(file, view(signal_indices, range(start, length = number), 1))
        start += number
    end
    signal_indices[:, 1] .+= 1
    return signal_indices
end

function linear2cartesian!(
    signal_indices::AbstractMatrix{<:Integer};
    width::Integer,
    height::Integer,
)
    number_of_pixels = height * width
    if signal_indices[end, 1] <= number_of_pixels
        @warn "The last signal is not a delimiter."
    end
    set_width_indices!(view(signal_indices, :, 2), view(signal_indices, :, 1), height)
    set_height_indices!(view(signal_indices, :, 3), view(signal_indices, :, 1), height)
    set_frame_indices!(
        view(signal_indices, :, 4),
        view(signal_indices, :, 1),
        number_of_pixels,
    )
    return signal_indices
end

function set_width_indices!(
    width_indices::AbstractVector{<:Integer},
    linear::AbstractVector{<:Integer},
    height::Integer,
)
    width_indices .= fld1.(linear, height)
    return width_indices
end

function set_height_indices!(
    height_indices::AbstractVector{<:Integer},
    linear::AbstractVector{<:Integer},
    height::Integer,
)
    height_indices .= mod1.(linear, height)
    return height_indices
end

function set_frame_indices!(
    frame_indices::AbstractVector{<:Integer},
    signals::AbstractVector{<:Integer},
    number_of_pixels::Integer,
)
    frame_index = one(eltype(frame_indices))
    @inbounds for signal_index in eachindex(signals)
        if signals[signal_index] > number_of_pixels
            frame_index += 1
        end
        frame_indices[signal_index] = frame_index
    end
    return frame_indices
end

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