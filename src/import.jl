# The binary files exclusively contain a list of pixel indices that have been activated. Ranging from 0 to 512×512-1, these indices are represented as 32-bit integers. Frames are separated by elements whose values are 512×512 in the list.

# Both signed and unsigned integers would work, as the number of pixels is much fewer than the upper limit of either type.

function readbin(path::AbstractString; width::Integer = 512, height::Integer = 512)
    files = checkpath(path)
    printfilelist(files)
    signals = _readbin(files)
    return signal2indices!(signals, width * height)
end

checkpath(path::AbstractString) =
    if isbinfile(path)
        [path]
    elseif isbindir(path)
        listbins(path)
    else
        throw(ErrorException("Cannot find any binary file in $path."))
    end

isbinfile(path::AbstractString) = isfile(path) && endswith(path, ".bin")

isbindir(path::AbstractString) = isdir(path) && any(f -> endswith(f, ".bin"), readdir(path))

listbins(dir::AbstractString) = filter(f -> endswith(f, ".bin"), readdir(dir, join = true))

function printfilelist(files::AbstractVector{<:AbstractString})
    println("Found the following binary file(s):")
    for f in files
        println(f)
    end
end

function _readbin(files::AbstractVector{String})
    number_of_signals = countsignals(files)
    signals = Vector{Int32}(undef, sum(number_of_signals))
    readsignals!(files, signals, number_of_signals)
    return signals
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

function signal2indices!(indices::Vector{<:Integer}, framesize::Integer)
    delimiters = popdelimiters!(indices, framesize)
    indices = convert(Vector{Int}, indices)
    return shiftindices!(indices, delimiters, framesize)
end

function popdelimiters!(signals::Vector{<:Integer}, framesize::Integer)
    delimiters = findall(>(framesize), signals)
    delimiters[end] != length(signals) && @warn "The last signal is not a delimiter."
    deleteat!(signals, delimiters)
    delimiters .-= 1:length(delimiters)
    return delimiters
end

function shiftindices!(
    indices::Vector{<:Integer},
    delimiters::Vector{<:Integer},
    framesize::Integer,
)
    @inbounds for i = 1:length(delimiters)-1
        indices[delimiters[i]+1:delimiters[i+1]] .+= i * framesize
    end
    return indices
end

function getframes(
    indices::AbstractVector{<:Integer};
    width::Integer,
    height::Integer,
    batchsize::Integer,
)
    framesize = width * height
    count, framesleft = countframes(indices[end], framesize, batchsize)
    framesleft != 0 &&
        @warn "The last batch contains $framesleft frames, fewer than the batchsize provided."
    frames = _getframes(indices, framesize, batchsize, count)
    return reshape(frames, width, height, count)
end

function countframes(lastindex::Integer, framesize::Integer, batchsize::Integer)
    numofbinframe = fld1(lastindex, framesize)
    return ifelse(batchsize == 1, (numofbinframe, 0), divrem(numofbinframe, batchsize))
end

function _getframes(
    indices::AbstractVector{<:Integer},
    framesize::Integer,
    batchsize::Integer,
    count::Integer,
)
    numofpx = framesize * count
    if batchsize == 1
        frames = falses(numofpx)
        frames[indices] .= true
    else
        frames = counts(batchindices(indices, framesize, batchsize), 1:numofpx) # See StatsBase.counts
    end
    return frames
end

batchindices(indices::AbstractVector{<:Integer}, framesize::Integer, batchsize::Integer) =
    @. (indices - 1) ÷ (framesize * batchsize) * framesize + mod1(indices, framesize)

function getROIindices(
    indices::AbstractVector{<:Integer},
    ROIbounds::AbstractMatrix{<:Integer},
    width::Integer,
    height::Integer,
)
    newwidth, newheight =
        ROIbounds[2, 1] - ROIbounds[1, 1] + 1, ROIbounds[2, 2] - ROIbounds[1, 2] + 1
    cartesianindices = cartesianize(indices, width, width * height)
    ROIcartesianindices = extractROI!(cartesianindices, ROIbounds)
    return linearize(ROIcartesianindices, newwidth, newheight)
end

function cartesianize(
    indices::AbstractVector{<:Integer},
    width::Integer,
    framesize::Integer,
)
    cartesianindices = Matrix{eltype(indices)}(undef, length(indices), 3)
    return cartesianize!(cartesianindices, indices, width, framesize)
end

function cartesianize!(
    cartesianindices::AbstractMatrix{<:Integer},
    indices::AbstractVector{<:Integer},
    width::Integer,
    framesize::Integer,
)
    cartesianindices[:, 1] .= mod1.(indices, width)
    cartesianindices[:, 2] .= fld1.(mod1.(indices, framesize), width)
    cartesianindices[:, 3] .= fld1.(indices, framesize)
    return cartesianindices
end

function extractROI!(
    cartesianindices::AbstractMatrix{<:Integer},
    ROIbounds::AbstractMatrix{<:Integer},
)
    newupperbounds = view(ROIbounds, 1:1, :) .- 1
    cartesianindices .-= newupperbounds
    newupperbounds .= view(ROIbounds, 2:2, :) .- newupperbounds
    inROI = vec(all(0 .< cartesianindices .≤ newupperbounds, dims = 2))
    return cartesianindices[inROI, :]
end

function linearize(
    cartesianindices::AbstractMatrix{<:Integer},
    width::Integer,
    height::Integer,
)
    indices = similar(cartesianindices, axes(cartesianindices, 1))
    return linearize!(indices, cartesianindices, height, width)
end

function linearize!(
    indices::AbstractVector{<:Integer},
    cartesianindices::AbstractMatrix{<:Integer},
    width::Integer,
    height::Integer,
)
    @views @. indices =
        (cartesianindices[:, 3] - 1) * height * width +
        (cartesianindices[:, 2] - 1) * height +
        cartesianindices[:, 1]
    return indices
end