getdarkrates(frames::AbstractArray{<:Integer}) =
    dropdims(sum(frames, dims = 3) ./ size(frames, 3), dims = 3)
