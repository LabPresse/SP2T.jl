function randc(p::AbstractArray{<:Real})
    r = rand() - p[1]
    c = 1
    while r > 0
        c += 1
        r -= p[c]
    end
    return c
end

function randc!(s::AbstractArray{<:Integer}, p::AbstractArray{<:Real})
    for i in eachindex(s)
        s[i] = randc(p)
    end
    return s
end

function randb(p::Real)
    return rand() < p
end