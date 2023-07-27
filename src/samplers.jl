function randc(p)
    i = 1
    c = p[1]
    u = rand()
    while c < u
        c += p[i+=1]
    end
    return i
end

function randc!(s::AbstractArray{<:Integer}, p::AbstractArray{<:Real})
    for i in eachindex(s)
        s[i] = randc(p)
    end
    return s
end

randb(p) = rand() < p