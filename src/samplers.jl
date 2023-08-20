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

randg(k::Real) = rand(Gamma(k, 1))
randg(k::Real, N::Integer) = rand(Gamma(k, 1), N)

randig(k::Real, θ::Real) = θ / randg(k)
randig(k::Real, θ::Real, N::Integer) = θ ./ randg(k, N)