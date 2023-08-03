abstract type Annealing end

struct PolynomialAnnealing{FT<:AbstractFloat} <: Annealing
    T₀::FT
    𝑖::FT
    order::Int
    PolynomialAnnealing{FT}(
        init_temperature::Real = 1,
        cutoff_iteration::Real = 1,
        order::Integer = 2,
    ) where {FT<:AbstractFloat} = new{FT}(init_temperature, cutoff_iteration, order)
end

function get_temperature(i::Integer, a::PolynomialAnnealing)
    FT = fieldtype(a, :T₀)
    if i >= a.𝑖
        return FT(1)
    else
        return a.T₀ * (i / a.𝑖 - 1)^a.order
    end
end

ftypof(p::PolynomialAnnealing{FT}) where {FT} = FT
ftypof(p::Nothing) = Nothing