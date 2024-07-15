abstract type AbstractAnnealing{T} end

struct NoAnnealing{T} <: AbstractAnnealing{T} end

NoAnnealing() = NoAnnealing{Float32}()

temperature(::NoAnnealing{T}, i) where {T} = one(T)

struct PolynomialAnnealing{T} <: AbstractAnnealing{T}
    init_temperature::T
    last_iter::T
    order::Int
end

PolynomialAnnealing{T}() where {T} = PolynomialAnnealing{T}(1, 1, 0)

temperature(a::PolynomialAnnealing{T}, i::Real) where {T} =
    i >= a.last_iter ? one(T) : a.init_temperature * (i / a.last_iter - oneunit(T))^a.order

ftypof(::PolynomialAnnealing{T}) where {T} = T

# ftypof(::Nothing) = Nothing