ftypof(::AbstractAnnealing{T}) where {T} = T

struct ConstantAnnealing{T} <: AbstractAnnealing{T}
    temperature::T
end

struct PolynomialAnnealing{T} <: AbstractAnnealing{T}
    init_temperature::T
    last_iter::T
    order::Int
end

PolynomialAnnealing{T}() where {T} = PolynomialAnnealing{T}(1, 1, 0)

temperature(a::ConstantAnnealing, i::Real) = a.temperature

temperature(a::PolynomialAnnealing{T}, i::Real) where {T} =
    i >= a.last_iter ? one(T) : a.init_temperature * (i / a.last_iter - oneunit(T))^a.order
