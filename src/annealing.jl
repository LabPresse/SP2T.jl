"""
    Annealing{T}

An abstract type representing a generic annealing schedule. This serves as a base type for defining specific annealing implementations. The type parameter `T` can be used to specify the type of data.
"""
abstract type Annealing{T} end

"""
    ConstantAnnealing{T} <: Annealing{T}

A struct that represents a constant annealing schedule (fixed temperature `𝑇`).
"""
struct ConstantAnnealing{T} <: Annealing{T}
    𝑇::T
end

"""
    PolynomialAnnealing{T} <: Annealing{T}

A struct that represents a polynomial annealing schedule. Its fields include the initial temperature `𝑇₀`, the iteration number `I` when the temperature becomes and stays at one, and the `order` of the polynomial. Before iteration `I`, the temperature is `1 + (𝑇₀ - 1) * (1 - i / I)^order`.
"""
struct PolynomialAnnealing{T} <: Annealing{T}
    𝑇₀::T
    I::T
    order::Int
end

PolynomialAnnealing{T}() where {T} = PolynomialAnnealing{T}(1, 1, 0)

"""
    temperature(a::Annealing, i::Real)

Return the temperature given the annealing schedule `a` and iteration number `i`.
"""
temperature(a::ConstantAnnealing, i::Real) = a.𝑇
temperature(a::PolynomialAnnealing{T}, i::Real) where {T} =
    i >= a.I ? one(T) : one(T) + (a.𝑇₀ - one(T)) * (one(T) - i / a.I)^a.order

anneal!(ll::AbstractArray{T}, 𝑇::T) where {T} = ll ./= 𝑇