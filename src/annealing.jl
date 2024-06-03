abstract type AbstractAnnealing{T} end

struct NoAnnealing{T} <: AbstractAnnealing{T} end

NoAnnealing() = NoAnnealing{Int}()

temperature(::NoAnnealing{T}, i) where {T} = one(T)

struct PolynomialAnnealing{T} <: AbstractAnnealing{T}
    init_temperature::T
    last_iter::T
    order::Int
end

PolynomialAnnealing{T}() where {T} = PolynomialAnnealing{T}(1, 1, 0)

temperature(a::PolynomialAnnealing{T}, i) where {T} =
    i >= a.last_iter ? one(T) : a.init_temperature * (i / a.last_iter - oneunit(T))^a.order

ftypof(::PolynomialAnnealing{T}) where {T} = T

ftypof(::Nothing) = Nothing

# PolynomialAnnealing{T}(
#     init_temperature::Real = 1,
#     cutoff_iteration::Real = 1,
#     order::Integer = 2,
# ) where {T<:AbstractFloat} = get_temperature(
#     PolynomialAnnealing{T}(zero(T), init_temperature, cutoff_iteration, order),
#     1,
# )
