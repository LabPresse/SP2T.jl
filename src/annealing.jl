mutable struct PolynomialAnnealing{T<:AbstractFloat} <: AbstractAnnealing{T}
    temperature::T
    init_temperature::T
    last_iter::T
    order::Int
end

PolynomialAnnealing{T}(
    init_temperature::Real = 1,
    cutoff_iteration::Real = 1,
    order::Integer = 2,
) where {T<:AbstractFloat} = set_temperature!(
    PolynomialAnnealing{T}(zero(T), init_temperature, cutoff_iteration, order),
    1,
)

function set_temperature!(a::PolynomialAnnealing{T}, i::Integer) where {T}
    a.temperature = ifelse(
        i >= a.last_iter,
        oneunit(T),
        a.init_temperature * (i / a.last_iter - oneunit(T))^a.order,
    )
    return a
end

ftypof(::PolynomialAnnealing{T}) where {T} = T

ftypof(::Nothing) = Nothing